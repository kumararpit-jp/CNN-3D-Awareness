"""CNN-driven multi-view matching and sparse 3D reconstruction.

Pipeline
--------
1. Load the views (from bottle_images.zip or a folder), naturally sorted
   (bottle_1, bottle_2, ..., bottle_12) and resized.
2. Detect ORB keypoints in every view.
3. Describe the *same* keypoints in two ways:
     - ORB binary descriptors (hand-crafted baseline)
     - CNN descriptors: ImageNet-pretrained ResNet18 feature maps
       (layer1 + layer2 + layer3, image upscaled 2x) bilinearly sampled at
       each keypoint location
4. Match consecutive view pairs (mutual nearest neighbour + ratio test) and
   filter outliers geometrically with RANSAC on the fundamental matrix.
   For CNN descriptors the ratio test compares against the best candidate
   that is spatially far from the best match: neighbouring keypoints share
   almost the same coarse CNN feature, which is not a real ambiguity.
5. For the pair with the most CNN inliers, estimate the essential matrix with
   approximate intrinsics, recover the relative pose and triangulate the
   inliers into a sparse 3D point cloud (up to an unknown scale).
6. Save a per-pair comparison table (CSV) and an interactive Plotly HTML
   point cloud; optionally show it in Open3D.

Usage
-----
    python cnn_3d_awareness.py                      # uses bottle_images.zip
    python cnn_3d_awareness.py --images path/to/dir --max-side 800
    python cnn_3d_awareness.py --pair 3 4           # reconstruct a chosen pair
"""
import argparse
import csv
import io
import os
import re
import zipfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def natural_key(name):
    """Sort 'bottle_2' before 'bottle_10'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def load_images(source, max_side=640):
    """Load images from a .zip file or a directory, naturally sorted and resized."""
    items = []
    if os.path.isfile(source) and source.lower().endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            for name in zf.namelist():
                if name.lower().endswith(IMAGE_EXTS) and not name.startswith("__MACOSX"):
                    buf = np.frombuffer(zf.read(name), dtype=np.uint8)
                    items.append((os.path.basename(name), cv2.imdecode(buf, cv2.IMREAD_COLOR)))
    elif os.path.isdir(source):
        for name in os.listdir(source):
            if name.lower().endswith(IMAGE_EXTS):
                items.append((name, cv2.imread(os.path.join(source, name))))
    else:
        raise FileNotFoundError(f"No zip file or folder at: {source}")

    items = [(n, im) for n, im in items if im is not None]
    if len(items) < 2:
        raise ValueError("Need at least two images")
    items.sort(key=lambda t: natural_key(t[0]))

    names, images = [], []
    for name, img in items:
        scale = max_side / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        names.append(name)
        images.append(img)
    return names, images


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def detect_orb(images, n_features=3000):
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = [], []
    for img in images:
        kp, des = orb.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors


class ResNetFeatures(torch.nn.Module):
    """ImageNet-pretrained ResNet18 trunk returning layer1/2/3 maps (strides 4, 8, 16)."""

    DIM = 64 + 128 + 256

    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3

    def forward(self, x):
        f1 = self.layer1(self.stem(x))
        f2 = self.layer2(f1)
        return f1, f2, self.layer3(f2)


@torch.no_grad()
def cnn_descriptors(model, img, keypoints, device, upscale=2.0):
    """Sample L2-normalised ResNet18 features at keypoint locations.

    Each layer's sampled feature is normalised separately (so every scale
    contributes equally) and the concatenation is normalised again.
    """
    if not keypoints:
        return np.zeros((0, ResNetFeatures.DIM), dtype=np.float32)
    h, w = img.shape[:2]
    net_in = img if upscale == 1 else cv2.resize(img, None, fx=upscale, fy=upscale,
                                                 interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(cv2.cvtColor(net_in, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1)[None] / 255.0
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(device)
    maps = model(x)

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    # Pixel coords -> [-1, 1] grid coords (align_corners=False convention)
    grid = torch.from_numpy(np.stack([(pts[:, 0] + 0.5) / w * 2 - 1,
                                      (pts[:, 1] + 0.5) / h * 2 - 1], axis=1))
    grid = grid.view(1, 1, -1, 2).to(device)

    parts = []
    for fmap in maps:
        sampled = F.grid_sample(fmap, grid, mode="bilinear", align_corners=False)
        parts.append(F.normalize(sampled[0, :, 0, :].T, dim=1))   # (N, C)
    return F.normalize(torch.cat(parts, dim=1), dim=1).cpu().numpy()


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def match_orb(des1, des2, ratio=0.8):
    """Hamming kNN matching with Lowe's ratio test and a mutual (cross) check."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    fwd = bf.knnMatch(des1, des2, k=2)
    back = {m.queryIdx: m.trainIdx for m in bf.match(des2, des1)}
    out = []
    for pair in fwd:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance and back.get(m.trainIdx) == m.queryIdx:
            out.append((m.queryIdx, m.trainIdx))
    return out


def match_cnn(d1, d2, kp2, ratio=0.9, radius=16.0):
    """Cosine-similarity mutual nearest neighbours with a spatially aware ratio test.

    The second-best candidate is taken only among keypoints more than `radius`
    pixels away from the best match, because keypoints that are close together
    sample nearly the same coarse CNN feature.
    """
    if len(d1) < 2 or len(d2) < 2:
        return []
    sim = d1 @ d2.T
    nn12 = sim.argmax(1)
    nn21 = sim.argmax(0)
    p2 = np.float32([kp.pt for kp in kp2])
    out = []
    for i, j in enumerate(nn12):
        if nn21[j] != i:
            continue
        far = np.linalg.norm(p2 - p2[j], axis=1) > radius
        if not far.any():
            continue
        d_best = np.sqrt(max(2 - 2 * sim[i, j], 0.0))
        d_second = np.sqrt(max(2 - 2 * sim[i, far].max(), 0.0))
        if d_best < ratio * d_second:
            out.append((i, int(j)))
    return out


def ransac_inliers(kp1, kp2, matches, thresh=3.0):
    """Fundamental-matrix RANSAC; returns a boolean inlier mask."""
    if len(matches) < 8:
        return np.zeros(len(matches), dtype=bool)
    p1 = np.float32([kp1[i].pt for i, _ in matches])
    p2 = np.float32([kp2[j].pt for _, j in matches])
    _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, thresh, 0.999)
    if mask is None:
        return np.zeros(len(matches), dtype=bool)
    return mask.ravel().astype(bool)


# --------------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------------- #
def triangulate_pair(img1, kp1, kp2, matches, focal_scale=1.2):
    """Two-view reconstruction (up to scale) with approximate pinhole intrinsics.

    The camera is uncalibrated, so focal length is assumed to be
    focal_scale * image width with the principal point at the centre.
    """
    h, w = img1.shape[:2]
    f = focal_scale * w
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    p1 = np.float64([kp1[i].pt for i, _ in matches])
    p2 = np.float64([kp2[j].pt for _, j in matches])

    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    if E is None:
        raise RuntimeError("Essential matrix estimation failed")
    E = E[:3]
    _, R, t, pose_mask = cv2.recoverPose(E, p1, p2, K, mask=mask)
    keep = pose_mask.ravel() > 0

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    X = cv2.triangulatePoints(P1, P2, p1[keep].T, p2[keep].T)
    X = (X[:3] / X[3]).T

    # Keep points in front of both cameras, then trim extreme depth outliers
    z2 = (X @ R.T + t.ravel())[:, 2]
    front = (X[:, 2] > 0) & (z2 > 0)
    X, p1k = X[front], p1[keep][front]
    if len(X) > 10:
        lo, hi = np.percentile(X[:, 2], [2, 98])
        ok = (X[:, 2] >= lo) & (X[:, 2] <= hi)
        X, p1k = X[ok], p1k[ok]

    colors = np.array([img1[int(round(y)), int(round(x))][::-1] for x, y in p1k], dtype=np.uint8)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

    # Median reprojection error in view 1 (pixels) as a quality check
    proj = (P1 @ np.hstack([X, np.ones((len(X), 1))]).T).T
    reproj = float(np.median(np.linalg.norm(proj[:, :2] / proj[:, 2:] - p1k, axis=1))) if len(X) else float("nan")
    return X, colors, angle, reproj


def save_plotly(points, colors, path, title):
    import plotly.graph_objects as go

    rgb = [f"rgb({r},{g},{b})" for r, g, b in colors]
    fig = go.Figure(data=[go.Scatter3d(
        # Flip y/z so the view matches image orientation (y down in images)
        x=points[:, 0], y=points[:, 2], z=-points[:, 1],
        mode="markers", marker=dict(size=3, color=rgb))])
    fig.update_layout(title=title, scene=dict(xaxis_title="X", yaxis_title="depth (Z)",
                                              zaxis_title="Y", aspectmode="data"),
                      width=900, height=750)
    fig.write_html(path, include_plotlyjs="cdn")


def show_open3d(points, colors):
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed; skipping Open3D viewer (pip install open3d)")
        return
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.visualization.draw_geometries([pcd])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args):
    cv2.setRNGSeed(args.seed)          # makes RANSAC results reproducible
    names, images = load_images(args.images, args.max_side)
    print(f"Loaded {len(images)} views: {', '.join(names)}")

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = ResNetFeatures().eval().to(device)

    keypoints, orb_des = detect_orb(images, args.n_features)
    cnn_des = [cnn_descriptors(model, img, kp, device, args.cnn_upscale)
               for img, kp in zip(images, keypoints)]
    print(f"ORB keypoints per view: {[len(k) for k in keypoints]}")
    print(f"CNN descriptor dim: {cnn_des[0].shape[1]} (ResNet18 layer1 + layer2 + layer3)")

    rows, cnn_matches = [], {}
    print(f"\n{'pair':>8} | {'ORB matches':>11} {'inliers':>8} {'ratio':>6} | "
          f"{'CNN matches':>11} {'inliers':>8} {'ratio':>6}")
    for a in range(len(images) - 1):
        b = a + 1
        stats = {"view_a": names[a], "view_b": names[b]}
        for kind, matches in (("orb", match_orb(orb_des[a], orb_des[b])),
                              ("cnn", match_cnn(cnn_des[a], cnn_des[b], keypoints[b],
                                                args.cnn_ratio, args.cnn_radius))):
            inl = ransac_inliers(keypoints[a], keypoints[b], matches, args.ransac_thresh)
            stats[f"{kind}_matches"] = len(matches)
            stats[f"{kind}_inliers"] = int(inl.sum())
            stats[f"{kind}_inlier_ratio"] = round(float(inl.mean()) if len(inl) else 0.0, 3)
            if kind == "cnn":
                cnn_matches[(a, b)] = [m for m, k in zip(matches, inl) if k]
        rows.append(stats)
        print(f"{a + 1:>3}-{b + 1:<4} | {stats['orb_matches']:>11} {stats['orb_inliers']:>8} "
              f"{stats['orb_inlier_ratio']:>6.2f} | {stats['cnn_matches']:>11} "
              f"{stats['cnn_inliers']:>8} {stats['cnn_inlier_ratio']:>6.2f}")

    tot = {k: sum(r[k] for r in rows) for k in ("orb_matches", "orb_inliers", "cnn_matches", "cnn_inliers")}
    print(f"{'total':>8} | {tot['orb_matches']:>11} {tot['orb_inliers']:>8} "
          f"{tot['orb_inliers'] / max(tot['orb_matches'], 1):>6.2f} | {tot['cnn_matches']:>11} "
          f"{tot['cnn_inliers']:>8} {tot['cnn_inliers'] / max(tot['cnn_matches'], 1):>6.2f}")

    os.makedirs(args.out, exist_ok=True)
    table = os.path.join(args.out, "matching_comparison.csv")
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-pair comparison to {table}")

    if args.pair:
        a, b = args.pair[0] - 1, args.pair[1] - 1
        if b != a + 1 or not 0 <= a < len(images) - 1:
            raise ValueError("--pair must be two consecutive view numbers, e.g. --pair 3 4")
    else:
        a, b = max(cnn_matches, key=lambda k: len(cnn_matches[k]))
    inliers = cnn_matches[(a, b)]
    vis = cv2.drawMatches(images[a], keypoints[a], images[b], keypoints[b],
                          [cv2.DMatch(i, j, 0) for i, j in inliers], None,
                          matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    vis_path = os.path.join(args.out, f"cnn_matches_{a + 1}_{b + 1}.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"Saved CNN inlier matches for views {a + 1}-{b + 1} to {vis_path}")
    if len(inliers) < 8:
        print("Not enough CNN inliers to reconstruct this pair.")
        return
    points, colors, angle, reproj = triangulate_pair(images[a], keypoints[a], keypoints[b],
                                                     inliers, args.focal_scale)
    print(f"Reconstructed views {a + 1}-{b + 1}: {len(points)} 3D points from {len(inliers)} "
          f"CNN inlier matches (estimated rotation {angle:.1f} deg, "
          f"median reprojection error {reproj:.2f} px)")

    html = os.path.join(args.out, "point_cloud.html")
    save_plotly(points, colors, html, f"Sparse 3D points from views {a + 1}-{b + 1} "
                                      f"(CNN-descriptor matches, up to scale)")
    print(f"Saved interactive point cloud to {html}")
    if args.open3d:
        show_open3d(points, colors)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--images", default=os.path.join(here, "bottle_images.zip"),
                   help="zip file or folder of views (default: bottle_images.zip)")
    p.add_argument("--out", default=os.path.join(here, "outputs"))
    p.add_argument("--max-side", type=int, default=640, help="resize longest side to this")
    p.add_argument("--n-features", type=int, default=3000, help="ORB keypoints per view")
    p.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC threshold (px)")
    p.add_argument("--cnn-upscale", type=float, default=2.0,
                   help="upscale factor for the CNN input (finer feature maps)")
    p.add_argument("--cnn-ratio", type=float, default=0.9, help="CNN ratio-test threshold")
    p.add_argument("--cnn-radius", type=float, default=16.0,
                   help="min distance (px) of the second-best candidate in the ratio test")
    p.add_argument("--focal-scale", type=float, default=1.2,
                   help="assumed focal length as a multiple of image width")
    p.add_argument("--pair", type=int, nargs=2, metavar=("A", "B"),
                   help="consecutive views to reconstruct (1-based); default: most CNN inliers")
    p.add_argument("--open3d", action="store_true", help="also open the cloud in Open3D")
    p.add_argument("--seed", type=int, default=0, help="RANSAC random seed")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
