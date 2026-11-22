# CNN-Driven Multi-View Matching and Sparse 3D Reconstruction

Can features from an ImageNet-pretrained CNN find correspondences between views of an object
better than a hand-crafted descriptor? This project compares the two on 12 photos of a water
bottle taken from different angles, then turns the CNN matches into a sparse 3D point cloud.

## Method

1. **Load views:** `bottle_images.zip`, naturally sorted (`bottle_1 … bottle_12`), longest side
   resized to 640 px.
2. **Keypoints:** ORB, up to 3000 per view.
3. **Two descriptors for the same keypoints:**
   - **ORB** 256-bit binary descriptors (baseline)
   - **CNN:** ResNet18 (ImageNet weights) `layer1`, `layer2` and `layer3` feature maps, computed
     on a 2× upscaled image and bilinearly sampled at each keypoint. Each layer is
     L2-normalised and concatenated (448-D).
4. **Matching** on every consecutive view pair:
   - ORB: Hamming kNN, Lowe ratio test (0.8) and a cross-check
   - CNN: cosine-similarity mutual nearest neighbours with a *spatially aware* ratio test. The
     second-best candidate must be more than 16 px from the best match, because neighbouring
     keypoints sample almost the same coarse feature-map cell. With a plain ratio test
     (`--cnn-radius 0`) the CNN gets 351 matches / 270 inliers in total. The spatially aware
     test gets 1494 / 1075.
5. **Geometric verification:** RANSAC on the fundamental matrix (3 px threshold, fixed seed).
6. **Reconstruction:** for the pair with the most CNN inliers, estimate the essential matrix
   with approximate intrinsics (focal length ≈ 1.2 × width), recover the relative pose, and
   triangulate the inliers. Points must lie in front of both cameras and depth outliers are
   trimmed.
7. **Output:** a per-pair comparison CSV, an image of the inlier matches, and an interactive
   Plotly point cloud (optionally Open3D).

## Results

These come from running `python cnn_3d_awareness.py`. RANSAC inliers per consecutive pair:

| Pair | ORB matches | ORB inliers | CNN matches | CNN inliers |
|---|---|---|---|---|
| 1-2 | 82 | 58 | 265 | 183 |
| 2-3 | 56 | 39 | 142 | 99 |
| 3-4 | 73 | 52 | 199 | 133 |
| 4-5 | 152 | 121 | 369 | 273 |
| 5-6 | 29 | 14 | 73 | 49 |
| 6-7 | 21 | 13 | 70 | 54 |
| 7-8 | 48 | 36 | 120 | 90 |
| 8-9 | 17 | 10 | 38 | 16 |
| 9-10 | 50 | 32 | 72 | 60 |
| 10-11 | 33 | 25 | 55 | 43 |
| 11-12 | 55 | 42 | 91 | 75 |
| **Total** | **616** | **442** | **1494** | **1075** |

- The ResNet18 descriptors give **~2.4× more geometrically consistent matches** than ORB
  descriptors on the same keypoints (1075 vs 442), at the same inlier ratio (0.72 vs 0.72).
- The CNN finds more inliers on every pair, including the hardest one (8 → 9, where the bottle
  goes from standing to lying down): 16 vs 10.
- The best pair (views 4-5, 273 CNN inliers) triangulates to 220 sparse 3D points.

### Limitations

- "Inliers" means consistent with a fundamental matrix under RANSAC. There is no ground-truth
  correspondence, so a few wrong matches lying near epipolar lines can still count.
- The camera is uncalibrated and the bottle is transparent and was moved by hand between shots
  (the lid also opens and closes). The reconstruction is therefore a **rough sparse
  approximation up to an unknown scale**, not a metric 3D model. The relative pose from two
  views like these is poorly constrained.
- Pretrained features are used as-is (no fine-tuning or learned matcher such as SuperPoint or
  LoFTR).

## How to run

```bash
pip install -r requirements.txt
python cnn_3d_awareness.py                 # writes results to outputs/
python cnn_3d_awareness.py --pair 1 2      # reconstruct a specific consecutive pair
python cnn_3d_awareness.py --open3d        # also open the cloud in Open3D (pip install open3d)
python cnn_3d_awareness.py --help
```

Outputs (git-ignored):

- `outputs/matching_comparison.csv`: per-pair ORB vs CNN match and inlier counts
- `outputs/cnn_matches_A_B.jpg`: inlier matches for the reconstructed pair
- `outputs/point_cloud.html`: interactive 3D point cloud

The notebook `cnn_3d_awareness.ipynb` walks through the same steps with plots. It runs locally
or in Google Colab (upload `cnn_3d_awareness.py` and `bottle_images.zip` when asked).

## Files

```
cnn_3d_awareness.py      # pipeline + CLI
cnn_3d_awareness.ipynb   # step-by-step walkthrough with outputs
bottle_images.zip        # 12 input views
requirements.txt
```
