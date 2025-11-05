# CNN-Driven 3D Awareness Demo

This project demonstrates how **2D convolutional neural network features** can be used to approximate **3D object structure**, 
using multiple images of a water bottle taken from different angles.

### 🔥 Key Idea
Even though CNNs operate in **2D**, the spatial patterns they learn contain enough structure to infer **depth and geometry** 
when comparing views of the same object.

### 🧠 Techniques Used
- Pretrained **ResNet18** feature extraction
- ORB-based multi-view keypoint matching (OpenCV)
- Depth approximation via keypoint displacement
- 3D point cloud rendering (Open3D WebRTC Viewer)

### 🚀 Outcome
We obtain a **3D point cloud** that represents the rough geometry of the object.

### 📂 Directory Structure
