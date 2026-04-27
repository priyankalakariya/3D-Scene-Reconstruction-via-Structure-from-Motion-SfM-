# 3D Scene Reconstruction via Structure-from-Motion (SfM)

A computer vision pipeline implementing **Stage 1 of Structure-from-Motion** — covering feature detection, descriptor matching, outlier rejection via RANSAC, and epipolar geometry validation — applied to a multi-view image dataset of a bronze dog statue captured at Northeastern University.

---

## Pipeline Overview

```
Multi-View Images → SIFT Detection → Lowe's Ratio Filtering → RANSAC (Fundamental Matrix) → Epipolar Geometry Validation
```

---

## Stage 1: Feature Extraction & Geometric Verification
 
### 1. SIFT Keypoint Detection
- Images downscaled by `0.5×` to reduce compute while preserving descriptor quality
- SIFT detects and computes 128-dimensional descriptors across all 58 images
- Average 1199 keypoints per image
- **GrabCut masking** applied to isolate the statue from the background before detection
  - GrabCut iteratively segments foreground from background using a bounding rectangle initialized at 5% image margin
  - Morphological closing applied post-segmentation to clean mask boundaries
  - Keypoint detection restricted to the masked statue region, suppressing background wall features

### 2. Feature Matching — Lowe's Ratio Test
- BFMatcher with L2 norm used for nearest-neighbour matching
- Lowe's ratio threshold `t = 0.75` filters low-confidence matches
- Only matches where `d(m) < 0.75 × d(n)` are retained

### 3. Outlier Rejection — RANSAC
- Fundamental matrix `F` estimated via RANSAC per consecutive image pair
  - Reprojection threshold: `3.0 px`
  - Confidence: `99.9%`
- Inlier matches retained where reprojection error `< 3 px`
- Target: ≥ 70% inlier ratio per image pair
- **Result: 51 / 58 pairs passed** — 7 failures attributed to abrupt viewpoint jumps during capture
---
 
![Inlier Matches](inlier_matches.png)

### 4. Epipolar Geometry Validation
- Epilines computed via `cv2.computeCorrespondEpilines`
- 12 sampled point-line pairs visualized per passing image pair
- Epilines from left image project onto corresponding points in right image
- Tight point-to-epiline alignment confirms correct Fundamental Matrix estimation across all passing pairs

![Epipolar Geometry](epipolar_geometry.png)

### Stage 1 Dataset
 
| Property | Value |
|---|---|
| Subject | Bronze Husky statue, Northeastern University |
| Images | 58 multi-view photographs |
| Format | JPEG (smartphone capture) |
| Capture pattern | Consecutive views orbiting the subject |
| Passing pairs | 51 / 58 (87.9%) |
| Failing pairs | 7 — large viewpoint jumps between consecutive captures |
 

---

## Dataset
- **Subject**: Bronze Husky statue, Northeastern University
- **Images**: 60 multi-view photographs captured by handheld smartphone
- **Capture pattern**: Consecutive views with ~30° rotation increments around the subject
- **Failing pairs** (07↔08, 08↔09, 11↔12) attributed to large viewpoint jumps between captures

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Feature Detection | SIFT (`cv2.SIFT_create`) |
| Matching | BFMatcher + Lowe's Ratio Test |
| Outlier Rejection | RANSAC (`cv2.findFundamentalMat`) |
| Epipolar Validation | `cv2.computeCorrespondEpilines` |
| Environment | Python, OpenCV, NumPy, Matplotlib, Google Colab |

---

## Stage 2: Camera Pose Recovery & 3D Reconstruction
 
### 1. Camera Intrinsics Estimation
- HEIC images carried EXIF metadata but lacked `FocalPlaneXResolution`
- Three estimation methods attempted:
  - Direct EXIF conversion — failed, tag absent
  - 35mm equivalent: `f = (f_35 / 36) × width` — gave `fx = 240 px`, pre-BA error 2.03 px
  - OpenCV SfM approximation: `f = 0.9 × max(width, height)` at SIFT working resolution — gave `fx = 432 px`, pre-BA error 1.15 px
- Final method adopted: `f = 0.9 × max(width, height)` — 43% lower pre-BA error than 35mm equivalent
- Final camera matrix at working resolution:
      K = [[432,   0, 180],
           [  0, 432, 240],
           [  0,   0,   1]]
 
### 2. Windowed Pair Matching
- Each image paired with its next 4 neighbors (`window = 5`)
- Generated 238 candidate pairs, of which 206 passed RANSAC at ≥ 70% inlier ratio
- Windowed matching provides redundant overlap paths so a single failing pair does not break the reconstruction chain

### 3. Essential Matrix & Baseline Pose Recovery
- Essential Matrix computed from Fundamental Matrix: `E = K.T @ F @ K`
- Baseline pair selected as mid-sequence pair with highest inlier count
  - Pair `15 ↔ 16`: 805 inliers, 98.3% inlier ratio
- `cv2.recoverPose` decomposes E into R and t using cheirality to disambiguate the four candidate solutions
- Camera 15 set as world origin `[I | 0]`, Camera 16 initialized with recovered `[R | t]`

### 4. Initial Triangulation
- Baseline pair triangulated using `cv2.triangulatePoints` with `P = K @ [R | t]`
- Three sequential filters applied:
  - **Cheirality**: removes points with negative depth in either camera (805 → 743)
  - **Depth**: rejects points beyond 200 scene units (743 → 706)
  - **Reprojection**: rejects points with error > 5 px in either view (no additional cuts)
- RGB color sampled from reference image at each keypoint location

### 5. Incremental PnP Registration
- For each unregistered camera, 2D–3D correspondences built by cross-referencing RANSAC inlier matches against a keypoint-to-3D lookup table
- Cameras with fewer than 6 correspondences deferred to later passes (up to 5 passes)
- Pose estimation via `cv2.solvePnPRansac` with EPnP algorithm
  - EPnP requires no initial pose estimate and handles sparse correspondence noise robustly
  - Reprojection threshold: `8 px`, 1000 iterations, 99.9% confidence
- After each registration, new 3D points triangulated with all registered neighbors sharing passing RANSAC pairs
- **Triangulation angle filter** (`> 1°`): rejects points from near-parallel viewing rays which have poor depth resolution despite low reprojection error
- **Result: all 62 cameras registered in a single pass with 0 failures**

### 6. Bundle Adjustment
- Implemented via `scipy.optimize.least_squares` with Trust Region Reflective (TRF) method
- Each camera: 6 parameters (Rodrigues rotation vector + translation vector)
- Each 3D point: 3 parameters
  - Total parameter vector: `6 × 62 + 3 × 12,968 = 39,276`
- Jacobian sparsity structure built in COO sparse format — dense Jacobian would be computationally intractable at this scale
- Residual function computes per-observation reprojection error across all camera-point pairs
- **Result: 1.15 px → 0.83 px over 9 iterations**

### 7. Statistical Outlier Filtering
- Mean distance to 30 nearest neighbors computed per point
- Points exceeding `μ + 0.5σ` threshold rejected
- Tighter than conventional `2σ` to remove background points at large distances from the statue cluster
- **Result: 12,968 → 8,672 points retained**

### Stage 2 Dataset
 
| Property | Value |
|---|---|
| Subject | Bronze Husky statue, Northeastern University |
| Images | 62 multi-view photographs |
| Format | HEIC (iPhone, original resolution) |
| Full resolution | 6048 × 8064 px |
| Loaded resolution | Max 1920 px width |
| Working resolution | ~360 × 480 px (0.25× of loaded size) |
| Capture pattern | Semi-circular arc with ~15° increments |
 
---
 
## Stage 2 Results
 
| Metric | Value |
|---|---|
| Cameras registered | 62 / 62 (100%) |
| Passing RANSAC pairs | 206 / 238 (86.6%) |
| Baseline pair | 15 ↔ 16, 805 inliers, 98.3% |
| Initial triangulated points | 706 |
| Points after PnP triangulation | 12,968 |
| Reprojection error before BA | 1.15 px |
| Reprojection error after BA | 0.83 px |
| Points after outlier filtering | 8,672 |
| Output format | PLY with XYZ + RGB |
 
---
 
## Tech Stack
 
| Component | Tool |
|---|---|
| Feature Detection | SIFT (`cv2.SIFT_create`) |
| Background Masking | GrabCut (`cv2.grabCut`) |
| Feature Matching | BFMatcher + Lowe's Ratio Test |
| Outlier Rejection | RANSAC (`cv2.findFundamentalMat`) |
| Epipolar Validation | `cv2.computeCorrespondEpilines` |
| Intrinsics Estimation | Pillow EXIF + OpenCV SfM approximation |
| HEIC Conversion | `pillow-heif` |
| Pose Recovery | `cv2.recoverPose`, `cv2.solvePnPRansac` (EPnP) |
| Triangulation | `cv2.triangulatePoints` |
| Bundle Adjustment | `scipy.optimize.least_squares` (TRF, COO sparse) |
| Outlier Filtering | `sklearn.neighbors.NearestNeighbors` |
| Point Cloud Export | PLY (ASCII format with RGB) |
| Visualization | MeshLab, Matplotlib 3D |
| Environment | Python, OpenCV, NumPy, SciPy, Google Colab |
