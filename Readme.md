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
- SIFT detects and computes descriptors across all 14 images
- Average **1199 keypoints per image**

![SIFT Keypoints](sift_keypoints.png)

---

### 2. Feature Matching — Lowe's Ratio Test
- BFMatcher with L2 norm used for nearest-neighbour matching
- Lowe's ratio threshold `t = 0.75` filters low-confidence matches
- Only matches where `d(m) < 0.75 × d(n)` are retained

---

### 3. Outlier Rejection — RANSAC
- Fundamental matrix `F` estimated via RANSAC
  - Reprojection threshold: `3.0 px`
  - Confidence: `99.9%`
- Inlier matches retained where reprojection error `< 3px`
- Target: **≥ 70% inlier ratio** per image pair

![Inlier Matches](inlier_matches.png)

---

### 4. Epipolar Geometry Validation
- Epilines computed via `cv2.computeCorrespondEpilines`
- 12 sampled point-line pairs visualized per image pair
- Epilines from left image project onto corresponding points in right image, confirming geometric consistency of estimated `F`

![Epipolar Geometry](epipolar_geometry.png)
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

## Repository Structure

```
├── main.ipynb          # Full Stage 1 pipeline notebook
├── project2/           # Input image dataset (Google Drive)
└── README.md
```

---

## Planned: Stage 2
- Camera pose estimation from Essential Matrix
- Triangulation of 3D point cloud
- Bundle adjustment for global refinement
- Dense reconstruction

---

## Authors
Priyanka Lakariya — Northeastern University MS Robotics
```
