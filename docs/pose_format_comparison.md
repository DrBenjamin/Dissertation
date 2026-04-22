# Comparison: YOLO Pose Format vs. Google MediaPipe Pose Detection

This document provides a comprehensive comparison between the YOLO pose annotation format used in the project's image dataset (`data/posture/labels`) and the Google MediaPipe ML Kit Pose Detection used in the analysis script (`code/human_posture_analysis.py`).

## Table of Contents

1. [Overview](#overview)
2. [YOLO Pose Format](#yolo-pose-format)
3. [MediaPipe Pose Detection](#mediapipe-pose-detection)
4. [Key Differences](#key-differences)
5. [Keypoint Mapping](#keypoint-mapping)
6. [Usage in Project](#usage-in-project)
7. [References](#references)

---

## Overview

The dissertation project utilizes two different pose estimation systems:

- **YOLO Pose Format**: Used for annotating the training dataset (`data/posture/`) with 4 keypoints per person
- **MediaPipe Pose Detection**: Used for runtime pose analysis in `human_posture_analysis.py` with 33 landmarks

These systems serve complementary purposes in the project workflow:
- YOLO annotations provide training data for custom pose models
- MediaPipe provides real-time pose detection and analysis capabilities

---

## YOLO Pose Format

### Overview

The YOLO pose format is used in the posture dataset annotations located in `data/posture/labels/`. This format extends YOLO's object detection capabilities with keypoint annotations for pose estimation.

**Documentation**: [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose/)

### Dataset Configuration

From `data/posture/data.yaml`:

```yaml
path: ../data/posture
train: images/train
val: images/val
test: images/val

# Keypoints
kpt_shape: [4, 3]  # 4 keypoints, 3 dimensions (x, y, visibility)
flip_idx: [0, 1, 2, 3]

# Classes
names:
    0: person
```

### Dataset Statistics

- **Training samples**: 250 annotated images
- **Validation samples**: 250 annotated images
- **Keypoints per person**: 4
- **Data source**: [Kaggle Posture Keypoints Detection Dataset](https://www.kaggle.com/datasets/melsmm/posture-keypoints-detection)

### Label Format Structure

Each label file contains one line per detected person:

```
<class_id> <bbox_cx> <bbox_cy> <bbox_w> <bbox_h> <kp1_x> <kp1_y> <kp1_v> <kp2_x> <kp2_y> <kp2_v> <kp3_x> <kp3_y> <kp3_v> <kp4_x> <kp4_y> <kp4_v>
```

**Components**:
- `class_id`: Object class (0 = person)
- `bbox_cx, bbox_cy`: Bounding box center coordinates (normalized 0-1)
- `bbox_w, bbox_h`: Bounding box width and height (normalized 0-1)
- `kpN_x, kpN_y`: Keypoint N coordinates (normalized 0-1)
- `kpN_v`: Keypoint N visibility flag
  - `0`: Not labeled
  - `1`: Labeled but not visible (occluded)
  - `2`: Labeled and visible

### Example Label

From `data/posture/labels/train/92.txt`:

```
0 0.541831 0.567294 0.099790 0.740937 0.491935 0.196825 2 0.591726 0.434095 2 0.537210 0.718270 2 0.500677 0.937762 2
```

**Parsed**:
- Class: 0 (person)
- Bounding box center: (0.542, 0.567)
- Bounding box size: (0.100, 0.741)
- Keypoint 1: (0.492, 0.197) - visible
- Keypoint 2: (0.592, 0.434) - visible
- Keypoint 3: (0.537, 0.718) - visible
- Keypoint 4: (0.501, 0.938) - visible

### Keypoint Definitions (4 Total)

Based on typical posture datasets and the vertical distribution of coordinates in the examples:

1. **Keypoint 1** (Head/Top): Head or nose position
2. **Keypoint 2** (Shoulder): Shoulder region (likely left or center)
3. **Keypoint 3** (Hip): Hip region (likely left or center)
4. **Keypoint 4** (Ankle/Foot): Lower extremity (ankle or foot)

These 4 points form a minimal skeleton for side-view posture analysis, enabling measurement of:
- Overall body alignment
- Spinal curvature
- Head-shoulder-hip-ankle alignment
- Forward head posture

---

## MediaPipe Pose Detection

### Overview

MediaPipe Pose provides a comprehensive 33-landmark full-body pose estimation solution, used in `code/human_posture_analysis.py` for real-time posture analysis.

**Documentation**: [Google MediaPipe Pose Detection](https://developers.google.com/ml-kit/vision/pose-detection)

### Landmark Set (33 Total)

MediaPipe Pose detects 33 landmarks covering the entire body:

| Index | Landmark Name | Index | Landmark Name |
|-------|--------------|-------|--------------|
| 0 | NOSE | 17 | LEFT_PINKY |
| 1 | LEFT_EYE_INNER | 18 | RIGHT_PINKY |
| 2 | LEFT_EYE | 19 | LEFT_INDEX |
| 3 | LEFT_EYE_OUTER | 20 | RIGHT_INDEX |
| 4 | RIGHT_EYE_INNER | 21 | LEFT_THUMB |
| 5 | RIGHT_EYE | 22 | RIGHT_THUMB |
| 6 | RIGHT_EYE_OUTER | 23 | LEFT_HIP |
| 7 | LEFT_EAR | 24 | RIGHT_HIP |
| 8 | RIGHT_EAR | 25 | LEFT_KNEE |
| 9 | MOUTH_LEFT | 26 | RIGHT_KNEE |
| 10 | MOUTH_RIGHT | 27 | LEFT_ANKLE |
| 11 | LEFT_SHOULDER | 28 | RIGHT_ANKLE |
| 12 | RIGHT_SHOULDER | 29 | LEFT_HEEL |
| 13 | LEFT_ELBOW | 30 | RIGHT_HEEL |
| 14 | RIGHT_ELBOW | 31 | LEFT_FOOT_INDEX |
| 15 | LEFT_WRIST | 32 | RIGHT_FOOT_INDEX |
| 16 | RIGHT_WRIST | | |

### Data Format

Each MediaPipe landmark includes:

- **x, y**: Normalized image coordinates (0-1)
- **z**: Depth coordinate (relative to hips, in same scale as x)
- **visibility**: Confidence score (0-1) that landmark is visible in image
- **presence**: Confidence score (0-1) that landmark is present in image

MediaPipe also provides world coordinates for 3D pose:
- **x, y, z**: Real-world 3D coordinates in meters (relative to hips as origin)

### Landmarks Used in `human_posture_analysis.py`

The script uses a subset of 4 landmarks for posture analysis:

```python
# Extracted landmarks (line 84-92)
l_shldr_x = landmarks.landmark[lm_pose.LEFT_SHOULDER].x    # Index 11
l_shldr_y = landmarks.landmark[lm_pose.LEFT_SHOULDER].y
r_shldr_x = landmarks.landmark[lm_pose.RIGHT_SHOULDER].x   # Index 12
r_shldr_y = landmarks.landmark[lm_pose.RIGHT_SHOULDER].y
l_ear_x = landmarks.landmark[lm_pose.LEFT_EAR].x           # Index 7
l_ear_y = landmarks.landmark[lm_pose.LEFT_EAR].y
l_hip_x = landmarks.landmark[lm_pose.LEFT_HIP].x           # Index 23
l_hip_y = landmarks.landmark[lm_pose.LEFT_HIP].y
```

### Posture Metrics Calculated

1. **Shoulder Alignment** (line 95):
   ```python
   offset = find_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
   ```
   - Threshold: < 100 pixels = aligned

2. **Neck Inclination** (line 100):
   ```python
   neck_inclination = find_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
   ```
   - Good posture threshold: < 40 degrees

3. **Torso Inclination** (line 101):
   ```python
   torso_inclination = find_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
   ```
   - Good posture threshold: < 10 degrees

### Posture Classification

```python
# Good posture criteria (line 114)
if neck_inclination < 40 and torso_inclination < 10:
    # Display green annotations - "Good Posture"
else:
    # Display red annotations - "Poor Posture"
```

---

## Key Differences

### 1. Number of Keypoints

| Aspect | YOLO Pose | MediaPipe Pose |
|--------|-----------|----------------|
| **Total landmarks** | 4 keypoints | 33 landmarks |
| **Coverage** | Minimal skeleton (head, shoulder, hip, foot) | Full body (face, torso, arms, legs) |
| **Detail level** | Basic posture alignment | Comprehensive pose tracking |

### 2. Data Format

#### YOLO Pose Format
```
Format: class cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
- Coordinates: Normalized (0-1) relative to image dimensions
- Visibility: Discrete flag (0, 1, 2)
- Includes bounding box for detection
- Single line per person in text file
```

#### MediaPipe Format
```
Format: x, y, z, visibility, presence per landmark
- Image coordinates: Normalized (0-1)
- World coordinates: 3D position in meters (relative to hips)
- Visibility: Continuous score (0-1)
- Presence: Continuous score (0-1)
- Depth information (z coordinate)
- Returned as structured object, not text file
```

### 3. Coordinate Systems

| System | YOLO Pose | MediaPipe Pose |
|--------|-----------|----------------|
| **Image space** | 2D normalized (x, y) | 2D normalized (x, y) |
| **3D support** | No | Yes (world coordinates) |
| **Depth** | No | Yes (z coordinate) |
| **Origin** | Top-left corner | Top-left corner (image) / Hips (world) |
| **Scale** | 0-1 (normalized) | 0-1 (image), meters (world) |

### 4. Visibility Encoding

| Aspect | YOLO Pose | MediaPipe Pose |
|--------|-----------|----------------|
| **Type** | Discrete flag | Continuous scores |
| **Values** | 0 (unlabeled)<br>1 (occluded)<br>2 (visible) | visibility: 0.0-1.0<br>presence: 0.0-1.0 |
| **Semantics** | Binary visible/not visible | Probabilistic confidence |

### 5. Purpose and Use Case

#### YOLO Pose (Dataset Annotations)
- **Primary use**: Training custom pose models
- **Optimization**: Fast object detection + minimal keypoints
- **Trade-off**: Speed and efficiency over detail
- **Application**: Real-time detection on resource-constrained devices
- **Annotation workflow**: Manual labeling or pre-trained model predictions

#### MediaPipe Pose (Runtime Detection)
- **Primary use**: Real-time pose analysis and tracking
- **Optimization**: Comprehensive pose representation
- **Trade-off**: Rich information over minimal data
- **Application**: Detailed pose analysis, fitness tracking, AR effects
- **Detection**: On-device ML inference

### 6. Integration in Project

| Component | YOLO Pose | MediaPipe Pose |
|-----------|-----------|----------------|
| **Location** | `data/posture/labels/` | `code/human_posture_analysis.py` |
| **Purpose** | Training data annotations | Runtime pose detection |
| **Model** | YOLO11n (Ultralytics) | MediaPipe Pose Landmarker |
| **Training** | Transfer learning with custom dataset | Pre-trained model (no retraining) |
| **Notebook** | `code/posture-keypoints.ipynb` | `code/human_posture_analysis.ipynb` |

---

## Keypoint Mapping

While the two systems have different numbers of keypoints, we can establish a conceptual mapping based on the anatomical regions they represent:

### YOLO → MediaPipe Correspondence

| YOLO Index | YOLO Keypoint | MediaPipe Index | MediaPipe Landmark | Notes |
|------------|---------------|-----------------|-------------------|-------|
| 1 | Head/Top | 0 or 7 | NOSE or LEFT_EAR | Head position reference |
| 2 | Shoulder | 11 | LEFT_SHOULDER | Shoulder alignment point |
| 3 | Hip | 23 | LEFT_HIP | Torso-leg junction |
| 4 | Ankle/Foot | 27 or 31 | LEFT_ANKLE or LEFT_FOOT_INDEX | Lower extremity |

### MediaPipe Subset for YOLO-Like Analysis

The `human_posture_analysis.py` script effectively replicates YOLO's minimal keypoint approach by using only 4 MediaPipe landmarks:

```python
Landmarks used:
- LEFT_SHOULDER (11)   → Similar to YOLO keypoint 2
- RIGHT_SHOULDER (12)  → For bilateral alignment (not in YOLO)
- LEFT_EAR (7)         → Similar to YOLO keypoint 1
- LEFT_HIP (23)        → Similar to YOLO keypoint 3
```

This demonstrates that the same posture metrics can be calculated with either:
- 4 carefully chosen keypoints (YOLO approach)
- 4 landmarks selected from a comprehensive 33-point set (MediaPipe approach)

---

## Usage in Project

### Training Workflow (YOLO)

1. **Dataset**: `data/posture/` with YOLO format annotations
   - 250 training images
   - 250 validation images
   - 4 keypoints per person
   - Manual annotations from Kaggle dataset

2. **Model**: YOLO11n (nano variant for efficiency)
   - Pretrained weights: `data/models/yolo11n.pt`
   - Transfer learning on posture dataset

3. **Training**: `code/posture-keypoints.ipynb`
   ```python
   from ultralytics import YOLO
   model = YOLO("../data/models/yolo11n.pt")
   results = model.train(
       data="../data/posture/data.yaml",
       epochs=200,
       imgsz=640,
       batch=32
   )
   ```

4. **Output**: Custom-trained YOLO model for posture keypoint detection

### Analysis Workflow (MediaPipe)

1. **Input**: Image or video file
   ```bash
   python code/human_posture_analysis.py \
       --mode image \
       --input-video ./data/images/input.png \
       --output-video ./data/images/output.png
   ```

2. **Detection**: MediaPipe Pose Landmarker
   ```python
   mp_pose = mp.solutions.pose
   pose = mp_pose.Pose(static_image_mode=True)
   keypoints = pose.process(rgb_image)
   ```

3. **Analysis**: Calculate posture metrics
   - Shoulder alignment (Euclidean distance)
   - Neck inclination (angle from vertical)
   - Torso inclination (angle from vertical)

4. **Output**: Annotated image/video with:
   - Posture quality label (Good/Poor)
   - Angle measurements
   - Alignment metrics
   - Visual overlays (circles, lines, text)

### Complementary Roles

The two systems work together in the project:

```
YOLO Training Pipeline:
Dataset (YOLO labels) → YOLO11 Training → Custom Pose Model → Fast Detection

MediaPipe Analysis Pipeline:
Input → MediaPipe Detection → Posture Analysis → Annotated Output
```

**YOLO advantages**:
- Lightweight model suitable for edge devices
- Fast inference for real-time applications
- Custom training on specific posture types
- Efficient 4-keypoint representation

**MediaPipe advantages**:
- Rich 33-landmark pose information
- Pre-trained, ready-to-use model
- 3D world coordinates for spatial analysis
- No training data required
- Comprehensive body tracking

---

## References

### YOLO Pose

- [Ultralytics YOLO11 Documentation](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics Pose Estimation](https://docs.ultralytics.com/tasks/pose/)
- [YOLO Pose Format Specification](https://docs.ultralytics.com/datasets/pose/)
- [Kaggle Posture Dataset](https://www.kaggle.com/datasets/melsmm/posture-keypoints-detection)

### MediaPipe Pose

- [MediaPipe Solutions - Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe Pose Detection - ML Kit](https://developers.google.com/ml-kit/vision/pose-detection)
- [MediaPipe Studio - Pose Landmarker Demo](https://mediapipe-studio.webapps.google.com/studio/demo/pose_landmarker)
- [MediaPipe GitHub Repository](https://github.com/google/mediapipe)

### Implementation References

- Project repository: `code/human_posture_analysis.py`
- Training notebook: `code/posture-keypoints.ipynb`
- Dataset configuration: `data/posture/data.yaml`
- OpenCV posture analysis tutorial: [LearnOpenCV - Body Posture Analysis](https://learnopencv.com/building-a-body-posture-analysis-system-using-mediapipe/)

---

## Conclusion

This comparison highlights the complementary nature of YOLO pose annotations and MediaPipe pose detection in the dissertation project:

- **YOLO** provides an efficient, trainable system optimized for specific posture detection tasks with minimal keypoints
- **MediaPipe** offers comprehensive, ready-to-use pose tracking with rich anatomical detail

The choice between systems depends on the specific use case:
- Use **YOLO** when: training custom models, optimizing for speed, deploying to edge devices
- Use **MediaPipe** when: requiring detailed pose analysis, no training data available, need 3D tracking

Both systems represent normalized coordinates (0-1) and detect body keypoints, but differ significantly in scope, detail, and intended application. The project effectively leverages both approaches to combine the benefits of custom training (YOLO) with robust pre-trained detection (MediaPipe).
