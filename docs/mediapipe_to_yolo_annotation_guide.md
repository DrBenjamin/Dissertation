# MediaPipe to YOLO Annotation Guide

## Overview

This guide explains how to adapt the `code/human_posture_analysis.py` script to write YOLO-format pose annotations and compare them with the existing YOLO annotations in the `data/posture/labels/` directory.

**Purpose**: Enable validation and comparison between:
- MediaPipe pose detection landmarks (33 total)
- YOLO pose keypoint annotations (4 keypoints: cervical spine/head, thoracic spine/shoulder, lumbar spine/hip, sacral spine/ankle)

## Table of Contents

1. [Understanding the Formats](#understanding-the-formats)
2. [Landmark Mapping Strategy](#landmark-mapping-strategy)
3. [Adapting the Script](#adapting-the-script)
4. [Writing YOLO Format Annotations](#writing-yolo-format-annotations)
5. [Comparing Annotations](#comparing-annotations)
6. [Implementation Example](#implementation-example)
7. [Usage Instructions](#usage-instructions)

---

## Understanding the Formats

### YOLO Pose Format (Existing Dataset)

The YOLO annotations in `data/posture/labels/` follow this structure:

```
<class_id> <bbox_cx> <bbox_cy> <bbox_w> <bbox_h> <kp1_x> <kp1_y> <kp1_v> <kp2_x> <kp2_y> <kp2_v> <kp3_x> <kp3_y> <kp3_v> <kp4_x> <kp4_y> <kp4_v>
```

**Components**:
- `class_id`: Always 0 (person)
- `bbox_cx, bbox_cy`: Bounding box center (normalized 0-1)
- `bbox_w, bbox_h`: Bounding box dimensions (normalized 0-1)
- `kpN_x, kpN_y`: Keypoint N coordinates (normalized 0-1)
- `kpN_v`: Visibility flag (0=unlabeled, 1=occluded, 2=visible)

**4 Keypoints** (based on side-view spinal imaging):
1. **Keypoint 1**: Cervical spine region (head/nose)
2. **Keypoint 2**: Thoracic spine region (shoulder)
3. **Keypoint 3**: Lumbar spine region (hip)
4. **Keypoint 4**: Sacral spine region (ankle/foot)

**Example from dataset**:
```
0 0.541831 0.567294 0.099790 0.740937 0.491935 0.196825 2 0.591726 0.434095 2 0.537210 0.718270 2 0.500677 0.937762 2
```

### MediaPipe Pose Format (Script Output)

MediaPipe provides 33 landmarks with:
- `x, y`: Normalized coordinates (0-1)
- `z`: Depth coordinate
- `visibility`: Confidence score (0-1)

**Relevant landmarks for spinal alignment** (side view):

| MediaPipe Index | Landmark Name | YOLO Mapping |
|----------------|---------------|--------------|
| 0 | NOSE | Keypoint 1 (cervical) |
| 7 | LEFT_EAR | Alternative for Keypoint 1 |
| 8 | RIGHT_EAR | Alternative for Keypoint 1 |
| 11 | LEFT_SHOULDER | Keypoint 2 (thoracic) |
| 12 | RIGHT_SHOULDER | Alternative for Keypoint 2 |
| 23 | LEFT_HIP | Keypoint 3 (lumbar) |
| 24 | RIGHT_HIP | Alternative for Keypoint 3 |
| 27 | LEFT_ANKLE | Keypoint 4 (sacral) |
| 28 | RIGHT_ANKLE | Alternative for Keypoint 4 |

---

## Landmark Mapping Strategy

### Side View Mapping (Primary Use Case)

For side-view images showing cervical, thoracic, lumbar, and sacral spine:

```python
# YOLO Keypoint 1: Cervical spine (head/nose region)
mediapipe_kp1 = landmarks[0]  # NOSE
# Alternative: Use LEFT_EAR (7) or average of ears for side profiles

# YOLO Keypoint 2: Thoracic spine (shoulder region)
mediapipe_kp2 = landmarks[11]  # LEFT_SHOULDER
# Alternative: Use RIGHT_SHOULDER (12) or average both shoulders

# YOLO Keypoint 3: Lumbar spine (hip region)
mediapipe_kp3 = landmarks[23]  # LEFT_HIP
# Alternative: Use RIGHT_HIP (24) or average both hips

# YOLO Keypoint 4: Sacral spine (ankle/foot region)
mediapipe_kp4 = landmarks[27]  # LEFT_ANKLE
# Alternative: Use RIGHT_ANKLE (28) or LEFT_FOOT_INDEX (31)
```

### Handling Bilateral Landmarks (Left/Right Selection)

For side-view images, choose the visible side:

```python
def select_visible_side(landmarks, lm_pose):
    """Select the most visible side based on visibility scores."""
    
    left_shoulder_vis = landmarks.landmark[lm_pose.LEFT_SHOULDER].visibility
    right_shoulder_vis = landmarks.landmark[lm_pose.RIGHT_SHOULDER].visibility
    
    # Use the side with higher visibility
    use_left = left_shoulder_vis >= right_shoulder_vis
    
    if use_left:
        return {
            'shoulder': lm_pose.LEFT_SHOULDER,
            'hip': lm_pose.LEFT_HIP,
            'ankle': lm_pose.LEFT_ANKLE,
            'ear': lm_pose.LEFT_EAR
        }
    else:
        return {
            'shoulder': lm_pose.RIGHT_SHOULDER,
            'hip': lm_pose.RIGHT_HIP,
            'ankle': lm_pose.RIGHT_ANKLE,
            'ear': lm_pose.RIGHT_EAR
        }
```

### Visibility Conversion

Convert MediaPipe visibility to YOLO format:

```python
def mediapipe_to_yolo_visibility(mp_visibility, mp_presence=None):
    """Convert MediaPipe visibility score to YOLO visibility flag.
    
    Args:
        mp_visibility: MediaPipe visibility score (0.0-1.0)
        mp_presence: MediaPipe presence score (0.0-1.0), optional
        
    Returns:
        YOLO visibility flag: 0 (unlabeled), 1 (occluded), 2 (visible)
    """
    # Consider both visibility and presence if available
    confidence = mp_visibility
    if mp_presence is not None:
        confidence = min(mp_visibility, mp_presence)
    
    if confidence > 0.5:
        return 2  # Visible
    elif confidence > 0.0:
        return 1  # Occluded but present
    else:
        return 0  # Not labeled/detected
```

---

## Adapting the Script

### Required Modifications to `human_posture_analysis.py`

The script needs the following enhancements:

1. **Extract YOLO-compatible landmarks** instead of only posture metrics
2. **Calculate bounding box** from detected landmarks
3. **Write annotations** in YOLO format to text files
4. **Support batch processing** for comparing multiple images
5. **Implement comparison function** to measure annotation differences

### Key Functions to Add

```python
def extract_yolo_keypoints(landmarks, lm_pose, frame_width, frame_height):
    """Extract 4 keypoints in YOLO format from MediaPipe landmarks.
    
    Returns:
        dict: {
            'keypoints': [(x1, y1, v1), (x2, y2, v2), (x3, y3, v3), (x4, y4, v4)],
            'bbox': (cx, cy, w, h)
        }
    """
    
def calculate_bounding_box(landmarks, frame_width, frame_height):
    """Calculate bounding box from all visible landmarks.
    
    Returns:
        tuple: (center_x, center_y, width, height) normalized to 0-1
    """
    
def write_yolo_annotation(image_path, keypoints, bbox, output_dir):
    """Write annotation in YOLO format to corresponding .txt file."""
    
def compare_annotations(yolo_file, mediapipe_keypoints):
    """Compare YOLO ground truth with MediaPipe predictions.
    
    Returns:
        dict: Metrics including average distance, per-keypoint errors
    """
```

---

## Writing YOLO Format Annotations

### Complete Implementation

```python
import os
import numpy as np

def extract_yolo_keypoints(landmarks, lm_pose, frame_width, frame_height):
    """Extract 4 keypoints in YOLO format from MediaPipe landmarks.
    
    Args:
        landmarks: MediaPipe pose landmarks
        lm_pose: MediaPipe PoseLandmark enum
        frame_width: Image width in pixels
        frame_height: Image height in pixels
        
    Returns:
        dict with 'keypoints' (list of 4 tuples) and 'bbox' (tuple)
    """
    # Select visible side
    left_vis = landmarks.landmark[lm_pose.LEFT_SHOULDER].visibility
    right_vis = landmarks.landmark[lm_pose.RIGHT_SHOULDER].visibility
    use_left = left_vis >= right_vis
    
    # Define landmark indices based on visible side
    if use_left:
        shoulder_idx = lm_pose.LEFT_SHOULDER
        hip_idx = lm_pose.LEFT_HIP
        ankle_idx = lm_pose.LEFT_ANKLE
        ear_idx = lm_pose.LEFT_EAR
    else:
        shoulder_idx = lm_pose.RIGHT_SHOULDER
        hip_idx = lm_pose.RIGHT_HIP
        ankle_idx = lm_pose.RIGHT_ANKLE
        ear_idx = lm_pose.RIGHT_EAR
    
    # Extract keypoints (use NOSE for head, more stable than ears)
    nose_idx = lm_pose.NOSE
    
    keypoints = []
    
    # Keypoint 1: Cervical (NOSE or EAR)
    kp = landmarks.landmark[nose_idx]
    x_norm = kp.x
    y_norm = kp.y
    vis = 2 if kp.visibility > 0.5 else (1 if kp.visibility > 0.0 else 0)
    keypoints.append((x_norm, y_norm, vis))
    
    # Keypoint 2: Thoracic (SHOULDER)
    kp = landmarks.landmark[shoulder_idx]
    x_norm = kp.x
    y_norm = kp.y
    vis = 2 if kp.visibility > 0.5 else (1 if kp.visibility > 0.0 else 0)
    keypoints.append((x_norm, y_norm, vis))
    
    # Keypoint 3: Lumbar (HIP)
    kp = landmarks.landmark[hip_idx]
    x_norm = kp.x
    y_norm = kp.y
    vis = 2 if kp.visibility > 0.5 else (1 if kp.visibility > 0.0 else 0)
    keypoints.append((x_norm, y_norm, vis))
    
    # Keypoint 4: Sacral (ANKLE)
    kp = landmarks.landmark[ankle_idx]
    x_norm = kp.x
    y_norm = kp.y
    vis = 2 if kp.visibility > 0.5 else (1 if kp.visibility > 0.0 else 0)
    keypoints.append((x_norm, y_norm, vis))
    
    # Calculate bounding box
    bbox = calculate_bounding_box(landmarks, frame_width, frame_height)
    
    return {
        'keypoints': keypoints,
        'bbox': bbox
    }


def calculate_bounding_box(landmarks, frame_width, frame_height):
    """Calculate bounding box from all visible landmarks.
    
    Args:
        landmarks: MediaPipe pose landmarks
        frame_width: Image width in pixels
        frame_height: Image height in pixels
        
    Returns:
        tuple: (center_x_norm, center_y_norm, width_norm, height_norm)
    """
    # Collect all visible landmark coordinates
    x_coords = []
    y_coords = []
    
    for landmark in landmarks.landmark:
        if landmark.visibility > 0.5:
            x_coords.append(landmark.x)
            y_coords.append(landmark.y)
    
    if not x_coords:
        # No visible landmarks, return default bbox
        return (0.5, 0.5, 1.0, 1.0)
    
    # Calculate bounding box from min/max coordinates
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Add small padding (5%)
    padding = 0.05
    width = (x_max - x_min) * (1 + padding)
    height = (y_max - y_min) * (1 + padding)
    
    # Calculate center
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    
    # Ensure values are in valid range [0, 1]
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return (center_x, center_y, width, height)


def write_yolo_annotation(image_path, keypoints, bbox, output_dir):
    """Write annotation in YOLO format to corresponding .txt file.
    
    Args:
        image_path: Path to the image file
        keypoints: List of 4 tuples (x_norm, y_norm, visibility)
        bbox: Tuple (center_x_norm, center_y_norm, width_norm, height_norm)
        output_dir: Directory to write annotation files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image filename without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(output_dir, f"{image_name}.txt")
    
    # Format annotation line
    class_id = 0  # person
    cx, cy, w, h = bbox
    
    # Build annotation string
    annotation_parts = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    
    # Add keypoints
    for x, y, v in keypoints:
        annotation_parts.extend([f"{x:.6f}", f"{y:.6f}", str(int(v))])
    
    annotation_line = " ".join(annotation_parts)
    
    # Write to file
    with open(label_path, 'w') as f:
        f.write(annotation_line + "\n")
    
    return label_path
```

---

## Comparing Annotations

### Comparison Metrics

```python
import numpy as np

def parse_yolo_annotation(yolo_file):
    """Parse YOLO annotation file.
    
    Returns:
        dict: {
            'class_id': int,
            'bbox': (cx, cy, w, h),
            'keypoints': [(x1, y1, v1), (x2, y2, v2), (x3, y3, v3), (x4, y4, v4)]
        }
    """
    with open(yolo_file, 'r') as f:
        line = f.readline().strip()
    
    parts = line.split()
    
    class_id = int(parts[0])
    bbox = tuple(map(float, parts[1:5]))
    
    keypoints = []
    for i in range(5, len(parts), 3):
        x = float(parts[i])
        y = float(parts[i + 1])
        v = int(parts[i + 2])
        keypoints.append((x, y, v))
    
    return {
        'class_id': class_id,
        'bbox': bbox,
        'keypoints': keypoints
    }


def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points (normalized coordinates)."""
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compare_annotations(yolo_annotation, mediapipe_annotation):
    """Compare YOLO ground truth with MediaPipe predictions.
    
    Args:
        yolo_annotation: Dict from parse_yolo_annotation
        mediapipe_annotation: Dict from extract_yolo_keypoints
        
    Returns:
        dict: Comparison metrics
    """
    yolo_kpts = yolo_annotation['keypoints']
    mp_kpts = mediapipe_annotation['keypoints']
    
    # Calculate per-keypoint errors (normalized coordinates)
    keypoint_errors = []
    keypoint_names = ['Cervical', 'Thoracic', 'Lumbar', 'Sacral']
    
    for i, (name, yolo_kpt, mp_kpt) in enumerate(zip(keypoint_names, yolo_kpts, mp_kpts)):
        # Only compare if both keypoints are visible
        if yolo_kpt[2] >= 2 and mp_kpt[2] >= 2:
            distance = calculate_euclidean_distance(yolo_kpt, mp_kpt)
            keypoint_errors.append({
                'name': name,
                'index': i + 1,
                'distance': distance,
                'yolo': yolo_kpt[:2],
                'mediapipe': mp_kpt[:2]
            })
    
    # Calculate bounding box IoU
    yolo_bbox = yolo_annotation['bbox']
    mp_bbox = mediapipe_annotation['bbox']
    bbox_iou = calculate_bbox_iou(yolo_bbox, mp_bbox)
    
    # Average keypoint error
    avg_error = np.mean([e['distance'] for e in keypoint_errors]) if keypoint_errors else None
    
    return {
        'keypoint_errors': keypoint_errors,
        'average_error': avg_error,
        'bbox_iou': bbox_iou,
        'num_compared_keypoints': len(keypoint_errors)
    }


def calculate_bbox_iou(bbox1, bbox2):
    """Calculate IoU (Intersection over Union) for two bounding boxes.
    
    Args:
        bbox1, bbox2: (center_x, center_y, width, height) normalized
        
    Returns:
        float: IoU value (0-1)
    """
    # Convert from center format to corner format
    def center_to_corners(bbox):
        cx, cy, w, h = bbox
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = center_to_corners(bbox1)
    x1_2, y1_2, x2_2, y2_2 = center_to_corners(bbox2)
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def print_comparison_report(comparison, image_name):
    """Print a formatted comparison report."""
    print(f"\n{'='*60}")
    print(f"Comparison Report: {image_name}")
    print(f"{'='*60}")
    
    print(f"\nBounding Box IoU: {comparison['bbox_iou']:.4f}")
    print(f"Number of Compared Keypoints: {comparison['num_compared_keypoints']}/4")
    
    if comparison['average_error'] is not None:
        print(f"Average Keypoint Error: {comparison['average_error']:.6f} (normalized)")
        print(f"Average Keypoint Error (pixels @ 640x640): {comparison['average_error'] * 640:.2f} px")
    
    print(f"\nPer-Keypoint Errors:")
    print(f"{'Keypoint':<12} {'Index':<8} {'Distance':<12} {'YOLO (x,y)':<20} {'MediaPipe (x,y)':<20}")
    print(f"{'-'*80}")
    
    for error in comparison['keypoint_errors']:
        yolo_str = f"({error['yolo'][0]:.4f}, {error['yolo'][1]:.4f})"
        mp_str = f"({error['mediapipe'][0]:.4f}, {error['mediapipe'][1]:.4f})"
        print(f"{error['name']:<12} {error['index']:<8} {error['distance']:.6f}   {yolo_str:<20} {mp_str:<20}")
    
    print(f"{'='*60}\n")
```

---

## Implementation Example

### Modified `process_image` Function

Here's how to modify the existing `process_image` function to export YOLO annotations:

```python
def process_image_with_yolo_export(input_path, output_path, yolo_output_dir=None, 
                                   compare_with_yolo=None, display=True):
    """Process image and optionally export YOLO annotations.
    
    Args:
        input_path: Path to input image
        output_path: Path to save annotated image
        yolo_output_dir: Directory to save YOLO annotations (optional)
        compare_with_yolo: Path to existing YOLO annotation for comparison (optional)
        display: Whether to display the image
    """
    # Setting up MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {input_path}")
    
    frame_height, frame_width = image.shape[:2]
    
    # ... (existing visualization code) ...
    
    try:
        # Convert to RGB for MediaPipe processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(rgb_image)
        image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        landmarks = keypoints.pose_landmarks
        if landmarks is None:
            logger.info("No pose landmarks detected in the image.")
            # ... (existing code) ...
        else:
            # ... (existing posture analysis code) ...
            
            # NEW: Extract YOLO-format keypoints
            if yolo_output_dir is not None:
                yolo_data = extract_yolo_keypoints(landmarks, mp_pose.PoseLandmark, 
                                                   frame_width, frame_height)
                label_path = write_yolo_annotation(input_path, yolo_data['keypoints'], 
                                                   yolo_data['bbox'], yolo_output_dir)
                logger.info(f"YOLO annotation saved to: {label_path}")
                
                # NEW: Compare with existing YOLO annotation if provided
                if compare_with_yolo is not None and os.path.exists(compare_with_yolo):
                    yolo_annotation = parse_yolo_annotation(compare_with_yolo)
                    comparison = compare_annotations(yolo_annotation, yolo_data)
                    print_comparison_report(comparison, os.path.basename(input_path))
        
        # ... (existing save and display code) ...
        
    finally:
        pose.close()
        if display:
            cv2.destroyAllWindows()
```

### Batch Processing Script

```python
def batch_process_and_compare(image_dir, yolo_label_dir, output_annotation_dir, 
                              output_image_dir=None):
    """Process multiple images and compare with existing YOLO annotations.
    
    Args:
        image_dir: Directory containing input images
        yolo_label_dir: Directory containing existing YOLO labels
        output_annotation_dir: Directory to save MediaPipe YOLO-format annotations
        output_image_dir: Directory to save annotated images (optional)
    """
    import glob
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    all_comparisons = []
    
    for image_path in image_files:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        yolo_label_path = os.path.join(yolo_label_dir, f"{image_name}.txt")
        
        # Output path for annotated image
        output_path = None
        if output_image_dir:
            os.makedirs(output_image_dir, exist_ok=True)
            output_path = os.path.join(output_image_dir, os.path.basename(image_path))
        
        # Check if YOLO label exists
        compare_with = yolo_label_path if os.path.exists(yolo_label_path) else None
        
        try:
            # Process image
            process_image_with_yolo_export(
                input_path=image_path,
                output_path=output_path,
                yolo_output_dir=output_annotation_dir,
                compare_with_yolo=compare_with,
                display=False
            )
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    logger.info("Batch processing complete")
```

---

## Usage Instructions

### 1. Export MediaPipe Annotations in YOLO Format

```bash
# Process a single image and export YOLO annotation
python code/human_posture_analysis.py \
    --mode image \
    --input-video ./data/posture/images/train/image001.jpg \
    --output-video ./output/annotated/image001.jpg \
    --yolo-output-dir ./output/yolo_annotations/
```

### 2. Compare with Existing YOLO Annotations

```bash
# Process and compare with existing annotation
python code/human_posture_analysis.py \
    --mode image \
    --input-video ./data/posture/images/train/image001.jpg \
    --output-video ./output/annotated/image001.jpg \
    --yolo-output-dir ./output/yolo_annotations/ \
    --compare-yolo ./data/posture/labels/train/image001.txt
```

### 3. Batch Process All Images

```python
# In Python script or notebook
from human_posture_analysis import batch_process_and_compare

batch_process_and_compare(
    image_dir='./data/posture/images/train/',
    yolo_label_dir='./data/posture/labels/train/',
    output_annotation_dir='./output/mediapipe_yolo_annotations/',
    output_image_dir='./output/annotated_images/'
)
```

### 4. Analyze Comparison Results

After batch processing, you can aggregate the comparison metrics:

```python
import json
import numpy as np

def aggregate_comparison_metrics(annotation_dir, yolo_dir):
    """Aggregate metrics across all comparisons."""
    
    metrics = {
        'total_images': 0,
        'successful_comparisons': 0,
        'average_errors': [],
        'bbox_ious': [],
        'per_keypoint_errors': {
            'Cervical': [],
            'Thoracic': [],
            'Lumbar': [],
            'Sacral': []
        }
    }
    
    # ... (implementation to aggregate results) ...
    
    return metrics
```

---

## Adapting the Script to Collect Same YOLO Landmarks

### Summary of Required Changes

To adapt `code/human_posture_analysis.py` to collect the same landmarks as YOLO annotations:

1. **Landmark Selection**:
   - YOLO Keypoint 1 (Cervical) → MediaPipe NOSE (0) or LEFT_EAR (7)
   - YOLO Keypoint 2 (Thoracic) → MediaPipe LEFT_SHOULDER (11)
   - YOLO Keypoint 3 (Lumbar) → MediaPipe LEFT_HIP (23)
   - YOLO Keypoint 4 (Sacral) → MediaPipe LEFT_ANKLE (27)

2. **Side Selection**: Use visibility scores to choose left or right landmarks for side-view images

3. **Coordinate Normalization**: MediaPipe already provides normalized coordinates (0-1), matching YOLO format

4. **Visibility Mapping**: Convert MediaPipe visibility (0.0-1.0) to YOLO visibility (0, 1, 2)

5. **Bounding Box Calculation**: Compute from all visible landmarks with padding

6. **File Output**: Write in YOLO format: `class_id cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4`

### Key Implementation Points

```python
# Add these functions to human_posture_analysis.py:
# 1. extract_yolo_keypoints() - Extract 4 keypoints from MediaPipe
# 2. calculate_bounding_box() - Compute bbox from landmarks
# 3. write_yolo_annotation() - Write YOLO format file
# 4. parse_yolo_annotation() - Read existing YOLO file
# 5. compare_annotations() - Calculate comparison metrics

# Modify process_image() to accept:
# - yolo_output_dir: Where to save annotations
# - compare_with_yolo: Path to existing YOLO label for comparison

# Add command-line arguments:
# --yolo-output-dir: Enable YOLO export
# --compare-yolo: Enable comparison with existing labels
# --batch-process: Process entire directory
```

---

## Conclusion

This guide provides a complete methodology to:

1. **Export MediaPipe detections in YOLO format** by mapping 4 key landmarks (cervical, thoracic, lumbar, sacral spine regions)

2. **Compare MediaPipe and YOLO annotations** using Euclidean distance for keypoints and IoU for bounding boxes

3. **Validate MediaPipe against ground truth** to assess detection accuracy on the posture dataset

The implementation enables quantitative evaluation of MediaPipe's pose detection performance compared to the manually-annotated YOLO dataset, specifically for side-view spinal alignment analysis.

### Next Steps

1. Implement the functions in `code/human_posture_analysis.py`
2. Test on a subset of images from `data/posture/images/train/`
3. Analyze comparison metrics to understand detection differences
4. Document findings on accuracy and suitability for spinal alignment analysis

---

## References

- [YOLO Pose Format Documentation](https://docs.ultralytics.com/tasks/pose/)
- [MediaPipe Pose Detection](https://developers.google.com/ml-kit/vision/pose-detection)
- [Existing comparison document](pose_format_comparison.md)
- [Kaggle Posture Dataset](https://www.kaggle.com/datasets/melsmm/posture-keypoints-detection)
