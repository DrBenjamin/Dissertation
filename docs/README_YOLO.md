# Documentation

This directory contains supplementary documentation for the dissertation project.

## Contents

### [Pose Format Comparison](pose_format_comparison.md)

A comprehensive comparison between YOLO Pose format (used in the dataset annotations) and Google MediaPipe Pose Detection (used in the runtime analysis).

**Key topics covered:**
- YOLO pose annotation format (4 keypoints)
- MediaPipe pose detection (33 landmarks)
- Format specifications and differences
- Keypoint mapping between systems
- Usage in the project workflow
- Code examples and references

This document was created to explain the relationship between the training dataset annotations (`data/posture/labels/`) and the pose detection implementation (`code/human_posture_analysis.py`).

### [MediaPipe to YOLO Annotation Guide](mediapipe_to_yolo_annotation_guide.md)

A practical implementation guide for adapting `code/human_posture_analysis.py` to export MediaPipe detections in YOLO format and compare them with existing YOLO annotations.

**Key topics covered:**
- Landmark mapping strategy (MediaPipe 33 landmarks → YOLO 4 keypoints)
- Complete code implementations for annotation export
- Comparison metrics and validation methods
- Batch processing workflows
- Usage instructions and examples
- Adapting the script to collect same YOLO landmarks for cervical, thoracic, lumbar, and sacral spine regions

This guide provides the practical solution to comparing MediaPipe pose detection against the YOLO ground truth annotations in the posture dataset.

---

For the main project documentation, see the [README.md](../README.md) in the project root.
