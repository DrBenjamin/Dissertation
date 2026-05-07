"""CLI demo: Klein-Vogelbach 4-body-block overlay annotation for static images.

Runs MediaPipe Pose locally on an input image (or a directory of images),
aggregates the 33 landmarks into the four Klein-Vogelbach functional
blocks (head, thorax, pelvis, lower extremities), classifies posture as
normal / hypotonic / hypertonic from the inter-block geometry, and writes
an annotated image. Optional JSON sidecar with the full numerical result.

Examples:
    python code/klein_vogelbach_overlay.py \\
        --input data/images/people/IMG_0325.jpg \\
        --output data/images/IMG_0325_kv.png

    python code/klein_vogelbach_overlay.py \\
        --input "data/images/posture/normal posture" \\
        --output data/images/kv_out \\
        --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import urllib.request
from dataclasses import asdict
from typing import List, Optional

import cv2
import numpy as np

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from klein_vogelbach_blocks import (  # noqa: E402
    KleinVogelbachThresholds,
    analyze_image,
    draw_block_overlay,
    landmarks_from_mediapipe,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


def _ensure_pose_model() -> str:
    """Download MediaPipe Pose Landmarker model if not already cached."""
    cache_dir = os.path.join(tempfile.gettempdir(), "mediapipe_models")
    model_path = os.path.join(cache_dir, "pose_landmarker_heavy.task")
    if not os.path.exists(model_path):
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Downloading MediaPipe Pose model to %s", model_path)
        urllib.request.urlretrieve(POSE_LANDMARKER_MODEL_URL, model_path)
    return model_path


class _PoseRunner:
    """Wrapper that hides differences between legacy and Tasks MediaPipe APIs."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "mediapipe is required. Install it (e.g. `pip install mediapipe==0.10.33`) "
                "or run inside the project's conda environment."
            ) from exc

        self._mp = __import__("mediapipe")
        self._legacy = None
        self._tasks_landmarker = None

        if hasattr(self._mp, "solutions") and hasattr(self._mp.solutions, "pose"):
            self._legacy = self._mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
            logger.info("Using mediapipe.solutions.pose for landmark detection.")
        else:
            model_path = _ensure_pose_model()
            base_options = self._mp.tasks.BaseOptions(model_asset_path=model_path)
            options = self._mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=self._mp.tasks.vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self._tasks_landmarker = self._mp.tasks.vision.PoseLandmarker.create_from_options(options)
            logger.info("Using mediapipe.tasks.vision.PoseLandmarker for landmark detection.")

    def detect(self, bgr_image: np.ndarray):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self._legacy is not None:
            result = self._legacy.process(rgb)
            return result.pose_landmarks
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._tasks_landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def close(self) -> None:
        if self._legacy is not None:
            self._legacy.close()
        if self._tasks_landmarker is not None:
            self._tasks_landmarker.close()


def _collect_inputs(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        files: List[str] = []
        for name in sorted(os.listdir(input_path)):
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                files.append(os.path.join(input_path, name))
        return files
    if os.path.isfile(input_path):
        return [input_path]
    raise SystemExit(f"Input path not found: {input_path}")


def _resolve_output_path(input_file: str, output: str, batch: bool) -> str:
    if batch:
        os.makedirs(output, exist_ok=True)
        base, ext = os.path.splitext(os.path.basename(input_file))
        out_ext = ext if ext.lower() in IMAGE_EXTENSIONS else ".png"
        return os.path.join(output, f"{base}_kv{out_ext}")
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return output


def _result_to_dict(result) -> dict:
    payload = asdict(result)
    payload["blocks"] = {name: asdict(block) for name, block in result.blocks.items()}
    return payload


def process_one(
    runner: _PoseRunner,
    input_file: str,
    output_file: str,
    include_arms_in_thorax: bool,
    thresholds: KleinVogelbachThresholds,
    write_json: bool,
    show_bbox: bool,
) -> Optional[dict]:
    image = cv2.imread(input_file)
    if image is None:
        logger.error("Failed to read image: %s", input_file)
        return None

    mp_landmarks = runner.detect(image)
    if mp_landmarks is None:
        logger.warning("No pose detected in %s — writing original image with note.", input_file)
        annotated = image.copy()
        cv2.putText(
            annotated,
            "No pose detected",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(output_file, annotated)
        return None

    landmarks = landmarks_from_mediapipe(mp_landmarks)
    _, result = analyze_image(
        image,
        landmarks,
        include_arms_in_thorax=include_arms_in_thorax,
        thresholds=thresholds,
    )
    annotated = draw_block_overlay(image, result, show_bbox=show_bbox)
    cv2.imwrite(output_file, annotated)
    logger.info(
        "Wrote %s | class=%s confidence=%.2f reasoning=%s",
        output_file,
        result.posture_class,
        result.confidence,
        result.reasoning,
    )

    payload = _result_to_dict(result)
    if write_json:
        json_path = os.path.splitext(output_file)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote JSON sidecar: %s", json_path)
    return payload


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Klein-Vogelbach 4-body-block overlay annotation (post-inference aggregation).",
    )
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory.")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output image path (file mode) or directory (batch mode).",
    )
    parser.add_argument(
        "--include-arms-in-thorax",
        action="store_true",
        help="Fold arm landmarks (13-22) into the thorax block centroid.",
    )
    parser.add_argument(
        "--no-bbox",
        action="store_true",
        help="Hide per-block bounding boxes; only draw centroids and chain.",
    )
    parser.add_argument(
        "--normal-max-offset",
        type=float,
        default=0.04,
        help="Max normalised |offset| considered normal posture (default: 0.04).",
    )
    parser.add_argument(
        "--hypertonic-min-offset",
        type=float,
        default=0.06,
        help="Min normalised |offset| for hypertonic when all signs agree (default: 0.06).",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.3,
        help="Per-landmark visibility threshold for inclusion in a block (default: 0.3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write a JSON sidecar with the full numerical result next to each output image.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    inputs = _collect_inputs(args.input)
    if not inputs:
        logger.error("No image files found at %s", args.input)
        return 1

    batch_mode = os.path.isdir(args.input)
    thresholds = KleinVogelbachThresholds(
        normal_max_offset=args.normal_max_offset,
        hypertonic_min_offset=args.hypertonic_min_offset,
        min_visibility=args.min_visibility,
    )

    runner = _PoseRunner()
    try:
        for input_file in inputs:
            output_file = _resolve_output_path(input_file, args.output, batch=batch_mode)
            process_one(
                runner=runner,
                input_file=input_file,
                output_file=output_file,
                include_arms_in_thorax=args.include_arms_in_thorax,
                thresholds=thresholds,
                write_json=args.json,
                show_bbox=not args.no_bbox,
            )
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
