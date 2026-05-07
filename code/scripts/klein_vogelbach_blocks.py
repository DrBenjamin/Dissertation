"""Klein-Vogelbach 4-body-block aggregation and posture classification.

Aggregates the 33 MediaPipe Pose landmarks into four functional segments
(head, thorax, pelvis, lower extremities) and derives Klein-Vogelbach
posture categories deterministically from the inter-block geometry.

Posture categories:
    - "normal":     all segment centroids close to the same vertical axis
                    (integrated alignment).
    - "hypotonic":  alternating anterior/posterior shifts across segments
                    (anterior-posterior dissociation, classic slumped posture).
    - "hypertonic": all upper segments shifted in the same anterior direction
                    relative to the lower-extremity base (global anterior
                    load shift, classic tensed/military posture).

The implementation is purely geometric and runs on top of any pose
landmark source exposing (x, y) in normalised image coordinates plus
visibility -- the standard MediaPipe Pose output. It is documented for
side-view input: the reference plumb line passes through the
lower-extremity centroid, and "anterior" is inferred from the nose
direction relative to that plumb line.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# MediaPipe Pose landmark indices grouped into Klein-Vogelbach functional blocks.
# Notes:
#   - "head" covers the face landmarks (nose, eyes, ears, mouth).
#   - "thorax" defaults to the shoulder landmarks only because the
#     Klein-Vogelbach thoracic block is the rib cage, not the arms.
#     Arm landmarks can be folded in via include_arms_in_thorax=True.
#   - "pelvis" covers the two hip landmarks.
#   - "lower_extremities" covers knees, ankles, heels, and foot indices.
BLOCK_LANDMARK_INDICES: Dict[str, List[int]] = {
    "head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "thorax": [11, 12],
    "pelvis": [23, 24],
    "lower_extremities": [25, 26, 27, 28, 29, 30, 31, 32],
}

ARM_LANDMARK_INDICES: List[int] = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

BLOCK_DRAW_ORDER: List[str] = ["head", "thorax", "pelvis", "lower_extremities"]

# BGR colors per block (OpenCV convention).
BLOCK_COLORS: Dict[str, Tuple[int, int, int]] = {
    "head": (255, 200, 0),
    "thorax": (0, 200, 255),
    "pelvis": (180, 100, 255),
    "lower_extremities": (100, 220, 100),
}

POSTURE_LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "normal": (80, 220, 80),
    "hypotonic": (60, 160, 240),
    "hypertonic": (60, 60, 240),
    "indeterminate": (200, 200, 200),
}

NOSE_LANDMARK_INDEX: int = 0
LEFT_EAR_LANDMARK_INDEX: int = 7
RIGHT_EAR_LANDMARK_INDEX: int = 8
LEFT_ANKLE_LANDMARK_INDEX: int = 27
RIGHT_ANKLE_LANDMARK_INDEX: int = 28


@dataclass
class Landmark:
    """Lightweight, MediaPipe-compatible landmark in normalised image space."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class BodyBlock:
    name: str
    centroid_px: Tuple[float, float]
    bbox_px: Tuple[float, float, float, float]
    mean_visibility: float
    landmark_count: int


@dataclass
class KleinVogelbachThresholds:
    """Thresholds expressed as fractions of the detected body height in pixels."""
    normal_max_offset: float = 0.04
    hypertonic_min_offset: float = 0.06
    min_visibility: float = 0.3


@dataclass
class KleinVogelbachResult:
    posture_class: str
    confidence: float
    horizontal_offsets_px: Dict[str, float]
    normalized_offsets: Dict[str, float]
    body_height_px: float
    anterior_direction: int
    plumb_line_x_px: float
    reasoning: str
    blocks: Dict[str, BodyBlock] = field(default_factory=dict)


def get_block_indices(include_arms_in_thorax: bool = False) -> Dict[str, List[int]]:
    """Return landmark-index groups, optionally folding arm landmarks into thorax."""
    indices = {name: list(idxs) for name, idxs in BLOCK_LANDMARK_INDICES.items()}
    if include_arms_in_thorax:
        indices["thorax"] = sorted(indices["thorax"] + ARM_LANDMARK_INDICES)
    return indices


def landmarks_from_mediapipe(mp_landmarks, expected_count: int = 33) -> List[Landmark]:
    """Convert a MediaPipe pose result into a list of Landmark objects.

    Accepts either the modern Tasks API list-of-NormalizedLandmark or the
    legacy ``mp.solutions.pose`` result that exposes ``.landmark``.
    """
    if mp_landmarks is None:
        return []
    iterable: Iterable = mp_landmarks.landmark if hasattr(mp_landmarks, "landmark") else mp_landmarks
    out: List[Landmark] = []
    for lm in iterable:
        out.append(
            Landmark(
                x=float(getattr(lm, "x", 0.0)),
                y=float(getattr(lm, "y", 0.0)),
                z=float(getattr(lm, "z", 0.0)),
                visibility=float(getattr(lm, "visibility", 1.0)),
            )
        )
    if 0 < len(out) < expected_count:
        logger.warning(
            "Received %d landmarks, expected %d. Higher indices will be skipped.",
            len(out),
            expected_count,
        )
    return out


def _aggregate_one_block(
    name: str,
    landmarks: Sequence[Landmark],
    indices: Sequence[int],
    image_w: int,
    image_h: int,
    min_visibility: float,
) -> Optional[BodyBlock]:
    xs: List[float] = []
    ys: List[float] = []
    vis: List[float] = []
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        if lm.visibility < min_visibility:
            continue
        xs.append(lm.x * image_w)
        ys.append(lm.y * image_h)
        vis.append(lm.visibility)

    if not xs:
        return None

    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    vis_arr = np.asarray(vis)

    return BodyBlock(
        name=name,
        centroid_px=(float(xs_arr.mean()), float(ys_arr.mean())),
        bbox_px=(
            float(xs_arr.min()),
            float(ys_arr.min()),
            float(xs_arr.max()),
            float(ys_arr.max()),
        ),
        mean_visibility=float(vis_arr.mean()),
        landmark_count=len(xs),
    )


def aggregate_blocks(
    landmarks: Sequence[Landmark],
    image_w: int,
    image_h: int,
    include_arms_in_thorax: bool = False,
    min_visibility: float = 0.3,
) -> Dict[str, BodyBlock]:
    """Aggregate 33 landmarks into 4 Klein-Vogelbach blocks (centroid + bbox)."""
    indices = get_block_indices(include_arms_in_thorax=include_arms_in_thorax)
    blocks: Dict[str, BodyBlock] = {}
    for name, idxs in indices.items():
        block = _aggregate_one_block(name, landmarks, idxs, image_w, image_h, min_visibility)
        if block is not None:
            blocks[name] = block
    return blocks


def _detect_anterior_direction(
    landmarks: Sequence[Landmark],
    plumb_line_x_px: float,
    image_w: int,
    fallback_head_centroid_x: Optional[float] = None,
    min_visibility: float = 0.3,
) -> int:
    """Infer anatomical anterior (face-facing) direction in image x.

    Returns +1 (anterior is right) or -1 (anterior is left).

    Strategy (most → least reliable):
        1. nose_x vs mean(ear_x): the nose is anterior to the ears in any
           anatomical side view, independent of posture. Most robust.
        2. nose_x vs plumb_line_x: usable when ears aren't visible, but can
           flip in extreme postures (e.g. severe head retraction) — only a
           fallback.
        3. head_centroid_x vs plumb_line_x: weakest fallback.
    """
    nose: Optional[Landmark] = None
    if NOSE_LANDMARK_INDEX < len(landmarks):
        candidate = landmarks[NOSE_LANDMARK_INDEX]
        if candidate.visibility >= min_visibility:
            nose = candidate

    ear_xs: List[float] = []
    for idx in (LEFT_EAR_LANDMARK_INDEX, RIGHT_EAR_LANDMARK_INDEX):
        if idx < len(landmarks) and landmarks[idx].visibility >= min_visibility:
            ear_xs.append(landmarks[idx].x * image_w)

    if nose is not None and ear_xs:
        delta = nose.x * image_w - (sum(ear_xs) / len(ear_xs))
        if abs(delta) > 1.0:
            return 1 if delta > 0 else -1

    if nose is not None:
        delta = nose.x * image_w - plumb_line_x_px
        if abs(delta) > 1.0:
            return 1 if delta > 0 else -1

    if fallback_head_centroid_x is not None:
        delta = fallback_head_centroid_x - plumb_line_x_px
        if abs(delta) > 1.0:
            return 1 if delta > 0 else -1
    return 1


def _compute_plumb_line_x(
    landmarks: Sequence[Landmark],
    blocks: Dict[str, BodyBlock],
    image_w: int,
    min_visibility: float = 0.3,
) -> float:
    """Plumb-line x in pixels: mean ankle x (lateral malleolus reference).

    The Klein-Vogelbach / standard postural-assessment plumb line passes
    through the lateral malleolus. Falls back to the lower-extremity block
    centroid only when the ankles are not visible.
    """
    ankle_xs: List[float] = []
    for idx in (LEFT_ANKLE_LANDMARK_INDEX, RIGHT_ANKLE_LANDMARK_INDEX):
        if idx < len(landmarks) and landmarks[idx].visibility >= min_visibility:
            ankle_xs.append(landmarks[idx].x * image_w)
    if ankle_xs:
        return float(sum(ankle_xs) / len(ankle_xs))
    if "lower_extremities" in blocks:
        return float(blocks["lower_extremities"].centroid_px[0])
    return float(image_w) / 2.0


def classify_klein_vogelbach(
    blocks: Dict[str, BodyBlock],
    landmarks: Sequence[Landmark],
    image_w: int,
    image_h: int,
    thresholds: Optional[KleinVogelbachThresholds] = None,
) -> KleinVogelbachResult:
    """Classify posture into normal / hypotonic / hypertonic from block geometry.

    Decision logic (side-view assumption):
        1. Plumb line = vertical line through the mean ankle x (lateral
           malleolus). Falls back to the lower-extremity block centroid
           only when ankles aren't visible.
        2. Anterior direction = sign(nose_x - mean_ear_x), i.e. derived
           anatomically. This is independent of posture and won't flip on
           severe head retraction or anterior shift. Falls back to nose-vs-
           plumb if ears aren't visible.
        3. Signed anterior offset per upper block = (centroid_x - plumb_line_x) * anterior_direction.
        4. Normalise by body height (max_y - min_y of all aggregated centroids+bboxes).
        5. If max |offset| <= normal_max_offset -> "normal".
        6. Else if all upper-block signed offsets >= 0 AND max >= hypertonic_min_offset
           -> "hypertonic" (uniform anterior load shift).
        7. Else -> "hypotonic" (mixed-sign / dissociated shifts).
    """
    thresholds = thresholds or KleinVogelbachThresholds()

    required = {"head", "thorax", "pelvis", "lower_extremities"}
    missing = required - set(blocks.keys())
    if missing:
        return KleinVogelbachResult(
            posture_class="indeterminate",
            confidence=0.0,
            horizontal_offsets_px={},
            normalized_offsets={},
            body_height_px=0.0,
            anterior_direction=1,
            plumb_line_x_px=0.0,
            reasoning=f"Missing blocks (visibility too low?): {sorted(missing)}",
            blocks=blocks,
        )

    plumb_x = _compute_plumb_line_x(
        landmarks,
        blocks,
        image_w=image_w,
        min_visibility=thresholds.min_visibility,
    )

    all_y = []
    for block in blocks.values():
        _, y_min, _, y_max = block.bbox_px
        all_y.extend([y_min, y_max])
    body_height_px = max(all_y) - min(all_y)
    if body_height_px <= 1.0:
        return KleinVogelbachResult(
            posture_class="indeterminate",
            confidence=0.0,
            horizontal_offsets_px={},
            normalized_offsets={},
            body_height_px=body_height_px,
            anterior_direction=1,
            plumb_line_x_px=plumb_x,
            reasoning="Degenerate body height; cannot normalise offsets.",
            blocks=blocks,
        )

    anterior_dir = _detect_anterior_direction(
        landmarks,
        plumb_line_x_px=plumb_x,
        image_w=image_w,
        fallback_head_centroid_x=blocks["head"].centroid_px[0],
        min_visibility=thresholds.min_visibility,
    )

    upper_block_names = ["head", "thorax", "pelvis"]
    horizontal_offsets_px: Dict[str, float] = {}
    normalized_offsets: Dict[str, float] = {}
    for name in upper_block_names:
        raw = blocks[name].centroid_px[0] - plumb_x
        signed_anterior = raw * anterior_dir
        horizontal_offsets_px[name] = float(signed_anterior)
        normalized_offsets[name] = float(signed_anterior / body_height_px)

    horizontal_offsets_px["lower_extremities"] = 0.0
    normalized_offsets["lower_extremities"] = 0.0

    upper_norm = [normalized_offsets[n] for n in upper_block_names]
    abs_norm = [abs(v) for v in upper_norm]
    max_abs = max(abs_norm)

    posture_class: str
    confidence: float
    reasoning: str

    if max_abs <= thresholds.normal_max_offset:
        posture_class = "normal"
        margin = thresholds.normal_max_offset - max_abs
        confidence = float(min(1.0, margin / max(thresholds.normal_max_offset, 1e-6)))
        reasoning = (
            f"All upper-segment offsets within normal range "
            f"(max |offset| = {max_abs:.3f} <= {thresholds.normal_max_offset:.3f})."
        )
    elif all(v >= 0 for v in upper_norm) and max_abs >= thresholds.hypertonic_min_offset:
        posture_class = "hypertonic"
        margin = max_abs - thresholds.hypertonic_min_offset
        confidence = float(min(1.0, margin / max(thresholds.hypertonic_min_offset, 1e-6) + 0.5))
        reasoning = (
            "All upper segments (head, thorax, pelvis) shifted anteriorly relative to "
            f"the ankle plumb line; max |offset| = {max_abs:.3f}."
        )
    else:
        posture_class = "hypotonic"
        sign_changes = sum(
            1
            for a, b in zip(upper_norm, upper_norm[1:])
            if (a > 0) != (b > 0) and abs(a) > thresholds.normal_max_offset / 2
            and abs(b) > thresholds.normal_max_offset / 2
        )
        confidence = float(min(1.0, 0.4 + 0.3 * sign_changes + (max_abs - thresholds.normal_max_offset)))
        reasoning = (
            "Mixed anterior/posterior shifts across upper segments "
            f"(offsets {[round(v, 3) for v in upper_norm]}) — "
            "interpreted as anterior-posterior dissociation."
        )

    return KleinVogelbachResult(
        posture_class=posture_class,
        confidence=max(0.0, min(1.0, confidence)),
        horizontal_offsets_px=horizontal_offsets_px,
        normalized_offsets=normalized_offsets,
        body_height_px=float(body_height_px),
        anterior_direction=anterior_dir,
        plumb_line_x_px=float(plumb_x),
        reasoning=reasoning,
        blocks=blocks,
    )


def draw_block_overlay(
    image: np.ndarray,
    result: KleinVogelbachResult,
    show_bbox: bool = True,
    show_plumb_line: bool = True,
    show_centroid_chain: bool = True,
    show_label: bool = True,
    expand_blocks: bool = True,
    block_gap_px: int = 10,
) -> np.ndarray:
    """Render a Klein-Vogelbach 4-block overlay on a BGR image (in-place safe copy)."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # Optionally compute expanded, vertically stacked block rectangles that
    # nearly touch (small configurable gap) to match the Klein-Vogelbach model
    # depiction more closely than tight landmark-bounded boxes.
    expanded_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    if expand_blocks and result.blocks:
        present_names: List[str] = [n for n in BLOCK_DRAW_ORDER if n in result.blocks]
        if present_names:
            # Global horizontal extent across all present blocks (tight bounds)
            x_mins = [result.blocks[n].bbox_px[0] for n in present_names]
            x_maxs = [result.blocks[n].bbox_px[2] for n in present_names]
            x_min_global = float(min(x_mins))
            x_max_global = float(max(x_maxs))

            # Expand horizontally to cover area around the body, with extra space
            # towards the posterior side (opposite to the detected anterior direction).
            # Symmetric base padding: ~5% image width; posterior extra reduced to ~10% (−33%).
            base_pad = 0.05 * w
            posterior_extra = 0.10 * w
            if getattr(result, "anterior_direction", 1) >= 0:
                # Anterior points to the right -> posterior on the left: expand more on left
                x_min_global = x_min_global - (base_pad + posterior_extra)
                x_max_global = x_max_global + base_pad
            else:
                # Anterior points to the left -> posterior on the right: expand more on right
                x_min_global = x_min_global - base_pad
                x_max_global = x_max_global + (base_pad + posterior_extra)

            # Clamp to image bounds with a tiny safety margin
            x_min_global = max(0.0, x_min_global)
            x_max_global = min(float(w - 1), x_max_global)

            # Decrease width from the posterior side by 10% (cut only from back).
            width = max(1.0, x_max_global - x_min_global)
            trim = 0.10 * width
            if getattr(result, "anterior_direction", 1) >= 0:
                # Anterior to the right, posterior on the left -> move left edge rightwards
                x_min_global = min(x_max_global - 1.0, x_min_global + trim)
            else:
                # Anterior to the left, posterior on the right -> move right edge leftwards
                x_max_global = max(x_min_global + 1.0, x_max_global - trim)

            # Vertical ordering via centroid y; top and bottom anchors from bboxes
            centroids_y = [result.blocks[n].centroid_px[1] for n in present_names]
            y_tops = [result.blocks[n].bbox_px[1] for n in present_names]
            y_bottoms = [result.blocks[n].bbox_px[3] for n in present_names]

            # Ensure order is top-to-bottom by centroid
            ordered = sorted(
                zip(present_names, centroids_y, y_tops, y_bottoms), key=lambda t: t[1]
            )
            names_ord = [t[0] for t in ordered]
            cents_ord = [float(t[1]) for t in ordered]
            y_topmost = max(0.0, float(min(t[2] for t in ordered)))
            y_bottommost = min(float(h - 1), float(max(t[3] for t in ordered)))

            # Internal boundaries: midpoints between adjacent centroids
            boundaries: List[float] = []
            for i in range(len(names_ord) - 1):
                b = 0.5 * (cents_ord[i] + cents_ord[i + 1])
                boundaries.append(b)

            # Build expanded rectangles with a configurable gap between blocks
            half_gap = max(0, int(round(block_gap_px))) / 2.0

            # Enlarge thorax by nudging the thorax–pelvis boundary downward a bit
            # (and thus reduce pelvis height), when both are present.
            body_h_for_v = float(result.body_height_px) if getattr(result, "body_height_px", 0) else float(y_bottommost - y_topmost)
            thorax_extra_down = 0.05 * body_h_for_v  # ~5% body height
            if "thorax" in names_ord:
                idx_th = names_ord.index("thorax")
                # boundary after thorax is between thorax and the next block
                if 0 <= idx_th < len(boundaries):
                    new_b = boundaries[idx_th] + thorax_extra_down
                    lower_lim = (y_topmost if idx_th == 0 else boundaries[idx_th - 1]) + 2.0 * half_gap
                    upper_lim = (y_bottommost if idx_th == len(boundaries) - 1 else boundaries[idx_th + 1]) - 2.0 * half_gap
                    boundaries[idx_th] = float(max(lower_lim, min(new_b, upper_lim)))

            # Shorten pelvis so legs start earlier by moving the pelvis–legs
            # boundary upward (~8% of body height) when both are present.
            pelvis_legs_up = 0.08 * body_h_for_v
            if "pelvis" in names_ord:
                idx_pelvis = names_ord.index("pelvis")
                # boundary after pelvis is between pelvis and the next block
                if 0 <= idx_pelvis < len(boundaries):
                    new_b = boundaries[idx_pelvis] - pelvis_legs_up
                    lower_lim = (y_topmost if idx_pelvis == 0 else boundaries[idx_pelvis - 1]) + 2.0 * half_gap
                    upper_lim = (y_bottommost if idx_pelvis == len(boundaries) - 1 else boundaries[idx_pelvis + 1]) - 2.0 * half_gap
                    boundaries[idx_pelvis] = float(max(lower_lim, min(new_b, upper_lim)))
            # Parameters for per-block horizontal sizing anchored at the anterior edge.
            # Place centroid at a fixed fraction from the anterior side so most width
            # extends posteriorly. This makes the front boundary "touch" the body.
            anterior_fraction = 0.20  # centroid sits ~20% from anterior edge
            min_block_width_px = 0.08 * float(w)  # avoid degenerate very thin blocks
            for i, name in enumerate(names_ord):
                y_top = y_topmost if i == 0 else boundaries[i - 1] + half_gap
                y_bot = y_bottommost if i == len(names_ord) - 1 else boundaries[i] - half_gap

                # Clip and ensure minimum height
                y_top = float(max(0.0, min(y_top, h - 1)))
                y_bot = float(max(0.0, min(y_bot, h - 1)))

                # Raise the head block further above skull level: start with the
                # larger of 60% of head height or 12% of body height, then add
                # another +60% of that last increase (i.e., 1.6x total).
                if name == "head":
                    head_h = max(1.0, (y_bot - y_top))
                    desired_raise = max(0.60 * head_h, 0.12 * body_h_for_v)
                    desired_raise *= 1.60
                    y_top = max(0.0, y_top - desired_raise)
                if y_bot <= y_top:
                    # Fallback to original bbox if degenerate
                    y_top = float(result.blocks[name].bbox_px[1])
                    y_bot = float(result.blocks[name].bbox_px[3])

                # Compute per-block horizontal extent anchored at the anterior edge
                # of this block's original bbox so the front boundary touches the body.
                blk_x_min, _, blk_x_max, _ = result.blocks[name].bbox_px
                cx = float(result.blocks[name].centroid_px[0])
                base_width = max(min_block_width_px, float(blk_x_max - blk_x_min))
                if getattr(result, "anterior_direction", 1) >= 0:
                    # Anterior to the right: front at blk_x_max
                    front = float(blk_x_max)
                    if name == "lower_extremities":
                        width_total = base_width
                    else:
                        dx_front = max(0.0, front - cx)
                        width_total = max(min_block_width_px, dx_front / max(1e-6, anterior_fraction))
                    x_min_b = front - width_total
                    x_max_b = front
                else:
                    # Anterior to the left: front at blk_x_min
                    front = float(blk_x_min)
                    if name == "lower_extremities":
                        width_total = base_width
                    else:
                        dx_front = max(0.0, cx - front)
                        width_total = max(min_block_width_px, dx_front / max(1e-6, anterior_fraction))
                    x_min_b = front
                    x_max_b = front + width_total

                # Apply per-block front/back expansions to better cover body boundaries
                width_now = max(1.0, x_max_b - x_min_b)
                back_mult = 0.0
                front_mult = 0.0
                if name == "head":
                    back_mult, front_mult = 0.50, 0.00
                elif name == "thorax":
                    back_mult, front_mult = 0.00, 0.78  # +30% from 0.60
                elif name == "pelvis":
                    back_mult, front_mult = 0.00, 0.78  # +30% from 0.60
                elif name == "lower_extremities":
                    back_mult, front_mult = 0.20, 0.50

                back_px = back_mult * width_now
                front_px = front_mult * width_now

                if getattr(result, "anterior_direction", 1) >= 0:
                    # Anterior to the right: posterior is left, front is right
                    x_min_b -= back_px
                    x_max_b += front_px
                else:
                    # Anterior to the left: posterior is right, front is left
                    x_min_b -= front_px
                    x_max_b += back_px

                # Clamp to image bounds and ensure minimum width
                x_min_b = max(0.0, x_min_b)
                x_max_b = min(float(w - 1), x_max_b)
                if x_max_b - x_min_b < 1.0:
                    # Fallback to original per-block bbox if degenerate
                    x_min_b, x_max_b = float(blk_x_min), float(blk_x_max)

                expanded_bboxes[name] = (
                    x_min_b,
                    y_top,
                    x_max_b,
                    y_bot,
                )

    if show_plumb_line and result.plumb_line_x_px > 0:
        x = int(round(result.plumb_line_x_px))
        cv2.line(annotated, (x, 0), (x, h - 1), (180, 180, 180), 1, lineType=cv2.LINE_AA)

    centroids: List[Tuple[int, int]] = []
    for name in BLOCK_DRAW_ORDER:
        block = result.blocks.get(name)
        if block is None:
            continue
        color = BLOCK_COLORS[name]
        cx, cy = block.centroid_px
        cx_i, cy_i = int(round(cx)), int(round(cy))
        centroids.append((cx_i, cy_i))

        if show_bbox:
            # Prefer expanded stacked rectangles if available, else original bbox
            if name in expanded_bboxes:
                x_min, y_min, x_max, y_max = expanded_bboxes[name]
            else:
                x_min, y_min, x_max, y_max = block.bbox_px
            cv2.rectangle(
                annotated,
                (int(round(x_min)), int(round(y_min))),
                (int(round(x_max)), int(round(y_max))),
                color,
                2,
                lineType=cv2.LINE_AA,
            )

        # Draw a small arrow indicating horizontal shift direction near the side
        # the block is facing (anterior/posterior relative to the plumb line).
        # Use signed anterior offset in pixels when available.
        off_px = result.horizontal_offsets_px.get(name, 0.0) if hasattr(result, "horizontal_offsets_px") else 0.0
        if name in expanded_bboxes and abs(off_px) > 1e-3:
            bx_min, by_min, bx_max, by_max = expanded_bboxes[name]
            bw = max(1.0, bx_max - bx_min)
            # Determine actual image-space arrow orientation based on anterior direction
            # Positive signed offset means anterior shift.
            anterior_dir = 1 if getattr(result, "anterior_direction", 1) >= 0 else -1
            arrow_points_right = (off_px > 0 and anterior_dir > 0) or (off_px < 0 and anterior_dir < 0)

            # Choose the edge to place the arrow near
            anterior_edge_x = bx_max if anterior_dir > 0 else bx_min
            posterior_edge_x = bx_min if anterior_dir > 0 else bx_max
            edge_x = anterior_edge_x if off_px > 0 else posterior_edge_x

            # Arrow geometry
            y_mid = 0.5 * (by_min + by_max)
            arrow_len = max(12.0, 0.20 * bw)
            margin = max(4.0, 0.05 * bw)

            if arrow_points_right:
                # Draw pointing right, near right edge
                end_x = min(float(w - 1), edge_x - margin)
                start_x = max(0.0, end_x - arrow_len)
            else:
                # Draw pointing left, near left edge
                end_x = max(0.0, edge_x + margin)
                start_x = min(float(w - 1), end_x + arrow_len)

            cv2.arrowedLine(
                annotated,
                (int(round(start_x)), int(round(y_mid))),
                (int(round(end_x)), int(round(y_mid))),
                color,
                2,
                tipLength=0.35,
            )

        cv2.circle(annotated, (cx_i, cy_i), 8, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(annotated, (cx_i, cy_i), 8, (30, 30, 30), 2, lineType=cv2.LINE_AA)

        offset = result.normalized_offsets.get(name, 0.0)
        text = f"{name} ({offset:+.2f})"
        cv2.putText(
            annotated,
            text,
            (cx_i + 12, cy_i + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if show_centroid_chain and len(centroids) >= 2:
        for p1, p2 in zip(centroids, centroids[1:]):
            cv2.line(annotated, p1, p2, (240, 240, 240), 2, lineType=cv2.LINE_AA)

    if show_label:
        label_color = POSTURE_LABEL_COLORS.get(result.posture_class, (255, 255, 255))
        label = f"Klein-Vogelbach: {result.posture_class.upper()} ({result.confidence:.2f})"
        cv2.rectangle(annotated, (8, 8), (8 + 12 * len(label), 38), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            label,
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            label_color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def analyze_image(
    image: np.ndarray,
    landmarks: Sequence[Landmark],
    include_arms_in_thorax: bool = False,
    thresholds: Optional[KleinVogelbachThresholds] = None,
) -> Tuple[Dict[str, BodyBlock], KleinVogelbachResult]:
    """Convenience wrapper: aggregate blocks then classify, given pose landmarks."""
    h, w = image.shape[:2]
    thresholds = thresholds or KleinVogelbachThresholds()
    blocks = aggregate_blocks(
        landmarks,
        image_w=w,
        image_h=h,
        include_arms_in_thorax=include_arms_in_thorax,
        min_visibility=thresholds.min_visibility,
    )
    result = classify_klein_vogelbach(
        blocks,
        landmarks,
        image_w=w,
        image_h=h,
        thresholds=thresholds,
    )
    return blocks, result
