"""Tile-splitting and cross-tile deduplication for dense-scene /detect handling.

Pure geometry/data functions only -- no I/O, no inference calls. See
docs/adr/0001-iou-merge-not-nms-for-tile-dedup.md for why IoU merge is used
instead of NMS (the engine exposes no per-box confidence to rank by).
"""

from __future__ import annotations

from dataclasses import dataclass

TILE_SIZE = 512
TILE_OVERLAP = 128  # 25% of TILE_SIZE
TILE_STEP = TILE_SIZE - TILE_OVERLAP
DEDUP_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class Tile:
    """A tile's bounds in whole-image pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class BoxDetection:
    """A single detection, already translated into whole-image pixel coordinates."""

    label: str
    box: list[float]  # [x1, y1, x2, y2]


def generate_tiles(
    width: int, height: int, tile_size: int = TILE_SIZE, step: int = TILE_STEP
) -> list[Tile]:
    """Cover a width x height image with overlapping tile_size tiles, step apart.

    If the image fits within a single tile on both axes, returns one tile
    covering the whole image (no splitting needed, no dedup required).
    Otherwise, the last tile on each axis is pulled back flush with the image
    edge rather than stepped past it, so every tile stays in bounds.
    """
    xs = _axis_starts(width, tile_size, step)
    ys = _axis_starts(height, tile_size, step)
    return [
        Tile(x, y, min(x + tile_size, width), min(y + tile_size, height)) for y in ys for x in xs
    ]


def _axis_starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def merge_detections(
    detections: list[BoxDetection], iou_threshold: float = DEDUP_IOU_THRESHOLD
) -> list[BoxDetection]:
    """Merge same-label detections whose IoU >= threshold into one envelope box.

    Detections must already be in whole-image pixel coordinates (tile-local
    box + tile origin offset) before calling this. Same-tile duplicates aren't
    expected -- locate-anything.cpp decodes a single tile deterministically --
    so this only needs to catch objects that land in more than one tile's
    overlap region. Merging two matched boxes takes their bounding-box union
    (there's no confidence score to prefer one box over the other -- see ADR
    0001 -- and the union tends to recover more of an object that was
    partially clipped in one of the two tiles).
    """
    merged: list[BoxDetection] = []
    for det in detections:
        match = next(
            (m for m in merged if m.label == det.label and iou(m.box, det.box) >= iou_threshold),
            None,
        )
        if match is None:
            merged.append(det)
        else:
            merged.remove(match)
            merged.append(_union(match, det))
    return merged


def _union(a: BoxDetection, b: BoxDetection) -> BoxDetection:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    return BoxDetection(
        label=a.label,
        box=[min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)],
    )
