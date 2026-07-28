"""Unit tests for tile geometry and cross-tile dedup -- pure functions, no
model/CLI dependency, fast."""

from tiling import BoxDetection, Tile, generate_tiles, iou, merge_detections, translate_and_clamp


def test_generate_tiles_small_image_returns_single_tile():
    tiles = generate_tiles(width=400, height=300, tile_size=512, step=384)
    assert tiles == [Tile(0, 0, 400, 300)]


def test_tile_width_and_height():
    tile = Tile(x0=300, y0=384, x1=812, y1=896)
    assert tile.width == 512
    assert tile.height == 512


def test_generate_tiles_covers_whole_image_with_overlap():
    tiles = generate_tiles(width=812, height=1092, tile_size=512, step=384)

    assert all(0 <= t.x0 < t.x1 <= 812 for t in tiles)
    assert all(0 <= t.y0 < t.y1 <= 1092 for t in tiles)
    # every tile is flush with the right/bottom edge unless it's the last
    # column/row, and the last column/row is always flush with the edge
    xs = sorted({t.x1 for t in tiles})
    ys = sorted({t.y1 for t in tiles})
    assert xs[-1] == 812
    assert ys[-1] == 1092


def test_generate_tiles_last_tile_flush_with_edge_not_stepped_past():
    tiles = generate_tiles(width=900, height=512, tile_size=512, step=384)
    xs = sorted({(t.x0, t.x1) for t in tiles})
    assert xs[-1] == (900 - 512, 900)


def test_generate_tiles_exact_multiple_of_step_has_no_dangling_edge_tile():
    # 512 + 384 = 896: second tile starts exactly at the edge-flush position
    tiles = generate_tiles(width=896, height=512, tile_size=512, step=384)
    x_starts = sorted({t.x0 for t in tiles})
    assert x_starts == [0, 384]


def test_iou_identical_boxes_is_one():
    box = [10.0, 10.0, 50.0, 50.0]
    assert iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert iou([0.0, 0.0, 10.0, 10.0], [100.0, 100.0, 110.0, 110.0]) == 0.0


def test_iou_partial_overlap():
    # two 10x10 boxes overlapping in a 5x10 region -> intersection 50,
    # union 200 - 50 = 150
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [5.0, 0.0, 15.0, 10.0]
    assert abs(iou(box_a, box_b) - 50 / 150) < 1e-9


def test_merge_detections_merges_overlapping_same_label():
    detections = [
        BoxDetection(label="screw", box=[100.0, 100.0, 150.0, 150.0]),
        BoxDetection(label="screw", box=[110.0, 100.0, 160.0, 150.0]),  # IoU ~0.71 with above
    ]
    merged = merge_detections(detections, iou_threshold=0.5)
    assert len(merged) == 1
    assert merged[0].label == "screw"
    assert merged[0].box == [100.0, 100.0, 160.0, 150.0]


def test_merge_detections_keeps_different_labels_separate():
    detections = [
        BoxDetection(label="screw", box=[100.0, 100.0, 150.0, 150.0]),
        BoxDetection(label="hex nut", box=[100.0, 100.0, 150.0, 150.0]),
    ]
    merged = merge_detections(detections, iou_threshold=0.5)
    assert len(merged) == 2


def test_merge_detections_keeps_low_overlap_boxes_separate():
    detections = [
        BoxDetection(label="screw", box=[0.0, 0.0, 10.0, 10.0]),
        BoxDetection(label="screw", box=[9.0, 0.0, 19.0, 10.0]),  # IoU ~0.05
    ]
    merged = merge_detections(detections, iou_threshold=0.5)
    assert len(merged) == 2


def test_translate_and_clamp_offsets_by_tile_origin():
    tile = Tile(x0=300, y0=384, x1=812, y1=896)
    raw = [{"label": "screw", "box": [10.0, 20.0, 30.0, 40.0]}]
    result = translate_and_clamp(raw, tile)
    assert result == [BoxDetection(label="screw", box=[310.0, 404.0, 330.0, 424.0])]


def test_translate_and_clamp_clamps_overshoot_to_tile_bounds():
    tile = Tile(x0=0, y0=0, x1=512, y1=512)
    # y2 overshoots the 512px-tall tile -- a real overshoot observed from the
    # model's box regression near a tile edge, not floating-point noise.
    raw = [{"label": "person", "box": [63.3, 418.2, 187.3, 530.9]}]
    result = translate_and_clamp(raw, tile)
    assert result == [BoxDetection(label="person", box=[63.3, 418.2, 187.3, 512.0])]
