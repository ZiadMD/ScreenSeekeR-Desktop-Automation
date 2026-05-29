import pytest
from src.grounding.scoring import (
    calculate_iou,
    gaussian_centrality,
    score_and_rank_candidates,
    apply_nms
)

def test_calculate_iou_no_overlap():
    box1 = (0.0, 0.0, 0.2, 0.2)
    box2 = (0.5, 0.5, 0.7, 0.7)
    iou = calculate_iou(box1, box2)
    assert iou == 0.0

def test_calculate_iou_perfect_overlap():
    box1 = (0.1, 0.1, 0.3, 0.3)
    box2 = (0.1, 0.1, 0.3, 0.3)
    iou = calculate_iou(box1, box2)
    assert iou == pytest.approx(1.0)

def test_calculate_iou_partial_overlap():
    box1 = (0.0, 0.0, 2.0, 2.0) # area = 4.0
    box2 = (1.0, 0.0, 3.0, 2.0) # area = 4.0
    # Intersection is (1.0, 0.0, 2.0, 2.0), area = 2.0
    # Union is 4.0 + 4.0 - 2.0 = 6.0
    # IoU = 2.0 / 6.0 = 0.3333
    iou = calculate_iou(box1, box2)
    assert iou == pytest.approx(1.0 / 3.0)

def test_gaussian_centrality():
    center = (0.5, 0.5)
    ref = (0.5, 0.5)
    # Distance = 0, centrality should be exp(0) = 1.0
    assert gaussian_centrality(center, ref, sigma=0.3) == 1.0
    
    # Distance increases, centrality should drop
    far_center = (0.8, 0.8)
    assert gaussian_centrality(far_center, ref, sigma=0.3) < 1.0

def test_score_and_rank_candidates():
    candidates = [
        {"x_min": 0.4, "y_min": 0.4, "x_max": 0.6, "y_max": 0.6, "confidence": 0.8}, # center (0.5, 0.5)
        {"x_min": 0.8, "y_min": 0.8, "x_max": 0.9, "y_max": 0.9, "confidence": 0.9}  # center (0.85, 0.85)
    ]
    
    ranked = score_and_rank_candidates(candidates, expected_center=(0.5, 0.5), sigma=0.3)
    
    # Even though candidate 2 has higher raw confidence (0.9 vs 0.8), candidate 1 is perfectly centered
    # and should have a higher final combined centrality score, ranking first.
    assert len(ranked) == 2
    assert ranked[0]["confidence"] == 0.8
    assert ranked[1]["confidence"] == 0.9
    assert ranked[0]["score"] > ranked[1]["score"]

def test_apply_nms():
    # Box 1 and 2 overlap heavily (IoU > 0.3), Box 3 is independent
    candidates = [
        {"x_min": 0.1, "y_min": 0.1, "x_max": 0.3, "y_max": 0.3, "score": 0.9, "description": "Box 1"},
        {"x_min": 0.12, "y_min": 0.12, "x_max": 0.32, "y_max": 0.32, "score": 0.8, "description": "Box 2"},
        {"x_min": 0.7, "y_min": 0.7, "x_max": 0.9, "y_max": 0.9, "score": 0.7, "description": "Box 3"}
    ]
    
    nms_res = apply_nms(candidates.copy(), iou_threshold=0.3)
    
    # Box 2 should be suppressed, Box 1 and Box 3 kept
    assert len(nms_res) == 2
    assert nms_res[0]["description"] == "Box 1"
    assert nms_res[1]["description"] == "Box 3"
