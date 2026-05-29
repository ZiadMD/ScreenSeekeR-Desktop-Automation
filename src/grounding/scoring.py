import math
from typing import Dict, Any, List, Tuple

def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Computes Intersection over Union (IoU) of two bounding boxes.
    Boxes format: (x_min, y_min, x_max, y_max)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Coordinates of intersection box
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    # Area of intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Area of boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area

def gaussian_centrality(
    box_center: Tuple[float, float],
    reference_point: Tuple[float, float],
    sigma: float = 0.3
) -> float:
    """
    Calculates Gaussian Centrality multiplier (Eq. 1 in ScreenSeekeR paper).
    sigma: default 0.3
    """
    bx, by = box_center
    rx, ry = reference_point

    # Euclidean distance
    dist_sq = (bx - rx) ** 2 + (by - ry) ** 2
    
    # Gaussian penalty
    penalty = math.exp(-dist_sq / (2 * (sigma ** 2)))
    return penalty

def score_and_rank_candidates(
    candidates: List[Dict[str, Any]],
    expected_center: Tuple[float, float] = (0.5, 0.5),
    sigma: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Scores candidates by combining Planner's confidence with Gaussian Centrality penalty
    relative to an expected target center (default screen center 0.5, 0.5).
    """
    scored_candidates = []
    for c in candidates:
        # Calculate candidate center
        x_min, y_min, x_max, y_max = c["x_min"], c["y_min"], c["x_max"], c["y_max"]
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        
        # Calculate centrality penalty
        centrality = gaussian_centrality((cx, cy), expected_center, sigma)
        
        # Final Score = Confidence * Centrality
        planner_confidence = c.get("confidence", 1.0)
        final_score = planner_confidence * centrality
        
        # Store back
        scored = c.copy()
        scored["score"] = final_score
        scored["centrality"] = centrality
        scored_candidates.append(scored)
        
    # Sort candidates by score descending
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)
    return scored_candidates

def apply_nms(
    candidates: List[Dict[str, Any]],
    iou_threshold: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Applies Non-Maximum Suppression (NMS) to eliminate highly overlapping candidate regions.
    Assumes candidates list is pre-sorted by score/confidence descending.
    """
    keep = []
    while candidates:
        best = candidates.pop(0)
        keep.append(best)
        
        box_best = (best["x_min"], best["y_min"], best["x_max"], best["y_max"])
        
        # Filter out remaining boxes that have IoU >= threshold with the best box
        remaining = []
        for c in candidates:
            box_c = (c["x_min"], c["y_min"], c["x_max"], c["y_max"])
            if calculate_iou(box_best, box_c) < iou_threshold:
                remaining.append(c)
        candidates = remaining
        
    return keep
