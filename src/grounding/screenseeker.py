import time
from typing import Tuple, Optional, Dict, Any, List
from PIL import Image
from src.config import settings
from src.grounding.screenshot import (
    capture_screen,
    physical_to_logical,
    annotate_screenshot,
    save_screenshot
)
from src.grounding.llm_client import LLMClient
from src.grounding.planner import Planner
from src.grounding.grounder import Grounder, map_relative_to_absolute
from src.grounding.scoring import score_and_rank_candidates, apply_nms
from src.utils.logging import logger

class ScreenSeeker:
    """
    Core ScreenSeekeR visual search engine orchestrator.
    Combines Planner, Grounder, Scoring, NMS, and recursive crops.
    """
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient()
        self.planner = Planner(self.client)
        self.grounder = Grounder(self.client)

    def locate_element(
        self,
        instruction: str,
        save_trace: bool = True,
        filename_prefix: str = "detection"
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        Locates a desktop UI element by natural language description.
        Returns:
            - Tuple[int, int]: Logical screen coordinates (x, y) for clicking (accounts for DPI scaling).
            - float: Confidence score (0.0 to 1.0)
        """
        logger.info(f"Locating UI element with description: '{instruction}'")
        
        # 1. Capture physical screen image
        full_screenshot = capture_screen()
        phys_w, phys_h = full_screenshot.size
        
        # 2. Ask Planner to propose candidate regions
        planner_response = self.planner.propose_candidate_regions(full_screenshot, instruction)
        candidates = planner_response.get("candidates", [])
        
        if not candidates:
            logger.error("No candidate regions proposed by Planner. Element location failed.")
            return None, 0.0

        # 3. Score and rank candidates by Gaussian centrality and planner confidence
        scored_candidates = score_and_rank_candidates(
            candidates=candidates,
            expected_center=(0.5, 0.5), # Standard screen centrality anchor
            sigma=0.3
        )
        
        # 4. Apply Non-Maximum Suppression to avoid duplicate/redundant crops
        filtered_candidates = apply_nms(scored_candidates, iou_threshold=settings.IoU_THRESHOLD)
        
        logger.info(f"NMS filtered candidates list from {len(scored_candidates)} down to {len(filtered_candidates)} active search regions.")
        
        search_traces: List[Tuple[float, float, float, float]] = []
        best_grounding: Optional[Dict[str, Any]] = None
        best_abs_center: Optional[Tuple[float, float]] = None
        best_abs_bbox: Optional[Tuple[float, float, float, float]] = None
        best_crop_box_phys: Optional[Tuple[int, int, int, int]] = None

        # 5. Search sequentially through the ranked candidate regions
        for i, cand in enumerate(filtered_candidates):
            logger.info(f"Processing candidate region {i+1}/{len(filtered_candidates)}: {cand['description']} (Score: {cand['score']:.3f})")
            
            # Map normalized candidate box to physical screenshot pixel coords
            x_min_phys = int(cand["x_min"] * phys_w)
            y_min_phys = int(cand["y_min"] * phys_h)
            x_max_phys = int(cand["x_max"] * phys_w)
            y_max_phys = int(cand["y_max"] * phys_h)
            
            crop_box_phys = (x_min_phys, y_min_phys, x_max_phys, y_max_phys)
            search_traces.append((float(x_min_phys), float(y_min_phys), float(x_max_phys), float(y_max_phys)))
            
            # Crop search region from full physical screenshot
            crop_img = full_screenshot.crop(crop_box_phys)
            
            # Call Grounder within the crop
            try:
                grounding_res = self.grounder.ground_element(crop_img, instruction)
                confidence = grounding_res.get("confidence", 0.0)
                
                # Check if this grounding meets the confidence threshold
                if confidence >= settings.CONFIDENCE_THRESHOLD:
                    # Map relative crop coordinates back to full physical screenshot coords
                    abs_bbox, abs_center = map_relative_to_absolute(
                        rel_x=grounding_res["x"],
                        rel_y=grounding_res["y"],
                        rel_w=grounding_res["width"],
                        rel_h=grounding_res["height"],
                        crop_box=crop_box_phys
                    )
                    
                    # Update best match if it's the highest confidence so far
                    if best_grounding is None or confidence > best_grounding["confidence"]:
                        best_grounding = grounding_res
                        best_abs_center = abs_center
                        best_abs_bbox = abs_bbox
                        best_crop_box_phys = crop_box_phys
                        
                        # Short-circuit if extremely high confidence
                        if confidence >= 0.85:
                            logger.info(f"High confidence match ({confidence:.1%}) found. Short-circuiting candidate search.")
                            break
                else:
                    logger.debug(f"Grounding candidate rejected: confidence ({confidence:.1%}) below threshold.")
            except Exception as e:
                logger.error(f"Error grounding candidate region {i+1}: {e}")
                continue

        # 6. Fallback if no matching region meets threshold: use the top candidate's center
        if best_grounding is None:
            logger.warning("No candidate region passed the grounding confidence threshold. Attempting absolute fallback to first candidate center.")
            if filtered_candidates:
                top_cand = filtered_candidates[0]
                cx_phys = int(((top_cand["x_min"] + top_cand["x_max"]) / 2.0) * phys_w)
                cy_phys = int(((top_cand["y_min"] + top_cand["y_max"]) / 2.0) * phys_h)
                
                logical_x, logical_y = physical_to_logical(cx_phys, cy_phys)
                logger.info(f"Fallback coordinates generated: logical=({logical_x}, {logical_y})")
                return (logical_x, logical_y), 0.10
            else:
                logger.error("Completely failed to locate the UI element.")
                return None, 0.0

        # 7. Optional Confirmation Step (Refinement)
        # Crop a tight 200x200 region around the best predicted physical point and re-ground to fine-tune
        if settings.CONFIRMATION_STEP and best_abs_center:
            logger.info("Executing optional confirmation/refinement step for maximum coordinate precision...")
            cx_phys, cy_phys = best_abs_center
            
            # Crop 200x200 bounding box
            ref_size = 200
            rx1 = max(0, int(cx_phys - ref_size // 2))
            ry1 = max(0, int(cy_phys - ref_size // 2))
            rx2 = min(phys_w, rx1 + ref_size)
            ry2 = min(phys_h, ry1 + ref_size)
            
            # In case near boundaries, correct shift
            if rx2 == phys_w:
                rx1 = max(0, rx2 - ref_size)
            if ry2 == phys_h:
                ry1 = max(0, ry2 - ref_size)
                
            ref_crop_box = (rx1, ry1, rx2, ry2)
            ref_crop_img = full_screenshot.crop(ref_crop_box)
            
            try:
                ref_grounding = self.grounder.ground_element(ref_crop_img, instruction)
                ref_conf = ref_grounding.get("confidence", 0.0)
                
                # If refinement is successful and confidence is high, accept refinement coordinates
                if ref_conf >= 0.30:
                    ref_abs_bbox, ref_abs_center = map_relative_to_absolute(
                        rel_x=ref_grounding["x"],
                        rel_y=ref_grounding["y"],
                        rel_w=ref_grounding["width"],
                        rel_h=ref_grounding["height"],
                        crop_box=ref_crop_box
                    )
                    logger.info(f"Refinement complete. Coordinates updated from {best_abs_center} to {ref_abs_center} (confidence: {ref_conf:.1%})")
                    best_abs_center = ref_abs_center
                    best_abs_bbox = ref_abs_bbox
            except Exception as e:
                logger.warning(f"Coordinate refinement failed: {e}. Keeping original estimates.")

        # 8. Convert absolute physical coordinates to PyAutoGUI logical coordinates
        abs_x, abs_y = best_abs_center
        logical_x, logical_y = physical_to_logical(abs_x, abs_y)
        
        # 9. Annotate and save the trace screenshot
        if save_trace:
            annotated_img = annotate_screenshot(
                image=full_screenshot,
                bbox=best_abs_bbox,
                point=(abs_x, abs_y),
                label=instruction,
                confidence=best_grounding.get("confidence", 0.0),
                search_trace=search_traces
            )
            save_screenshot(annotated_img, f"{filename_prefix}_{int(time.time())}.png")
            
        logger.info(f"UI Element located at logical coordinates ({logical_x}, {logical_y}) with {best_grounding.get('confidence', 0.0):.1%} confidence.")
        return (logical_x, logical_y), best_grounding.get("confidence", 0.0)
