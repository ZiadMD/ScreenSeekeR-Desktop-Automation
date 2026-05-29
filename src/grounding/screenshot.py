from pathlib import Path
from typing import Tuple, Optional, List
import mss
from PIL import Image, ImageDraw, ImageFont
from src.config import settings
from src.utils.logging import logger

def capture_screen() -> Image.Image:
    """
    Captures the primary monitor screen using mss and returns a PIL Image.
    This captures the screen in physical pixels (e.g., 2112x1188 for 1920x1080 at 100% scaling).
    """
    logger.debug("Capturing full screen screenshot...")
    with mss.mss() as sct:
        # Get primary monitor details
        monitor = sct.monitors[1]  # 0 is all monitors, 1 is primary monitor
        sct_img = sct.grab(monitor)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        logger.debug(f"Captured screenshot size: {img.width}x{img.height}")
        return img

def physical_to_logical(x: float, y: float) -> Tuple[int, int]:
    """
    Converts physical screen coordinates (from screenshot pixels)
    to logical coordinates (for PyAutoGUI automation) using the DPI scaling factor.
    """
    logical_x = round(x / settings.DPI_SCALING)
    logical_y = round(y / settings.DPI_SCALING)
    logger.debug(f"Coordinate conversion (physical -> logical): ({x:.1f}, {y:.1f}) -> ({logical_x}, {logical_y})")
    return logical_x, logical_y

def logical_to_physical(x: float, y: float) -> Tuple[int, int]:
    """
    Converts logical coordinates (from PyAutoGUI space)
    to physical coordinates (for screenshot cropping/processing).
    """
    phys_x = round(x * settings.DPI_SCALING)
    phys_y = round(y * settings.DPI_SCALING)
    logger.debug(f"Coordinate conversion (logical -> physical): ({x:.1f}, {y:.1f}) -> ({phys_x}, {phys_y})")
    return phys_x, phys_y

def annotate_screenshot(
    image: Image.Image,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    point: Optional[Tuple[float, float]] = None,
    label: str = "Target",
    confidence: float = 1.0,
    search_trace: Optional[List[Tuple[float, float, float, float]]] = None
) -> Image.Image:
    """
    Annotates a screenshot with a green bounding box around the target,
    a red crosshair on the click point, and a confidence label.
    Optionally shows candidate crop boxes in yellow.
    Coordinates should be provided in physical pixels.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Try to load a nicer font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Draw optional search traces (candidate crops) in yellow
    if search_trace:
        for trace_box in search_trace:
            x1, y1, x2, y2 = trace_box
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
            draw.text((x1 + 5, y1 + 5), "Search Crop", fill="yellow", font=font)

    # Draw bounding box in green
    if bbox:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
        
        # Add label with background block
        text = f"{label} ({confidence:.2%})"
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # Make sure label is within bounds
        label_y = y1 - text_h - 8 if y1 - text_h - 8 > 0 else y2 + 5
        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 8], fill="green")
        draw.text((x1 + 5, label_y + 4), text, fill="white", font=font)

    # Draw target click point in red (crosshair)
    if point:
        px, py = point
        cross_size = 15
        draw.line([px - cross_size, py, px + cross_size, py], fill="red", width=3)
        draw.line([px, py - cross_size, px, py + cross_size], fill="red", width=3)
        draw.ellipse([px - 4, py - 4, px + 4, py + 4], outline="red", fill="white", width=2)

    return annotated

def save_screenshot(image: Image.Image, filename: str) -> Path:
    """
    Saves screenshot to the configured screenshots directory.
    """
    if not filename.endswith(".png"):
        filename += ".png"
    filepath = settings.SCREENSHOTS_DIR / filename
    image.save(filepath, "PNG")
    logger.info(f"Screenshot saved to: {filepath}")
    return filepath
