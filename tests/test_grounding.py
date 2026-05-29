import pytest
from src.grounding.screenshot import physical_to_logical, logical_to_physical
from src.grounding.grounder import map_relative_to_absolute

def test_dpi_coordinate_conversion(monkeypatch):
    # Test with 110% scaling (1.10)
    # Physical screen of 2112x1188 maps to logical 1920x1080
    
    # 1. Physical -> Logical
    lx, ly = physical_to_logical(2112.0, 1188.0)
    assert lx == 1920
    assert ly == 1080
    
    lx, ly = physical_to_logical(110.0, 220.0)
    assert lx == 100
    assert ly == 200

    # 2. Logical -> Physical
    px, py = logical_to_physical(1920, 1080)
    assert px == 2112
    assert py == 1188

def test_map_relative_to_absolute():
    # Crop is at physical pixels x: 100-300 (width=200), y: 200-400 (height=200)
    crop_box = (100, 200, 300, 400)
    
    # Element is at center of crop relative coordinates (0.5, 0.5), size (0.1, 0.1)
    rel_x = 0.5
    rel_y = 0.5
    rel_w = 0.1
    rel_h = 0.1
    
    bbox, center = map_relative_to_absolute(rel_x, rel_y, rel_w, rel_h, crop_box)
    
    # Center physical should be:
    # cx = 100 + 0.5 * 200 = 200
    # cy = 200 + 0.5 * 200 = 300
    assert center == (200.0, 300.0)
    
    # Size physical should be:
    # w = 0.1 * 200 = 20
    # h = 0.1 * 200 = 20
    # bbox boundaries:
    # x1 = 200 - 10 = 190
    # y1 = 300 - 10 = 290
    # x2 = 200 + 10 = 210
    # y2 = 300 + 10 = 310
    assert bbox == (190.0, 290.0, 210.0, 310.0)
