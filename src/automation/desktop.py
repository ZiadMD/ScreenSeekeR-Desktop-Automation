import time
import pyautogui
from src.utils.logging import logger

# Set pyautogui safety fail-safe (move mouse to any corner to abort execution)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5  # Add a 500ms delay between PyAutoGUI actions for UI stability

def click_at(x: int, y: int, double: bool = False):
    # Clamp to safe screen bounds — avoid fail-safe corners
    x = max(10, int(x))
    y = max(10, int(y))

    logger.info(f"Moving mouse to ({x}, {y}) and performing {'double-click' if double else 'single-click'}...")
    pyautogui.moveTo(x, y, duration=0.6, tween=pyautogui.easeInOutQuad)
    
    time.sleep(0.2)
    
    if double:
        pyautogui.doubleClick()
    else:
        pyautogui.click()
    
    time.sleep(0.5)  # Let UI respond

# TODO: edit this to include shortcuts that has more than 2 keys
def send_shortcut(key1: str, key2: str):
    """
    Triggers a standard keyboard shortcut combo (e.g. 'ctrl', 's').
    """
    logger.info(f"Triggering shortcut: {key1} + {key2}")
    pyautogui.hotkey(key1, key2)
    time.sleep(0.5)

def type_text(text: str, interval: float = 0.01):
    """
    Types text into the active cursor position.
    """
    logger.info(f"Typing text (length: {len(text)})...")
    pyautogui.write(text, interval=interval)
    time.sleep(0.3)

def press_key(key: str, presses: int = 1):
    """
    Presses a single keyboard key (e.g. 'enter', 'tab', 'esc').
    """
    logger.info(f"Pressing key: '{key}' {presses} time(s)")
    pyautogui.press(key, presses=presses)
    time.sleep(0.3)
