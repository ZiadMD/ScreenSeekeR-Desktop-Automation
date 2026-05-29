import os
import subprocess
import time
from pathlib import Path
from typing import Optional
import pygetwindow as gw
from src.config import settings
from src.automation.desktop import click_at, send_shortcut, type_text, press_key
from src.grounding.screenseeker import ScreenSeeker
from src.utils.logging import logger
from src.utils.retry import robust_retry

class NotepadWorkflow:
    """
    Automates Windows 11 Notepad workflows including launching, typing, saving, and closing.
    """
    def __init__(self, screenseeker: Optional[ScreenSeeker] = None):
        self.screenseeker = screenseeker or ScreenSeeker()

    def launch_via_grounding(self, max_attempts: int = 3) -> bool:
        """
        Locates the Notepad shortcut icon on the desktop using vision grounding
        and double-clicks it to launch the app.
        """
        logger.info("Attempting to launch Notepad via vision grounding...")
        
        # Bring focus to desktop by sending Win+D first, ensuring desktop icons are visible
        logger.info("Sending Win+D shortcut to show desktop.")
        send_shortcut("win", "d")
        time.sleep(0.5)

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Launch attempt {attempt}/{max_attempts}")
            try:
                # Ask ScreenSeeker to locate Notepad icon
                coords, confidence = self.screenseeker.locate_element(
                    instruction="the Notepad icon shortcut on the desktop",
                    filename_prefix=f"launch_attempt_{attempt}"
                )
                
                if coords:
                    lx, ly = coords
                    logger.info(f"Notepad icon found at logical ({lx}, {ly}) with {confidence:.1%} confidence.")
                    
                    # Double-click to launch
                    click_at(lx, ly, double=True)
                    
                    # Verify window opened
                    if self.wait_for_notepad_window():
                        logger.info("Notepad window successfully opened and verified.")
                        return True
                else:
                    logger.warning(f"Notepad icon not found on attempt {attempt}.")
            except Exception as e:
                logger.error(f"Error on launch attempt {attempt}: {e}")
                
            time.sleep(1.0)
            
        # subprocess fallback just in case the icon is deleted/obscured (adds robustness)
        logger.warning("Vision launch failed. Falling back to launching notepad.exe via subprocess.")
        subprocess.Popen(["notepad.exe"])
        return self.wait_for_notepad_window()

    def wait_for_notepad_window(self, timeout: float = 5.0) -> bool:
        """
        Polls for Notepad window presence.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            windows = gw.getWindowsWithTitle("Notepad")
            if windows:
                # Bring the Notepad window to the foreground
                try:
                    notepad_win = windows[0]
                    if notepad_win.isMinimized:
                        notepad_win.restore()
                    notepad_win.activate()
                    logger.info(f"Notepad window activated: '{notepad_win.title}'")
                    time.sleep(0.5)
                    return True
                except Exception as e:
                    logger.warning(f"Failed to activate Notepad window: {e}")
            time.sleep(0.5)
        return False

    def write_and_save_post(self, post_id: int, post_content: str) -> bool:
        """
        Types the formatted post, triggers Save As, enters absolute path, closes the file,
        and verifies it exists on disk.
        """
        output_path = settings.desktop_output_dir / f"post_{post_id}.txt"
        logger.info(f"Beginning save workflow for post {post_id} -> {output_path}")

        # Ensure active Notepad window
        if not self.wait_for_notepad_window():
            logger.error("No active Notepad window found to type the post content.")
            return False

        # In Win11 Notepad, let's open a new tab/document to start clean
        logger.info("Opening a clean document in Notepad (Ctrl+N)")
        send_shortcut("ctrl", "n")
        time.sleep(0.5)

        # Type text (using pyautogui)
        type_text(post_content)
        time.sleep(0.5)

        # Trigger Save shortcut (Ctrl+S)
        logger.info("Triggering Save (Ctrl+S)")
        send_shortcut("ctrl", "s")
        time.sleep(1.0)  # Wait for save dialog

        # Type full absolute path in file dialog
        logger.info(f"Typing target filepath: {output_path}")
        type_text(str(output_path))
        time.sleep(0.5)

        # Press Enter to save
        logger.info("Pressing Enter to confirm save dialog.")
        press_key("enter")
        time.sleep(1.5)  # Wait for write

        # Verify file is written on disk
        if not self._verify_file_on_disk(output_path):
            logger.error(f"File verification failed! File {output_path} not found on disk.")
            return False

        # Close active tab or window
        logger.info("Closing Notepad tab/window (Ctrl+W)")
        send_shortcut("ctrl", "w")
        time.sleep(0.5)

        # Verify window is clean
        logger.info(f"Post {post_id} save workflow complete.")
        return True

    def close_all_notepad_windows(self):
        """
        Closes any active Notepad instances to ensure clean state.
        """
        logger.info("Cleaning up active Notepad processes...")
        windows = gw.getWindowsWithTitle("Notepad")
        for win in windows:
            try:
                win.close()
                time.sleep(0.3)
                # If there's an unsaved changes prompt, click Don't Save (alt+n or tab-enter)
                # Modern Notepad asks inside dialog. We will try pressing tab and enter to reject saving
                press_key("tab")
                press_key("enter")
            except Exception as e:
                logger.debug(f"Error closing Notepad instance: {e}")

    def _verify_file_on_disk(self, path: Path, timeout: float = 3.0) -> bool:
        """
        Polls for file creation on disk with a timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            if path.exists() and path.stat().st_size > 0:
                logger.info(f"File verified on disk! Size: {path.stat().st_size} bytes")
                return True
            time.sleep(0.3)
        return False
