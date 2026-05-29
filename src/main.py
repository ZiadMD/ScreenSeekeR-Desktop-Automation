import sys
import time
from src.config import settings
from src.utils.logging import logger
from src.api.posts import PostClient
from src.grounding.screenseeker import ScreenSeeker
from src.automation.notepad import NotepadWorkflow
from src.automation.popups import PopupHandler

def run_orchestrator():
    """
    Main orchestrator that coordinates the full vision-based desktop automation pipeline.
    """
    logger.info("=================================================================")
    logger.info("   STARTING VISION-BASED DESKTOP AUTOMATION PIPELINE (ScreenSeekeR) ")
    logger.info("=================================================================")
    logger.info(f"Target DPI Scaling: {settings.DPI_SCALING:.2f} (110% scaling correction active)")
    logger.info(f"Primary API Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Planning Model: {settings.PLANNER_MODEL}")
    logger.info(f"Grounding Model: {settings.GROUNDER_MODEL}")
    logger.info(f"Output Directory: {settings.desktop_output_dir}")
    logger.info("=================================================================")

    # 1. Fetch posts from API (with built-in retry and fallback)
    try:
        post_client = PostClient()
        posts = post_client.fetch_first_10_posts()
        logger.info(f"Retrieved {len(posts)} posts for automated writing.")
    except Exception as e:
        logger.critical(f"Critical error fetching posts: {e}. Aborting pipeline.")
        sys.exit(1)

    # 2. Initialize vision automation components
    try:
        screenseeker = ScreenSeeker()
        notepad_workflow = NotepadWorkflow(screenseeker)
        popup_handler = PopupHandler(screenseeker.planner_client)
    except Exception as e:
        logger.critical(f"Critical error initializing automation components: {e}. Check API keys.")
        sys.exit(1)

    # Ensure output directories exist
    settings.desktop_output_dir.mkdir(parents=True, exist_ok=True)
    settings.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Clean up active Notepad instances first
    logger.info("Preparing environment: closing any active Notepad windows.")
    notepad_workflow.close_all_notepad_windows()
    time.sleep(1.0)

    success_count = 0
    failure_count = 0

    # 4. Process each post sequentially
    for index, post in enumerate(posts):
        post_id = post.id
        formatted_text = post.to_formatted_text()
        logger.info(f"\n--------------------------------------------------------------")
        logger.info(f" PROCESSING POST {index + 1}/10 - ID: {post_id}")
        logger.info(f" Title: '{post.title[:50]}...'")
        logger.info(f"--------------------------------------------------------------")

        try:
            # Step 4a. Check and dismiss unexpected popups to ensure clear workspace
            logger.info("Checking for blocking dialogs or unexpected popups...")
            dismissed = popup_handler.check_and_dismiss_popups()
            if dismissed:
                logger.info("Watchdog cleared a blocking dialog. Pausing to let workspace settle...")
                time.sleep(1.0)
            
            # Step 4b. Launch Notepad using vision grounding
            logger.info("Locating and launching Notepad...")
            launched = notepad_workflow.launch_via_grounding()
            if not launched:
                logger.error(f"Failed to launch Notepad for post {post_id}. Skipping post.")
                failure_count += 1
                continue
                
            time.sleep(1.0)  # Wait for window active state

            # Step 4c. Type the post content and save it
            saved = notepad_workflow.write_and_save_post(post_id, formatted_text)
            
            if saved:
                logger.info(f"SUCCESS: Post {post_id} successfully saved as text file.")
                success_count += 1
            else:
                logger.error(f"FAILURE: Save sequence failed for post {post_id}.")
                failure_count += 1
                
        except Exception as e:
            logger.error(f"Unexpected error processing post {post_id}: {e}")
            failure_count += 1
            
        finally:
            # Ensure Notepad is closed cleanly for next iteration
            logger.info("Closing active Notepad windows...")
            notepad_workflow.close_all_notepad_windows()
            time.sleep(1.0)  # Give OS time to close process

    # 5. Summarize execution results
    logger.info("=================================================================")
    logger.info("                 AUTOMATION WORKFLOW COMPLETE                    ")
    logger.info("=================================================================")
    logger.info(f"Total Posts Processed: {len(posts)}")
    logger.info(f"Successful Saves:      {success_count}")
    logger.info(f"Failed Saves:          {failure_count}")
    logger.info(f"Output Location:       {settings.desktop_output_dir}")
    logger.info("=================================================================")

if __name__ == "__main__":
    run_orchestrator()
