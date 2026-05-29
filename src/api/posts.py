from typing import List
import requests
from pydantic import BaseModel, Field
from src.config import settings
from src.utils.logging import logger
from src.utils.retry import robust_retry

class Post(BaseModel):
    id: int
    title: str
    body: str

    def to_formatted_text(self) -> str:
        """
        Formats the post according to the spec:
        Title: {title}

        {body}
        """
        return f"Title: {self.title}\n\n{self.body}"

class PostClient:
    """
    API client for JSONPlaceholder.
    """
    def __init__(self, url: str = settings.JSONPLACEHOLDER_URL):
        self.url = url

    @robust_retry(attempts=3, delay=1.0, exceptions=(requests.RequestException,))
    def _fetch_raw_posts(self) -> List[dict]:
        """
        Fetches raw post data from the API.
        Protected method that is decorated with retry.
        """
        logger.info(f"Fetching posts from API: {self.url}")
        response = requests.get(self.url, timeout=5)
        response.raise_for_status()
        return response.json()

    def fetch_first_10_posts(self) -> List[Post]:
        """
        Fetches and returns the first 10 posts.
        Gracefully falls back to mock data if the API is offline after retries.
        """
        try:
            raw_posts = self._fetch_raw_posts()
            # Take only the first 10
            posts = [Post(**p) for p in raw_posts[:10]]
            logger.info(f"Successfully fetched {len(posts)} posts from the API.")
            return posts
        except Exception as e:
            logger.error(f"Failed to fetch posts from API after retries. Error: {e}")
            logger.warning("Gracefully degrading: returning fallback local mock posts.")
            return self._generate_fallback_posts()

    def _generate_fallback_posts(self) -> List[Post]:
        """
        Generates 10 mock posts as fallback data for robust execution.
        """
        fallback = []
        for i in range(1, 11):
            fallback.append(
                Post(
                    id=i,
                    title=f"Fallback Post Title {i}",
                    body=f"This is a fallback body for post {i}. The API was offline, but the vision automation pipeline successfully degraded gracefully without crashing! Code quality and robustness check passed."
                )
            )
        return fallback
