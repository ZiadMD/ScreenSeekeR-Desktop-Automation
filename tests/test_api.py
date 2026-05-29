import pytest
import requests
from src.api.posts import PostClient, Post

def test_post_formatting():
    post = Post(id=1, title="Hello World", body="This is a test post body.")
    formatted = post.to_formatted_text()
    
    assert "Title: Hello World" in formatted
    assert "This is a test post body." in formatted

def test_api_fetch_success(monkeypatch):
    client = PostClient()

    # Mock raw response
    mock_posts = [
        {"id": 1, "title": "Title 1", "body": "Body 1", "userId": 1},
        {"id": 2, "title": "Title 2", "body": "Body 2", "userId": 1}
    ]

    def mock_get(*args, **kwargs):
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                return mock_posts
        return MockResponse()

    monkeypatch.setattr(requests, "get", mock_get)

    posts = client.fetch_first_10_posts()
    
    assert len(posts) == 2
    assert posts[0].id == 1
    assert posts[0].title == "Title 1"
    assert posts[0].body == "Body 1"

def test_api_fetch_graceful_fallback(monkeypatch):
    client = PostClient()

    def mock_get(*args, **kwargs):
        raise requests.RequestException("API Offline Connection Error")

    monkeypatch.setattr(requests, "get", mock_get)

    # Calling fetch should not crash; it must return fallback mock posts
    posts = client.fetch_first_10_posts()
    
    assert len(posts) == 10
    assert posts[0].id == 1
    assert "Fallback Post Title" in posts[0].title
    assert "API was offline" in posts[0].body
