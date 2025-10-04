from typing import List, Dict
import requests
import zipfile
import io
import pickle
import frontmatter

class DataLoader:    
    def __init__(self):
        self.repo_urls: List[str] = []
        self.url_prefix = "https://codeload.github.com"
    
    def _validate_url(self, url: str) -> None:
        if not url.startswith("http://") and not url.startswith("https://"):
            raise ValueError(f"URL does not start with http:// or https:// -> {url}")
        
        if not url.contains("github.com"):
            raise ValueError(f"URL is not a valid github URL -> {url}")
    
    def load_repo(self, url: str) -> None:
        self._validate_url(url)
        
        pass


def load_docs() -> List[Dict]:
    """Placeholder loader returning an empty corpus.
    TODO: implement GitHub ZIP download + frontmatter parsing.
    """
    return []
