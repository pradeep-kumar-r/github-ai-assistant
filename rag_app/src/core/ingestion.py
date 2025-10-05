import pickle
from io import BytesIO
from zipfile import ZipFile

import frontmatter as fm
import requests

from ..logger import logger


class DataLoader:
    def __init__(self):
        self.url_prefix: str = "https://codeload.github.com"
        self.repo_data: list[dict] = []

    def _validate_url(self, url: str) -> None:
        if not url.startswith("http://") and not url.startswith("https://"):
            msg = f"URL does not start with http:// or https:// -> {url}"
            logger.error(msg)
            raise ValueError(msg)

        if "github.com" not in url:
            msg = f"URL is not a valid github URL -> {url}"
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def _parse_url(url: str) -> tuple[str, str]:
        components = url.split("/")
        repo_owner = components[-2]
        repo_name = components[-1]
        return repo_owner, repo_name

    def load_repo_from_url(self, url: str) -> list[dict]:
        self._validate_url(url)
        repo_owner, repo_name = self._parse_url(url)
        new_url = f"{self.url_prefix}/{repo_owner}/{repo_name}/zip/refs/heads/main"
        response = requests.get(new_url, timeout=30)

        if response.status_code != 200:
            msg = f"Failed to download repository -> {url}"
            logger.error(msg)
            raise ValueError(msg)

        zf = ZipFile(BytesIO(response.content))
        logger.info(f"Downloaded repository -> {url}")
        file_count = 0

        for file_info in zf.infolist():
            filename = file_info.filename
            if filename.endswith(".md") or filename.endswith(".mdx"):
                try:
                    with zf.open(filename, "r") as f:
                        content = f.read().decode(encoding="utf-8", errors="ignore")
                    data = fm.load(content).to_dict()
                    data['filename'] = filename
                    data['url'] = url
                    self.repo_data.append(data)
                    file_count += 1
                    logger.info(f"Added file to knowledge base -> {filename}")
                except Exception as e:
                    msg = f"Failed to process file {filename}: {str(e)}"
                    logger.error(msg)

        zf.close()
        logger.info(f"Processed {file_count} files from repository -> {url}")
        return self.repo_data

    def load_existing_repo_data(self, path: str) -> None:
        with open(path, "rb") as f:
            self.repo_data = pickle.load(f)
            logger.info(f"Loaded existing repo data from {path}")

    def save_repo_data(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.repo_data, f)
            logger.info(f"Saved repo data to {path}")
