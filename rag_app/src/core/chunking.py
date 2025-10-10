import re
from abc import ABC, abstractmethod
from typing import Any

from ..config.prompts.chunking import chunking_prompt_template
from .llm import LLM


class Chunker(ABC):
    def __init__(self):
        self.chunks: list[dict] = []

    @abstractmethod
    def chunk(self, text: str) -> list[dict]:
        pass


class SWChunker(Chunker):
    """Sliding Window Chunker with configurable chunk size and chunk overlap."""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 1000):
        super().__init__()
        if chunk_size <= 0 or chunk_overlap <= 0:
            raise ValueError("Chunk_size and chunk_overlap must be positive")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[dict]:
        n = len(text)
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, n, step):
            end_idx = min(i + self.chunk_size, n)
            chunk = text[i:end_idx]
            index = i // step
            self.chunks.append({"index": index, "start": i, "end": end_idx, "content": chunk})
        return self.chunks


class ParagraphChunker(Chunker):
    """Paragraph Chunker."""

    def __init__(self):
        super().__init__()

    def chunk(self, text: str) -> list[dict]:
        paragraphs = re.split(r"\n\s*\n", text.strip())
        for i, paragraph in enumerate(paragraphs):
            self.chunks.append({"index": i, "content": paragraph})
        return self.chunks


class SectionChunker(Chunker):
    """Section Chunker, with # level (0-6) configurable."""

    def __init__(self, level: int = 2):
        super().__init__()
        self.level = level

    def chunk(self, text: str) -> list[dict]:
        header_pattern = r"^(#{" + str(self.level) + r"} )(.+)$"
        pattern = re.compile(header_pattern, re.MULTILINE)
        parts = pattern.split(text)

        for i in range(1, len(parts), 3):
            header = parts[i] + parts[i + 1]  # "## " + "Title"
            header = header.strip()
            content = ""
            if i + 2 < len(parts):
                content = parts[i + 2].strip()
            if content:
                section = f"{header}\n\n{content}"
            else:
                section = header

            self.chunks.append({"index": i // 3, "content": section})
        return self.chunks


class LLMChunker(Chunker):
    """LLM Chunker, uses an LLM to understand the semantics of the text before chunking it into logical parts."""

    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm
        self.prompt_template = chunking_prompt_template.strip()

    def chunk(self, doc: dict[str, Any]) -> list[dict]:
        prompt = self.prompt_template.format(document=doc)
        _, response = self.llm.generate(prompt)
        sections = [s.strip() for s in response.split("---") if s.strip()]
        self.chunks = [{"index": i, "content": s.strip()} for i, s in enumerate(sections)]
        return self.chunks
