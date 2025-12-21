"""
Text chunker implementation
Based on existing logic with improvements for overlap and flexibility
"""

import re
from typing import List, Tuple, Optional

from .base_chunker import BaseChunker
from src.utils import get_logger


class TextChunker(BaseChunker):
    """
    Text chunker that divides texts into segments.
    Based on existing logic: ensure_length_segments + group_segments.
    Supports overlap and chapter detection.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 2000,
        overlap: int = 0,
        detect_chapters: bool = True
    ):
        """
        Initializes the text chunker.

        Args:
            chunk_size: Maximum size of each chunk (default 2000 characters).
            overlap: Number of characters to overlap between chunks (default 0).
            detect_chapters: Whether to detect chapters in text (default True).
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.detect_chapters = detect_chapters
        self.logger = get_logger(__name__)
        
        self.logger.info(
            "Initializing TextChunker",
            extra={
                "chunk_size": chunk_size,
                "overlap": overlap,
                "detect_chapters": detect_chapters
            }
        )

    def chunk(
        self,
        *,
        texts: List[str],
        return_metadata: bool = False
    ) -> List[str] | Tuple[List[str], List[dict]]:
        """
        Chunks a list of texts into smaller segments.

        Args:
            texts: List of texts to chunk (typically pages).
            return_metadata: If True, returns metadata (pages, chapters) along with chunks.

        Returns:
            If return_metadata=False: List[str] - List of chunked texts.
            If return_metadata=True: Tuple[List[str], List[dict]] - (chunks, metadata_list)
                where metadata_list contains dicts with 'pages' and optionally 'chapters'.
        """
        if not texts:
            self.logger.debug("Empty texts list provided, returning empty result")
            return [] if not return_metadata else ([], [])

        self.logger.debug(
            "Starting text chunking",
            extra={
                "input_texts_count": len(texts),
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "detect_chapters": self.detect_chapters,
                "return_metadata": return_metadata
            }
        )

        # Step 1: Ensure each text doesn't exceed chunk_size (split if necessary)
        separate_texts, pages = self._ensure_length_segments(texts=texts)

        # Step 2: Group segments up to chunk_size
        chunks, pages_groups = self._group_segments(
            texts=separate_texts,
            pages=pages
        )

        # Step 3: Detect chapters if enabled
        chapters = None
        if self.detect_chapters:
            chapters = self._get_chapters_of_segments(segments=chunks)

        if return_metadata:
            metadata_list = []
            for i, pages_group in enumerate(pages_groups):
                metadata = {
                    'pages': pages_group,
                    'chapters': chapters[i] if chapters and i < len(chapters) and chapters[i] else ""
                }
                metadata_list.append(metadata)
            
            self.logger.info(
                "Text chunking completed with metadata",
                extra={
                    "input_texts_count": len(texts),
                    "chunks_count": len(chunks),
                    "chapters_detected": sum(1 for c in chapters if c) if chapters else 0
                }
            )
            return chunks, metadata_list

        self.logger.info(
            "Text chunking completed",
            extra={
                "input_texts_count": len(texts),
                "chunks_count": len(chunks)
            }
        )
        return chunks

    def _ensure_length_segments(
        self,
        *,
        texts: List[str]
    ) -> Tuple[List[str], List[int]]:
        """
        Ensures that no text exceeds chunk_size by splitting if necessary.
        Splits at word boundaries to avoid cutting words.

        Args:
            texts: List of texts to process.

        Returns:
            Tuple[List[str], List[int]]: (separated texts, corresponding page numbers).
        """
        result = []
        pages = []
        split_count = 0

        for page, text in enumerate(texts, start=1):
            original_length = len(text)
            text = text.strip()

            while len(text) > self.chunk_size:
                # Find last space within chunk_size limit
                cut_point = text.rfind(' ', 0, self.chunk_size)
                if cut_point == -1:  # No space found, cut at chunk_size
                    cut_point = self.chunk_size

                result.append(text[:cut_point])
                pages.append(page)
                text = text[cut_point:].strip()
                split_count += 1

            if text:  # Add remaining text if not empty
                result.append(text)
                pages.append(page)

        if split_count > 0:
            self.logger.debug(
                "Texts split to ensure length constraints",
                extra={
                    "original_texts_count": len(texts),
                    "result_segments_count": len(result),
                    "splits_performed": split_count
                }
            )

        return result, pages

    def _group_segments(
        self,
        *,
        texts: List[str],
        pages: List[int]
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Groups text segments up to chunk_size, maintaining page information.

        Args:
            texts: List of text segments.
            pages: List of corresponding page numbers.

        Returns:
            Tuple[List[str], List[List[int]]]: (grouped chunks, list of page groups for each chunk).
        """
        grouped = []
        pages_groups = []

        current_group = []
        current_pages = set()
        current_length = 0

        for text, page in zip(texts, pages):
            text = text.strip()
            text_length = len(text)

            # Check if adding current text exceeds limit
            if current_length + text_length <= self.chunk_size:
                if text:  # Only add non-empty texts
                    current_group.append(text)
                    current_pages.add(page)
                    current_length += text_length
            else:
                # Save current group if not empty
                if current_group:
                    grouped_text = ' '.join(current_group)
                    grouped.append(grouped_text)
                    pages_groups.append(sorted(list(current_pages)))

                    # Apply overlap if configured
                    if self.overlap > 0 and current_group:
                        # Take last part of current group for overlap
                        overlap_text = self._get_overlap_text(group=current_group)
                        current_group = [overlap_text, text] if overlap_text else [text]
                        current_pages = {page}
                        current_length = len(' '.join(current_group))
                    else:
                        # Start new group with current text
                        current_group = [text]
                        current_pages = {page}
                        current_length = text_length

        # Add last group if not empty
        if current_group:
            grouped.append(' '.join(current_group))
            pages_groups.append(sorted(list(current_pages)))

        return grouped, pages_groups

    def _get_overlap_text(
        self,
        *,
        group: List[str]
    ) -> str:
        """
        Gets overlap text from the end of a group.

        Args:
            group: List of texts in current group.

        Returns:
            str: Overlap text (last part of group up to overlap size).
        """
        if not group or self.overlap == 0:
            return ""

        # Get last text in group
        last_text = group[-1]
        if len(last_text) <= self.overlap:
            return last_text

        # Find overlap point (last space within overlap limit)
        overlap_point = last_text.rfind(' ', len(last_text) - self.overlap)
        if overlap_point == -1:
            overlap_point = len(last_text) - self.overlap

        return last_text[overlap_point:].strip()

    def _get_chapters_of_segments(
        self,
        *,
        segments: List[str]
    ) -> List[List[str]]:
        """
        Detects chapters in segments.

        Args:
            segments: List of text segments.

        Returns:
            List[List[str]]: List of chapters for each segment (empty list if no chapters).
        """
        chapters = []
        current_chapter = None

        for segment in segments:
            lines = segment.split("\n")
            segment_chapters = set()

            for line in lines:
                if self._is_chapter_start(line=line):
                    current_chapter = line.strip()
                    # Limit chapter name length
                    if len(current_chapter) > 500:
                        current_chapter = current_chapter[:450]

                if current_chapter:
                    segment_chapters.add(current_chapter)

            if segment_chapters:
                chapters.append(list(segment_chapters))
            else:
                chapters.append([])

        return chapters

    @staticmethod
    def _is_chapter_start(*, line: str) -> bool:
        """
        Checks if a line is the start of a chapter.

        Args:
            line: Line to check.

        Returns:
            bool: True if line appears to be a chapter start.
        """
        line_stripped = line.strip()
        if not line_stripped:
            return False

        # Check for "capítulo" (case insensitive)
        if line_stripped.lower().startswith("capítulo"):
            return True

        # Check for Roman numerals (I, II, III, IV, V, etc.)
        if re.match(r'^[IVXLCDM]+\b', line_stripped):
            return True

        return False

