"""
PDF document extractor
"""
from typing import List
import base64
import fitz  # PyMuPDF

from src.utils import get_logger

from .base_extractor import BaseDocumentExtractor
from ..types import (
    ImageData,
    PDFFileMetadata,
    ExtractionResult,
)


def _mostly_inside(inner: fitz.Rect, outer: fitz.Rect, ratio: float = 0.5) -> bool:
    """True if more than ``ratio`` of ``inner``'s area lies within ``outer``."""
    area = inner.get_area()
    if area == 0:
        return outer.contains(inner.tl)
    return (inner & outer).get_area() / area > ratio


def _split_by_gaps(items: list, axis: str) -> list:
    """
    Splits (rect, text) items into groups separated by a gap that no item crosses.

    ``axis="x"`` finds columns (vertical gaps), ``axis="y"`` finds bands
    (horizontal gaps). Groups are returned left to right / top to bottom.
    """
    start, end = ("x0", "x1") if axis == "x" else ("y0", "y1")
    items = sorted(items, key=lambda it: getattr(it[0], start))
    groups = [[items[0]]]
    group_end = getattr(items[0][0], end)
    for item in items[1:]:
        if getattr(item[0], start) > group_end:
            groups.append([item])
        else:
            groups[-1].append(item)
        group_end = max(group_end, getattr(item[0], end))
    return groups


def _reading_order(items: list) -> list:
    """
    Sorts (rect, text) items in reading order with a recursive XY-cut.

    At each level the region is first split into columns (vertical gaps no
    item crosses) and each column is read entirely before the next one; if
    there are no columns, it is split into horizontal bands read top to bottom.
    This keeps multi-column brochure layouts readable: each column stays
    together and a label stays with the heading of its own column, even when
    the heading of the next column starts a few points higher.
    """
    if len(items) <= 1:
        return list(items)
    for axis in ("x", "y"):
        groups = _split_by_gaps(items, axis)
        if len(groups) > 1:
            return [item for group in groups for item in _reading_order(group)]
    # Overlapping items that cannot be separated: top to bottom, left to right
    return sorted(items, key=lambda it: (it[0].y0, it[0].x0))


def _chars(text: str) -> int:
    """Number of non-whitespace characters."""
    return sum(1 for c in text if not c.isspace())


class PDFExtractor(BaseDocumentExtractor[PDFFileMetadata]):
    """Extractor for PDF files"""
    
    def __init__(self, file_path: str):
        """
        Initializes the PDF extractor
        
        Args:
            file_path: Path to the PDF file
        """
        super().__init__(file_path)
        self.logger = get_logger(__name__)
        self.logger.debug(
            "Initializing PDFExtractor",
            extra={"file_path": str(self.file_path)}
        )
    
    def get_metadata(self) -> PDFFileMetadata:
        """
        Gets PDF file metadata including total pages and total images
        
        Returns:
            PDFFileMetadata with file name, type, total pages, and total images
        """
        base_metadata = super().get_metadata()
        
        try:
            self.logger.debug(f"Getting PDF metadata: {self.file_name}")
            doc = fitz.open(self.file_path)
            total_pages = len(doc)
            total_images = self._count_images(doc)
            doc.close()
            
            metadata = PDFFileMetadata(
                file_name=base_metadata.file_name,
                file_type=base_metadata.file_type,
                total_pages=total_pages,
                total_images=total_images,
                chapters=False,  # False by default until we analyze the text to determine if it has chapters
            )
            
            self.logger.debug(
                "PDF metadata retrieved",
                extra={
                    "file_name": metadata.file_name,
                    "total_pages": total_pages,
                    "total_images": total_images,
                    "chapters": False
                }
            )
            
            return metadata
            
        except Exception as e:
            self.logger.error(
                f"Error getting PDF metadata: {str(e)}",
                extra={"file_path": str(self.file_path)},
                exc_info=True
            )
            raise Exception(f"Error getting PDF metadata: {str(e)}") from e
    
    def _count_images(self, doc) -> int:
        """
        Counts total number of images in the PDF document
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Total number of images in the document
        """
        total_images = 0
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            total_images += len(image_list)
        return total_images
    
    def extract(self, extract_images: bool = False) -> ExtractionResult[PDFFileMetadata]:
        """
        Extracts content from a PDF file
        
        Args:
            extract_images: If True, also extracts images from the PDF
        
        Returns:
            ExtractionResult with extracted content, images, and metadata
        """
        try:
            self.logger.info(
                "Starting PDF extraction",
                extra={
                    "file_path": str(self.file_path),
                    "file_name": self.file_name,
                    "extract_images": extract_images
                }
            )
            
            # Extract text
            text_pages = self._extract_text_from_pdf()
            
            # Extract images if requested
            images = None
            if extract_images:
                extracted_images = self._extract_images_from_pdf()
                images = extracted_images if extracted_images else []
            
            # Get metadata
            metadata = self.get_metadata()
            
            result = ExtractionResult[PDFFileMetadata](
                content=text_pages,
                images=images,
                metadata=metadata
            )
            
            self.logger.info(
                "PDF extraction completed successfully",
                extra={
                    "file_path": str(self.file_path),
                    "file_name": self.file_name,
                    "pages_extracted": len(text_pages),
                    "images_extracted": len(images) if images else 0,
                    "extract_images": extract_images
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"Error extracting PDF content: {str(e)}",
                extra={
                    "file_path": str(self.file_path),
                    "file_name": self.file_name,
                    "extract_images": extract_images
                },
                exc_info=True
            )
            raise Exception(f"Error extracting content from PDF {self.file_name}: {str(e)}") from e
    
    def _extract_text_from_pdf(self) -> List[str]:
        """
        Extracts text from each page of the PDF
        
        Returns:
            List of strings, one per page
        """
        try:
            self.logger.debug(f"Extracting text from PDF: {self.file_name}")
            text_pages = []
            doc = fitz.open(self.file_path)
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                text_pages.append(self._page_to_text(page, page_num + 1))
            
            doc.close()
            
            self.logger.debug(
                "Text extraction from PDF completed",
                extra={
                    "file_name": self.file_name,
                    "pages_extracted": len(text_pages),
                    "total_pages": total_pages
                }
            )
            
            return text_pages
            
        except Exception as e:
            self.logger.error(
                f"Error extracting text from PDF: {str(e)}",
                extra={"file_path": str(self.file_path)},
                exc_info=True
            )
            raise Exception(f"Error extracting text from PDF: {str(e)}") from e
    
    def _page_to_text(self, page, page_number: int) -> str:
        """
        Extracts the text of a page keeping its layout readable for the LLM.

        - Tables are converted to markdown, so each value stays next to its
          row and column headers instead of ending up as a loose number.
        - The rest of the text is read block by block in visual reading order
          (columns, then bands; see ``_reading_order``), so labels like
          "Solo en Plug-in" stay next to the heading they belong to.

        If anything goes wrong, falls back to the plain ``page.get_text()``.
        """
        try:
            tables = self._find_tables(page)

            items = []  # (rect, text)
            skipped = {id(t): [] for t, _ in tables}
            for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
                if block_type != 0 or not text.strip():  # 0 = text block
                    continue
                block_rect = fitz.Rect(x0, y0, x1, y1)
                table = next((t for t, rect in tables if _mostly_inside(block_rect, rect)), None)
                if table is not None:
                    skipped[id(table)].append((block_rect, text.strip()))
                    continue
                items.append((block_rect, text.strip()))

            for table, rect in tables:
                markdown = self._table_to_markdown(page, table)
                # Safety net: if the markdown lost text, keep the original blocks.
                if _chars(markdown) < 0.8 * _chars("".join(t for _, t in skipped[id(table)])):
                    self.logger.debug(
                        "Table discarded: markdown lost text",
                        extra={"file_name": self.file_name, "page": page_number},
                    )
                    items.extend(skipped[id(table)])
                    continue
                items.append((rect, markdown))

            return "\n\n".join(text for _, text in _reading_order(items))

        except Exception as e:
            self.logger.warning(
                f"Layout-aware extraction failed, using plain text: {str(e)}",
                extra={"file_name": self.file_name, "page": page_number},
            )
            return page.get_text()

    @staticmethod
    def _find_tables(page) -> list:
        """Returns [(table, rect)] for tables with at least 2 rows and 2 columns."""
        found = []
        for table in page.find_tables().tables:
            if table.row_count >= 2 and table.col_count >= 2:
                found.append((table, fitz.Rect(table.bbox)))
        return found

    @staticmethod
    def _table_to_markdown(page, table) -> str:
        """
        Converts a detected table to markdown.

        The text of each cell is read again clipping the page to the cell box:
        the text that ``table.extract()`` returns comes out broken in PDFs with
        manual letter spacing (e.g. "P otenci (CV) a").
        """
        rows = []
        for row in table.rows:
            cells = []
            for cell in row.cells:
                text = page.get_text("text", clip=fitz.Rect(cell)) if cell else ""
                cells.append(" ".join(text.split()).replace("|", "\\|"))
            if any(cells):
                rows.append(cells)
        if not rows:
            return ""

        # Drop columns that are empty in every row (layout spacers)
        keep = [i for i in range(len(rows[0])) if any(r[i] for r in rows)]
        rows = [[r[i] for i in keep] for r in rows]

        lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(keep)]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n".join(lines)

    def _extract_images_from_pdf(self) -> List[ImageData]:
        """
        Extracts images from each page of the PDF
        
        Returns:
            List of ImageData objects with image information
        """
        try:
            self.logger.debug(f"Extracting images from PDF: {self.file_name}")
            images: List[ImageData] = []
            doc = fitz.open(self.file_path)
            total_image_counter = 0  # Counter for total images in the document
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                image_list = page.get_images()
                
                for image_index, img in enumerate(image_list):
                    # Get the image
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # Image data
                    image_data = base_image["image"]
                    image_format = base_image["ext"]
                    
                    # Convert to base64 for API usage (e.g., OpenAI Vision)
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # Increment total image counter
                    total_image_counter += 1
                    
                    images.append(ImageData(
                        page=page_num + 1,                    # Pages start at 1
                        image_number_in_page=image_index + 1,  # Image number in page (starting at 1)
                        image_number=total_image_counter,      # Total image number in document (starting at 1)
                        image_base64=image_base64,             # Base64 encoded image (for API usage)
                        image_format=image_format
                    ))
            
            doc.close()
            
            self.logger.debug(
                "Image extraction from PDF completed",
                extra={
                    "file_name": self.file_name,
                    "images_extracted": len(images),
                    "total_pages": total_pages
                }
            )
            
            return images
            
        except Exception as e:
            self.logger.error(
                f"Error extracting images from PDF: {str(e)}",
                extra={"file_path": str(self.file_path)},
                exc_info=True
            )
            raise Exception(f"Error extracting images from PDF: {str(e)}") from e


