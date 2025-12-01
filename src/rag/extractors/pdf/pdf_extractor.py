"""
PDF document extractor
"""
from typing import List
import base64
import fitz  # PyMuPDF
from ..base import (
    BaseDocumentExtractor,
    ImageData,
    PDFFileMetadata,
    ExtractionResult
)


class PDFExtractor(BaseDocumentExtractor[PDFFileMetadata]):
    """Extractor for PDF files"""
    
    def __init__(self, file_path: str):
        """
        Initializes the PDF extractor
        
        Args:
            file_path: Path to the PDF file
        """
        super().__init__(file_path)
    
    def get_metadata(self) -> PDFFileMetadata:
        """
        Gets PDF file metadata including total pages and total images
        
        Returns:
            PDFFileMetadata with file name, type, total pages, and total images
        """
        base_metadata = super().get_metadata()
        
        try:
            doc = fitz.open(self.file_path)
            total_pages = len(doc)
            total_images = self._count_images(doc)
            doc.close()
            
            return PDFFileMetadata(
                file_name=base_metadata.file_name,
                file_type=base_metadata.file_type,
                total_pages=total_pages,
                total_images=total_images
            )
            
        except Exception as e:
            raise Exception(f"Error getting PDF metadata: {str(e)}")
    
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
            # Extract text
            text_pages = self._extract_text_from_pdf()
            
            # Extract images if requested
            images = None
            if extract_images:
                extracted_images = self._extract_images_from_pdf()
                images = extracted_images if extracted_images else []
            
            # Get metadata
            metadata = self.get_metadata()
            
            return ExtractionResult[PDFFileMetadata](
                content=text_pages,
                images=images,
                metadata=metadata
            )
            
        except Exception as e:
            raise Exception(f"Error extracting content from PDF {self.file_name}: {str(e)}")
    
    def _extract_text_from_pdf(self) -> List[str]:
        """
        Extracts text from each page of the PDF
        
        Returns:
            List of strings, one per page
        """
        try:
            text_pages = []
            doc = fitz.open(self.file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_pages.append(text)
            
            doc.close()
            return text_pages
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _extract_images_from_pdf(self) -> List[ImageData]:
        """
        Extracts images from each page of the PDF
        
        Returns:
            List of ImageData objects with image information
        """
        try:
            images: List[ImageData] = []
            doc = fitz.open(self.file_path)
            total_image_counter = 0  # Counter for total images in the document
            
            for page_num in range(len(doc)):
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
                        image_number=total_image_counter, # Total image number in document (starting at 1)
                        image_base64=image_base64,            # Base64 encoded image (for API usage)
                        image_format=image_format
                    ))
            
            doc.close()
            return images
            
        except Exception as e:
            raise Exception(f"Error extracting images from PDF: {str(e)}")

