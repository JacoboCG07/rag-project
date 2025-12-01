"""
Manages extraction of information from one or multiple documents using extractors
"""
from pathlib import Path
from typing import List, Dict, Optional, Union
import inspect
from concurrent.futures import ProcessPoolExecutor, as_completed
from .factory import DocumentExtractorFactory
from .base.types import ExtractionResult, BaseFileMetadata


class DocumentExtractionManager:
    """Manages extraction of information from one or multiple documents using extractors"""
    
    def __init__(self, folder_path: str):
        """
        Initializes the document extraction manager
        
        Args:
            folder_path: Path to the folder containing documents to extract
        """
        self.folder_path = Path(folder_path)
        # Get supported extensions from factory (automatically matches available extractors)
        self.supported_extensions = DocumentExtractorFactory.get_supported_extensions()
    
    def get_files(self) -> List[Path]:
        """
        Gets the list of files in the folder that match supported extensions
        
        Returns:
            List of Paths to the found files
        """
        try:
            if not self.folder_path.exists():
                raise ValueError(f"Folder does not exist: {self.folder_path}")
            
            if not self.folder_path.is_dir():
                raise ValueError(f"Path is not a directory: {self.folder_path}")
            
            files = []
            for file_path in self.folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    files.append(file_path)
            
            return files
            
        except Exception as e:
            raise Exception(f"Error getting files from folder: {str(e)}")
    
    def get_files_by_extension(self, extension: str) -> List[Path]:
        """
        Gets files filtered by extension
        
        Args:
            extension: File extension (e.g., '.pdf')
            
        Returns:
            List of Paths to files with the specified extension
        """
        try:
            if not self.folder_path.exists():
                raise ValueError(f"Folder does not exist: {self.folder_path}")
            
            if not self.folder_path.is_dir():
                raise ValueError(f"Path is not a directory: {self.folder_path}")
            
            # Normalize extension (ensure it starts with dot and is lowercase)
            extension = extension.lower()
            if not extension.startswith('.'):
                extension = f'.{extension}'
            
            files = []
            for file_path in self.folder_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == extension:
                    files.append(file_path)
            
            return files
            
        except Exception as e:
            raise Exception(f"Error getting files by extension: {str(e)}")
    
    @staticmethod
    def _extract_file_internal(file_path: str, extract_images: bool) -> Dict:
        """
        Internal method to extract information from a document (used by both single and parallel extraction)
        
        Args:
            file_path: Path to the document to extract (as string for multiprocessing compatibility)
            extract_images: If True, extracts images from the document (only applies to PDF)
            
        Returns:
            Dict representation of ExtractionResult (for multiprocessing pickle compatibility)
        """
        from pathlib import Path
        
        file_path_obj = Path(file_path)
        
        # Validate file exists
        if not file_path_obj.exists():
            raise ValueError(f"File does not exist: {file_path_obj}")
        
        if not file_path_obj.is_file():
            raise ValueError(f"Path is not a file: {file_path_obj}")
        
        # Create extractor using factory
        extractor = DocumentExtractorFactory.create_extractor(str(file_path_obj))
        
        # Extract content
        # Check if extractor supports extract_images parameter (PDF only)
        extract_signature = inspect.signature(extractor.extract)
        if 'extract_images' in extract_signature.parameters:
            result = extractor.extract(extract_images=extract_images)
        else:
            # For extractors that don't support images (like TXT)
            result = extractor.extract()
        
        # Convert to dict for multiprocessing pickle compatibility
        # Pydantic models can be serialized as dicts
        return result.model_dump() if hasattr(result, 'model_dump') else result.dict()
    
    def extract_file(self, file_path: Path, extract_images: bool = False) -> ExtractionResult[BaseFileMetadata]:
        """
        Extracts information from a single document
        
        Args:
            file_path: Path to the document to extract
            extract_images: If True, extracts images from the document (only applies to PDF)
            
        Returns:
            ExtractionResult with extracted content from the document (typed Pydantic model)
        """
        try:
            result_dict = self._extract_file_internal(str(file_path), extract_images)
            # Reconstruct ExtractionResult from dict
            if hasattr(ExtractionResult, 'model_validate'):
                # Pydantic v2
                return ExtractionResult.model_validate(result_dict)
            else:
                # Pydantic v1
                return ExtractionResult.parse_obj(result_dict)
        except Exception as e:
            raise Exception(f"Error extracting document {file_path}: {str(e)}")
    
    def extract_files(self, extract_images: bool = False, max_workers: Optional[int] = None) -> List[ExtractionResult[BaseFileMetadata]]:
        """
        Extracts information from all supported documents in the folder using multiprocessing
        
        Args:
            extract_images: If True, extracts images from documents (only applies to PDF)
            max_workers: Maximum number of worker processes. If None, uses os.cpu_count()
        
        Returns:
            List of ExtractionResult with extracted content from each document (typed Pydantic models)
        """
        try:
            # Get all files to extract
            files = self.get_files()
            
            if not files:
                return []
            
            # Determine number of workers
            if max_workers is None:
                import os
                max_workers = os.cpu_count() or 1
            
            # Limit workers to number of files (no need for more workers than files)
            max_workers = min(max_workers, len(files))
            
            results = []
            errors = []
            
            # Extract documents in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks using the internal extraction method
                future_to_file = {
                    executor.submit(self._extract_file_internal, str(file_path), extract_images): file_path
                    for file_path in files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result_dict = future.result()
                        # result_dict is a dict representation of ExtractionResult
                        # Reconstruct ExtractionResult from dict for multiprocessing pickle compatibility
                        if hasattr(ExtractionResult, 'model_validate'):
                            # Pydantic v2
                            result = ExtractionResult.model_validate(result_dict)
                        else:
                            # Pydantic v1
                            result = ExtractionResult.parse_obj(result_dict)
                        results.append(result)
                    except Exception as e:
                        errors.append({
                            'file_path': str(file_path),
                            'error': str(e)
                        })
            
            # Log errors if any
            if errors:
                print(f"Warning: {len(errors)} document(s) failed to extract:")
                for error in errors:
                    print(f"  - {error['file_path']}: {error['error']}")
            
            return results
            
        except Exception as e:
            raise Exception(f"Error extracting documents: {str(e)}")

