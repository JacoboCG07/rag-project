"""
Metadata Extractor
Extracts document metadata using an LLM based on the user query.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.llms.text import BaseTextModel
from src.utils import get_logger
from src.utils.utils import PromptLoader
import json
import re


class MetadataExtractor:
    """
    Extracts relevant metadata from documents using an LLM.

    Analyses the user query and document markdown to identify
    specific metadata such as pages, chapters, images, etc.
    """
    
    def __init__(
        self,
        text_model: BaseTextModel,
        max_tokens: int = 2000,
        temperature: float = 0.2
    ):
        """
        Initializes the metadata extractor.
        
        Args:
            text_model: LLM used to extract metadata.
            max_tokens: Maximum tokens for the LLM response.
            temperature: LLM temperature (0.0-1.0).
        """
        self.text_model = text_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = get_logger(__name__)
        
        # Load prompt template
        self.prompt_template = self._load_prompt()
        
        self.logger.info(
            "MetadataExtractor initialized",
            extra={
                "model": text_model.__class__.__name__,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
    
    def _load_prompt(self) -> str:
        """
        Loads the prompt from metadata_extractor_prompt.md.

        Returns:
            str: Prompt template.

        Raises:
            FileNotFoundError: If metadata_extractor_prompt.md is missing.
        """
        prompt_path = Path(__file__).parent / "metadata_extractor_prompt.md"
        return PromptLoader.read_file(str(prompt_path))
    
    def extract(
        self,
        user_query: str,
        markdown_documents: str,
        documents_info: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts metadata from documents based on the user query.
        
        Args:
            user_query: User query.
            markdown_documents: Markdown describing the documents.
            documents_info: List of dicts with per-document info.
        
        Returns:
            Dict[str, Dict]: Extracted metadata keyed by file_id.
            
            Example:
            {
                "doc_001": {
                    "pages": [1, 2, 3],
                    "chapters": null,
                    "search_image": False,
                    "num_image": null,
                    "file_type": "PDF"
                }
            }
        """
        self.logger.debug(
            "Extracting metadata from documents",
            extra={
                "user_query": user_query,
                "num_documents": len(documents_info)
            }
        )
        
        try:
            # Build prompt and system_prompt
            prompt, system_prompt = self._build_prompt(user_query, markdown_documents)
            
            # Call LLM via call_text_model (same pattern as LLMDocumentChooser)
            self.logger.debug("Calling LLM for metadata extraction")
            response = self.text_model.call_text_model(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            self.logger.info(
                "Metadata extraction prompts and response",
                extra={
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "response": response,
                }
            )
            
            # Parse response
            metadata_dict = self._parse_response(response, documents_info)
            
            self.logger.info(
                "Metadata extraction completed",
                extra={
                    "documents_with_metadata": len(metadata_dict)
                }
            )
            
            return metadata_dict
            
        except Exception as e:
            self.logger.error(
                f"Error extracting metadata: {str(e)}",
                extra={"error_type": type(e).__name__},
                exc_info=True
            )
            raise
    
    def _build_prompt(
        self,
        user_query: str,
        markdown_documents: str,
    ) -> tuple[str, str]:
        """
        Builds the prompt and system_prompt for the LLM.
        
        Same structure as LLMDocumentChooser:
        - system_prompt: General instructions (from prompt.md)
        - prompt: Specific data (query + documents)
        
        Args:
            user_query: User query.
            markdown_documents: Document markdown.
        
        Returns:
            tuple[str, str]: (prompt, system_prompt)
        """
        # system_prompt: general instructions from prompt.md
        system_prompt = self.prompt_template
        
        # prompt: query + documents
        prompt = f"""# User query:
{user_query}

# Available documents:
{markdown_documents}
"""
        return prompt, system_prompt
    
    def _normalize_response(self, response: Any) -> str:
        """Ensures response is a non-None string to avoid JSONDecodeError on empty input."""
        if response is None:
            return ""
        if not isinstance(response, str):
            return str(response) if response else ""
        return response

    def _extract_json_from_markdown(self, text: str) -> Optional[str]:
        """
        Tries to extract a JSON block from markdown (```json ... ``` or ``` ... ```).
        Many LLMs wrap JSON in fenced code blocks.
        """
        if not text or not text.strip():
            return None
        text = text.strip()
        # ```json ... ``` (case insensitive)
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # First { to last } if there are no backticks
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
        return None

    def _default_metadata_dict(
        self, documents_info: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Builds default metadata (no filters) per document when the LLM returns invalid JSON,
        so search can still proceed by file_id.
        """
        result = {}
        for doc in documents_info:
            file_id = doc.get("file_id") if isinstance(doc, dict) else getattr(doc, "file_id", None)
            file_type = doc.get("file_type") if isinstance(doc, dict) else getattr(doc, "file_type", "PDF")
            if file_id is None:
                continue
            result[str(file_id)] = {
                "pages": None,
                "chapters": None,
                "search_image": False,
                "num_image": None,
                "file_type": file_type or "PDF",
            }
        return result

    def _parse_response(
        self,
        response: str,
        documents_info: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Parses the LLM response and validates data.
        If the response is empty or not valid JSON, returns default metadata
        per document (file_id/file_type only) so the flow does not break.
        """
        raw = self._normalize_response(response)
        text = raw.strip()

        if not text:
            self.logger.warning(
                "LLM returned empty response for metadata extraction; using default metadata per document"
            )
            return self._default_metadata_dict(documents_info)

        try:
            metadata_dict = json.loads(text)
            if isinstance(metadata_dict, dict):
                return metadata_dict
            self.logger.warning(
                "LLM response JSON is not an object; using default metadata",
                extra={"response_preview": text[:300]}
            )
            return self._default_metadata_dict(documents_info)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from ```json ... ``` blocks
        extracted = self._extract_json_from_markdown(raw)
        if extracted:
            try:
                metadata_dict = json.loads(extracted)
                if isinstance(metadata_dict, dict):
                    self.logger.info("Parsed JSON from markdown code block")
                    return metadata_dict
            except json.JSONDecodeError:
                pass

        self.logger.error(
            "Failed to parse LLM response as JSON; using default metadata per document",
            extra={
                "response_preview": raw[:500] if raw else "(empty)",
                "documents_count": len(documents_info)
            }
        )
        return self._default_metadata_dict(documents_info)
    
    def _validate_metadata(
        self,
        file_id: str,
        metadata: Dict[str, Any],
        document_info: Dict[str, Any]
    ) -> bool:
        """
        Validates that extracted metadata is consistent with the document.
        
        Args:
            file_id: Document id.
            metadata: Extracted metadata.
            document_info: Ground-truth document info.
        
        Returns:
            bool: True if valid.
        """
        # TODO: Add validations
        # - pages do not exceed total_pages
        # - chapters within valid range
        # - images do not exceed total_num_image
        # - etc.
        return True
