"""
LLM-based document chooser implementation
Uses text models to select relevant documents based on user queries
"""

from typing import Optional, List, Dict, Any
from llms.text import BaseTextModel, OpenAITextModel
from src.utils.utils import PromptLoader
from src.utils import get_logger


class LLMDocumentChooser:
    """
    Document selector using LLM models.
    Analyzes document descriptions and selects the most relevant ones according to user query.
    Supports various text models (OpenAI, Anthropic, etc.) through BaseTextModel interface.
    """

    def __init__(
        self,
        *,
        text_model: Optional[BaseTextModel] = None,
        max_tokens: int = 500,
        temperature: float = 0.2
    ):
        """
        Initializes LLM document selector.

        Args:
            text_model: BaseTextModel instance to use. Must be provided.
            max_tokens: Maximum tokens for response (default 500).
            temperature: Temperature for generation (default 0.2, lower for more focused selections).

        Raises:
            ValueError: If text_model is not provided or is not a BaseTextModel instance.
        """
        if not isinstance(text_model, BaseTextModel):
            raise ValueError(
                "The 'text_model' parameter must be provided and be a BaseTextModel instance. "
                "For other providers, provide a 'text_model' instance."
            )
        
        self.text_model: BaseTextModel = text_model
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature
        self.logger = get_logger(__name__)
        
        self.logger.info(
            "Initializing LLMDocumentChooser",
            extra={
                "max_tokens": max_tokens,
                "temperature": temperature,
                "text_model_type": type(text_model).__name__
            }
        )

    def choose_documents(
        self,
        markdown_descriptions: str,
        user_query: str,
        summaries: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Selects most relevant documents based on user query.

        Args:
            markdown_descriptions: Document descriptions in markdown format.
            user_query: User query or question.
            summaries: List of dictionaries with document information (must include file_id).

        Returns:
            List[str]: List of selected document file_ids.

        Raises:
            ValueError: If markdown_descriptions or user_query are empty.
            Exception: If selection fails.
        """
        # Validations
        if not markdown_descriptions or not isinstance(markdown_descriptions, str):
            self.logger.error("markdown_descriptions must be a non-empty string")
            raise ValueError("markdown_descriptions must be a non-empty string")

        if not user_query or not isinstance(user_query, str):
            self.logger.error("user_query must be a non-empty string")
            raise ValueError("user_query must be a non-empty string")

        markdown_descriptions = markdown_descriptions.strip()
        user_query = user_query.strip()

        if not markdown_descriptions or not user_query:
            self.logger.error("Parameters cannot be empty after strip")
            raise ValueError("Parameters cannot be empty after strip")

        self.logger.debug(
            "Starting document selection",
            extra={
                "markdown_length": len(markdown_descriptions),
                "user_query_length": len(user_query),
                "total_documents": len(summaries),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        )

        # Build prompt using template
        prompt, system_prompt = self._get_chooser_prompt(markdown_descriptions, user_query)

        # Select documents using text model
        try:
            response = self.text_model.call_text_model(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Parse response to extract file_ids
            selected_ids = self._parse_response(response, summaries)
            
            self.logger.info(
                "Documents selected successfully",
                extra={
                    "total_available": len(summaries),
                    "selected_count": len(selected_ids),
                    "selected_ids": selected_ids
                }
            )
            
            return selected_ids
            
        except Exception as e:
            self.logger.error(
                f"Error selecting documents: {str(e)}",
                extra={
                    "markdown_length": len(markdown_descriptions),
                    "user_query": user_query,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise Exception(f"Error selecting documents: {str(e)}") from e

    @staticmethod
    def _get_chooser_prompt(markdown_descriptions: str, user_query: str) -> tuple[str, str]:
        """
        Builds prompt for document selection.

        Args:
            markdown_descriptions: Document descriptions in markdown format.
            user_query: User query.

        Returns:
            tuple[str, str]: (prompt, system_prompt)
        """
        system_prompt = PromptLoader.read_file(
            "src/search/document_selection/chooser/prompt.md"
        )
        
        prompt = f"""# Consulta del usuario:
{user_query}

# Documentos disponibles:
{markdown_descriptions}
"""
        
        return prompt, system_prompt

    def _parse_response(self, response: str, summaries: List[Dict[str, Any]]) -> List[str]:
        """
        Parses LLM response to extract selected file_ids.

        Args:
            response: LLM model response.
            summaries: List of dictionaries with document information.

        Returns:
            List[str]: List of valid file_ids.
        """
        self.logger.debug(
            "Parsing LLM response",
            extra={"raw_response": response}
        )

        # Clean response
        response = response.strip()
        
        # Extract file_ids (separated by commas, spaces, or newlines)
        import re
        # Search for all possible file_ids (assuming format doc_XXX or similar)
        potential_ids = re.split(r'[,\s\n]+', response)
        
        # Clean and filter empty IDs
        potential_ids = [id.strip() for id in potential_ids if id.strip()]
        
        # Create set of valid file_ids from available documents
        valid_file_ids = {summary.get("file_id") for summary in summaries if summary.get("file_id")}
        
        # Filter only file_ids that exist in available documents
        selected_ids = [id for id in potential_ids if id in valid_file_ids]
        
        self.logger.debug(
            "Parsed IDs",
            extra={
                "potential_ids": potential_ids,
                "valid_ids": list(valid_file_ids),
                "selected_ids": selected_ids
            }
        )

        if not selected_ids:
            self.logger.warning(
                "Could not parse valid file_ids from response",
                extra={"response": response}
            )
        
        return selected_ids

    def choose_documents_with_details(
        self,
        markdown_descriptions: str,
        user_query: str,
        summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Selects documents and returns complete information.

        Args:
            markdown_descriptions: Document descriptions in markdown format.
            user_query: User query or question.
            summaries: List of dictionaries with document information.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with complete information of selected documents.
        """
        selected_ids = self.choose_documents(markdown_descriptions, user_query, summaries)
        
        # Filter summaries to get only selected ones
        selected_summaries = [
            summary for summary in summaries 
            if summary.get("file_id") in selected_ids
        ]
        
        self.logger.info(
            "Documents selected with details",
            extra={
                "selected_count": len(selected_summaries)
            }
        )
        
        return selected_summaries

