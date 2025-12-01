"""
Utility functions for reading and processing prompt templates
"""
import re
from pathlib import Path
from typing import Any


class PromptLoader:
    """Utility class for loading prompt templates from files"""

    @classmethod
    def read_file(cls, relative_path: str, **kwargs: Any) -> str:
        """
        Reads a prompt template from a file and replaces placeholders.
        
        Args:
            relative_path: Path to the prompt file (relative or absolute)
            **kwargs: Variables to replace (use {{variable_name}} in file)
        
        Returns:
            str: Prompt content with placeholders replaced
        """
        absolute_path = Path(relative_path).resolve()
        
        if not absolute_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {absolute_path}")

        try:
            with open(absolute_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise IOError(f"Error reading prompt file {absolute_path}: {str(e)}") from e

        content = cls._remove_code_fences(content)

        for key, value in kwargs.items():
            placeholder = "{{" + key + "}}"
            content = content.replace(placeholder, str(value))

        return content.strip()

    @staticmethod
    def _remove_code_fences(text: str) -> str:
        """Removes code fences from text"""
        return re.sub(r'~~~\w*\n|~~~', '', text)
