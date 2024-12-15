import os
import uuid
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.pdf import partition_pdf
from typing import Optional, Dict, Any
import base64

class DocumentProcessor:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def process_document(self, uploaded_file: Any, file_type: str) -> Optional[Dict]:
        """Process uploaded document and extract content."""
        temp_path = self._save_temp_file(uploaded_file)
        if not temp_path:
            return None

        try:
            raw_elements = self._partition_document(temp_path, file_type)
            if not raw_elements:
                return None

            processed_content = self._extract_content(raw_elements)
            return self._create_document_metadata(processed_content)
        except Exception as e:
            raise Exception(f"Document processing error: {e}")
        finally:
            self._cleanup_temp_file(temp_path)

    def _save_temp_file(self, uploaded_file: Any) -> Optional[str]:
        """Save uploaded file to temporary location."""
        try:
            temp_path = os.path.join(self.output_path, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            return temp_path
        except Exception as e:
            print(f"Error saving temporary file: {e}")
            return None

    def _partition_document(self, file_path: str, file_type: str):
        """Partition document based on file type."""
        partition_params = {
            "filename": file_path,
            "extract_images_in_pdf": True,
            "infer_table_structure": True,
            "chunking_strategy": "by_title",
            "max_characters": 5000,
            "new_after_n_chars": 3800,
            "combine_text_under_n_chars": 2000,
            "image_output_dir_path": self.output_path,
        }

        if file_type == 'pdf':
            return partition_pdf(**partition_params)
        elif file_type == 'pptx':
            return partition_pptx(**partition_params)
        return None

    def _extract_content(self, raw_elements):
        """Extract text content from raw elements."""
        return [
            element.text for element in raw_elements 
            if hasattr(element, 'text') and element.text.strip()
        ]

    def _create_document_metadata(self, content):
        """Create metadata for processed document."""
        return {
            'text_elements': content,
            'doc_id': str(uuid.uuid4())
        }

    def _cleanup_temp_file(self, temp_path: str):
        """Clean up temporary file."""
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")