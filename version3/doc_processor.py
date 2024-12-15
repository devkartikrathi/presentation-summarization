import os
import uuid
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.pdf import partition_pdf
from typing import Optional, Dict, Any

class DocumentProcessor:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def process_document(self, uploaded_file: Any, file_type: str) -> Optional[Dict]:
        temp_path = os.path.join(self.output_path, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            raw_elements = self._partition_document(temp_path, file_type)
            if not raw_elements:
                return None

            text_elements = [
                element.text for element in raw_elements 
                if hasattr(element, 'text') and element.text.strip()
            ]

            return {
                'text_elements': text_elements,
                'doc_id': str(uuid.uuid4())
            }

        except Exception as e:
            raise Exception(f"Document processing error: {e}")

    def _partition_document(self, file_path: str, file_type: str):
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