import os
import uuid
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.pdf import partition_pdf
from langchain.schema.document import Document

class DocumentProcessor:
    def __init__(self, output_path):
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def process_document(self, uploaded_file, file_type):
        temp_path = os.path.join(self.output_path, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            if file_type == 'pdf':
                raw_elements = partition_pdf(
                    filename=temp_path,
                    extract_images_in_pdf=True,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    max_characters=5000,
                    new_after_n_chars=3800,
                    combine_text_under_n_chars=2000,
                    image_output_dir_path=self.output_path,
                )
            elif file_type == 'pptx':
                raw_elements = partition_pptx(
                    filename=temp_path,
                    extract_images_in_pptx=True,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    max_characters=5000,
                    new_after_n_chars=3800,
                    combine_text_under_n_chars=2000,
                    image_output_dir_path=self.output_path,
                )
            else:
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

    def cleanup(self):
        """Clean up temporary files"""
        for file in os.listdir(self.output_path):
            file_path = os.path.join(self.output_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}")