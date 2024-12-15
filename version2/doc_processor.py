from unstructured.partition.pptx import partition_pptx
from unstructured.partition.pdf import partition_pdf
import uuid
from langchain.schema.document import Document
import base64
import os
from PIL import Image
import io
import imagehash

class DocumentProcessor:
    def __init__(self, output_path):
        self.output_path = output_path

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
                if 'CompositeElement' in str(type(element)) and element.text.strip()
            ]

            image_paths = [
                os.path.join(self.output_path, img) 
                for img in os.listdir(self.output_path) 
                if img.endswith(('.png', '.jpg', '.jpeg'))
            ]

            unique_images, duplicates = self.process_images(image_paths)
            image_elements = [self.encode_image(path) for path in unique_images]

            return {
                'text_elements': text_elements,
                'image_elements': image_elements,
                'image_paths': unique_images,
                'duplicates': duplicates
            }
        except Exception as e:
            raise Exception(f"Document processing error: {e}")

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def process_images(image_paths):
        image_info = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path)
                image_hash = str(imagehash.average_hash(image))
                image_info.append({'path': img_path, 'hash': image_hash})
            except Exception:
                continue

        unique_images = []
        unique_hashes = set()
        duplicates = []
        
        for info in image_info:
            if info['hash'] not in unique_hashes:
                unique_hashes.add(info['hash'])
                unique_images.append(info['path'])
            else:
                duplicates.append(info['path'])

        return unique_images, duplicates