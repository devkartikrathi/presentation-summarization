import os
import json
import hashlib
import google.generativeai as genai
import config
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
import base64

class DocumentUpdateChecker:
    def __init__(self, data_folder=config.DATA_FOLDER, hash_file='document_hashes.json'):
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.data_folder = data_folder
        self.hash_file = os.path.join(data_folder, hash_file)
        self.vector_store = VectorStoreManager()
        self.document_processor = DocumentProcessor(
            input_path=self.data_folder, 
            output_path=config.OUTPUT_PATH
        )

    def _compute_file_hash(self, filepath):
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def check_and_update_documents(self):
        current_hashes = {}
        updated_files = []

        for filename in os.listdir(self.data_folder):
            if filename.endswith('.pdf'):
                filepath = os.path.join(self.data_folder, filename)
                current_hash = self._compute_file_hash(filepath)
                current_hashes[filename] = current_hash

                text_elements, table_elements, image_elements = self.document_processor.process_pdf(filepath)
                
                text_summaries = []
                for text in text_elements:
                    summary_prompt = f"Summarize the following text concisely:\n\n{text}"
                    summary = self.model.generate_content(summary_prompt).text
                    text_summaries.append(summary)

                table_summaries = []
                for table in table_elements:
                    summary_prompt = f"Summarize the following table:\n\n{table}"
                    summary = self.model.generate_content(summary_prompt).text
                    table_summaries.append(summary)

                image_summaries = []
                for image in image_elements:
                    image_path = os.path.join(config.OUTPUT_PATH, f"image_{len(image_summaries)}.jpg")
                    with open(image_path, 'wb') as f:
                        f.write(base64.b64decode(image))
                    summary = self.document_processor.image_analysis(image_path)
                    image_summaries.append(summary)

                self.vector_store.add_documents(text_summaries, text_elements)
                self.vector_store.add_documents(table_summaries, table_elements)
                self.vector_store.add_documents(image_summaries, image_elements)

                updated_files.append(filename)

        return updated_files