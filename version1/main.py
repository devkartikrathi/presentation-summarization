import os
import base64
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Third-party libraries
import pytesseract
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.pdf import partition_pdf
from langchain_together import TogetherEmbeddings, ChatTogether
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from gtts import gTTS
from PIL import Image
import io
import imagehash
from collections import defaultdict

class ImageProcessor:
    @staticmethod
    def encode_image(image_path: str) -> str:
        """Encode an image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def decode_and_display_image(encoded_string: str):
        """Decode a base64 encoded image string and display it using PIL."""
        try:
            image_data = base64.b64decode(encoded_string)
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption="Relevant Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error decoding image: {e}")

    @staticmethod
    def find_most_relevant_images(response: str, image_documents: List[str], top_k: int = 3) -> List[str]:
        """Find most relevant images based on textual similarity to the response."""
        if not image_documents:
            return []

        def score_image(image_doc: str) -> float:
            keywords = response.lower().split()
            return sum(1 for keyword in keywords if keyword in image_doc.lower())

        scored_images = sorted(
            [(img, score_image(img)) for img in image_documents], 
            key=lambda x: x[1], 
            reverse=True
        )

        return [img for img, score in scored_images[:top_k]]

    @staticmethod
    def extract_duplicates_from_images(image_paths):
        """Detect duplicate images using perceptual hashing."""
        image_info = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path)
                image_hash = str(imagehash.average_hash(image))
                image_info.append({'path': img_path, 'hash': image_hash})
            except Exception as e:
                st.warning(f"Could not process image {img_path}: {e}")

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

class ModelProvider:
    @staticmethod
    def get_model_configs():
        """Define available model configurations."""
        return {
            "Meta - Llama 3.3 70B Instruct Turbo": {
                "provider": "together",
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "description": "Advanced instruction-tuned model for large-scale tasks",
                "type": "text"
            },
            "Meta - Llama 3.2 3B Instruct Turbo": {
                "provider": "together",
                "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                "description": "Compact instruction-tuned model for standard tasks",
                "type": "text"
            },
            "Qwen - QwQ-32B-Preview": {
                "provider": "together",
                "model": "Qwen/QwQ-32B-Preview",
                "description": "Preview version of Qwen’s powerful 32B model",
                "type": "text"
            },
            "Ollama - Llama 3.2 1B": {
                "provider": "ollama",
                "model": "llama3:1b",
                "description": "Lightweight model for quick responses",
                "type": "text"
            },
            "Ollama - Llama 3.2 3B": {
                "provider": "ollama",
                "model": "llama3:3b",
                "description": "Balanced model for general tasks",
                "type": "text"
            }
        }

    @staticmethod
    def create_model(model_config: Dict):
        """Create a language model based on the configuration."""
        if model_config['provider'] == 'together':
            return TogetherEmbeddings(
                model=model_config['model']
            ) if model_config['type'] == 'embeddings' else ChatTogether(
                model=model_config['model']
            )
        elif model_config['provider'] == 'ollama':
            return ChatOllama(
                model=model_config['model']
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_config['provider']}")

class ClinicalDocumentAssistant:
    def __init__(self):
        load_dotenv()
        self.setup_streamlit()
        self.setup_models()
        self.image_processor = ImageProcessor()
        self.model_provider = ModelProvider()

    def setup_streamlit(self):
        st.set_page_config(page_title="Clinical Document Assistant", page_icon="🩺")
        self.output_path = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_path, exist_ok=True)

    def setup_models(self):
        self.embeddings = TogetherEmbeddings(
                model="togethercomputer/m2-bert-80M-32k-retrieval"
            )

        self.vectorstore = Chroma(
            collection_name="clinical_knowledge", 
            embedding_function=self.embeddings
        )
        self.store = InMemoryStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore, 
            docstore=self.store, 
            id_key="doc_id",
            top_k=5
        )

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
                st.error("Unsupported file type")
                return None
        except Exception as e:
            st.error(f"Document processing error: {e}")
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

        unique_image_paths, duplicates = self.image_processor.extract_duplicates_from_images(image_paths)
        image_elements = [self.image_processor.encode_image(path) for path in unique_image_paths]

        return {
            'text_elements': text_elements,
            'image_elements': image_elements,
            'image_paths': unique_image_paths,
            'duplicates': duplicates
        }

    def add_documents_to_retriever(self, processed_docs: dict):
        if not processed_docs or not processed_docs['text_elements']:
            return False

        text_docs = processed_docs['text_elements']
        image_docs = processed_docs['image_elements']

        text_ids = [str(uuid.uuid4()) for _ in text_docs]
        image_ids = [str(uuid.uuid4()) for _ in image_docs]

        text_documents = [
            Document(page_content=doc, metadata={"doc_id": doc_id}) 
            for doc, doc_id in zip(text_docs, text_ids)
        ]
        self.retriever.vectorstore.add_documents(text_documents)
        self.retriever.docstore.mset(list(zip(text_ids, text_docs)))

        if image_docs:
            image_documents = [
                Document(page_content=f"Image document {i+1}", metadata={"doc_id": doc_id}) 
                for i, doc_id in enumerate(image_ids)
            ]
            self.retriever.vectorstore.add_documents(image_documents)
            self.retriever.docstore.mset(list(zip(image_ids, image_docs)))

        return True

    def get_rag_response(self, query: str, processed_docs: dict, selected_model_name: str) -> Tuple[str, List[str]]:
        if not processed_docs or not processed_docs.get('text_elements'):
            return "No valid document content found.", []

        self.add_documents_to_retriever(processed_docs)

        model_configs = self.model_provider.get_model_configs()
        model_config = model_configs.get(selected_model_name, 
            model_configs["Meta - Llama 3.2 3B Instruct Turbo"])

        llm = self.model_provider.create_model(model_config)

        template = """Medical Knowledge Assistant Protocol:

CONTEXT: {context}

QUERY: {question}

RESPONSE GUIDELINES:
1. Provide a precise, evidence-based medical response
2. Prioritize clinical actionability
3. Use clear, professional medical terminology
4. Structure your answer for immediate practical application

Deliver a comprehensive yet concise response."""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(query) if processed_docs['text_elements'] else "No context available to generate response."

        image_documents = processed_docs.get('image_elements', [])
        relevant_images = self.image_processor.find_most_relevant_images(
            response, image_documents
        )

        return response, relevant_images

    def generate_audio_summary(self, processed_docs: dict) -> Tuple[str, str]:
        """Generate an audio and text summary for the document."""
        all_content = "\n".join(processed_docs['text_elements'])
        summary_prompt = """Create a concise summary of the document's key medical concepts."""

        model_config = self.model_provider.get_model_configs()["Meta - Llama 3.3 70B Instruct Turbo"]
        llm = self.model_provider.create_model(model_config)

        text_summary = llm.invoke(summary_prompt + "\n" + all_content).content

        audio_filename = os.path.join(self.output_path, "summary_audio.mp3")
        tts = gTTS(text=text_summary, lang='en')
        tts.save(audio_filename)

        return text_summary, audio_filename

    def main(self):
        st.title("Clinical Document Assistant 🩺")

        uploaded_file = st.file_uploader("Upload Clinical Document (PDF or PPTX)", type=['pdf', 'pptx'])
        
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            with st.spinner("Processing document..."):
                processed_docs = self.process_document(uploaded_file, file_type)

            if processed_docs:
                st.markdown("### Preloading Tabs for Efficient Access")

                tabs = st.tabs(["Chat", "Relevant Images", "Slides", "Audio Summary"])

                with tabs[0]:
                    query = st.text_input("Ask a clinical question")
                    if query:
                        selected_model_name = st.selectbox("Select Model", list(self.model_provider.get_model_configs().keys()))
                        response, relevant_images = self.get_rag_response(query, processed_docs, selected_model_name)
                        st.write(response)

                with tabs[1]:
                    st.markdown("### Images Relevant to the Response")
                    for img in processed_docs['image_elements'][:3]:  # Displaying first 3 as example
                        self.image_processor.decode_and_display_image(img)

                with tabs[2]:
                    st.markdown("### Generate Slides from Processed Content")
                    for text in processed_docs['text_elements'][:5]:  # Show example elements
                        st.write(text)

                with tabs[3]:
                    st.markdown("### Generate Audio and Text Summary")
                    if st.button("Generate Summary"):
                        text_summary, audio_summary = self.generate_audio_summary(processed_docs)
                        st.write("**Text Summary:**")
                        st.text_area("Summary", text_summary, height=200)
                        st.audio(audio_summary, format="audio/mp3")

if __name__ == "__main__":
    assistant = ClinicalDocumentAssistant()
    assistant.main()
