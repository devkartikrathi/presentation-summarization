import os
import base64
import uuid
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Tuple

# Third-party libraries
import pytesseract
from unstructured.partition.pdf import partition_pdf
from langchain_together import TogetherEmbeddings, ChatTogether
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

class ClinicalPDFAssistant:
    def __init__(self):
        # Setup configurations
        load_dotenv()
        self.setup_streamlit()
        self.setup_models()

    def setup_streamlit(self):
        st.set_page_config(page_title="Clinical PDF Assistant", page_icon="🩺")
        self.output_path = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_path, exist_ok=True)

    def setup_models(self):
        self.embeddings = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        )
        self.llm = ChatTogether(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", 
            temperature=0.1
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
            top_k=3
        )

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_document(self, uploaded_file):
        temp_path = os.path.join(self.output_path, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        raw_pdf_elements = partition_pdf(
            filename=temp_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=5000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=self.output_path,
        )

        text_elements = [
            element.text for element in raw_pdf_elements 
            if 'CompositeElement' in str(type(element))
        ]

        image_paths = [
            os.path.join(self.output_path, img) 
            for img in os.listdir(self.output_path) 
            if img.endswith(('.png', '.jpg', '.jpeg'))
        ]

        image_elements = [self.encode_image(path) for path in image_paths]

        return {
            'text_elements': text_elements,
            'image_elements': image_elements,
            'image_paths': image_paths
        }

    def add_documents_to_retriever(self, processed_docs: dict):
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

        image_documents = [
            Document(page_content=f"Image document {i+1}", metadata={"doc_id": doc_id}) 
            for i, doc_id in enumerate(image_ids)
        ]
        self.retriever.vectorstore.add_documents(image_documents)
        self.retriever.docstore.mset(list(zip(image_ids, image_docs)))

    def get_rag_response(self, query: str, processed_docs: dict) -> Tuple[str, List[str]]:
        self.add_documents_to_retriever(processed_docs)

        template = """Answer the clinical question based only on the following context:
        {context}
        Question: {question}
        
        Provide a precise, evidence-based medical response with key insights."""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)

        # relevant_images = processed_docs['image_paths'][:3]  # First 3 images as example

        return response # , relevant_images

    def generate_slides(self, processed_docs: dict) -> List[Dict]:
        all_content = "\n".join(processed_docs['text_elements'])
        
        slide_prompt = f"""Create a comprehensive yet concise 10-slide presentation 
        that captures the key medical knowledge from the following document. 
        Ensure each slide has a clear, descriptive title and key bullet points.

        Document Context:
        {all_content}"""

        slides_response = self.llm.invoke(slide_prompt).content

        slides = []
        for idx, text in enumerate(slides_response.split('\n\n'), 1):
            slide = {
                'number': idx,
                'content': text.strip(),
                'image': processed_docs['image_paths'][idx-1] 
                         if idx <= len(processed_docs['image_paths']) else None
            }
            slides.append(slide)

        return slides

    def generate_assessment(self, processed_docs: dict) -> Dict:
        all_content = "\n".join(processed_docs['text_elements'])
        
        assessment_prompt = f"""Create 10 multiple-choice medical knowledge questions 
        covering key concepts from this document."""

        assessment_response = self.llm.invoke(assessment_prompt + "\n" + all_content).content

        questions = []
        raw_questions = assessment_response.split('\n\n')
        
        for q in raw_questions[:10]:  # Limit to 10 questions
            if 'Question:' in q:
                question_parts = q.split('\n')
                questions.append({
                    'text': question_parts[0].replace('Question:', '').strip(),
                    'options': question_parts[1:5],
                    'correct_answer': question_parts[2] if len(question_parts) > 2 else ''
                })

        return {
            'title': 'Clinical Knowledge Assessment',
            'questions': questions
        }

    def generate_audio_summary(self, processed_docs: dict) -> str:
        summary_prompt = """Create a concise audio summary of the document's key medical concepts."""

        all_content = "\n".join(processed_docs['text_elements'])
        summary_response = self.llm.invoke(summary_prompt + "\n" + all_content).content

        audio_filename = os.path.join(
            self.output_path, 
            f"clinical_summary_{hash(summary_response)}.mp3"
        )

        tts = gTTS(text=summary_response, lang='en')
        tts.save(audio_filename)

        return audio_filename

    def main(self):
        st.title("Clinical PDF Assistant 🩺")
        
        # File Upload
        uploaded_file = st.file_uploader("Upload Clinical PDF", type=['pdf'])
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                processed_docs = self.process_document(uploaded_file)
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "Chat", "Knowledge Summary", "Assessment", "Audio Summary"
            ])
            
            with tab1:
                query = st.text_input("Ask a clinical question")
                if query:
                    response, relevant_images = self.get_rag_response(query, processed_docs)
                    st.write(response)
                    
                    if relevant_images:
                        st.image(relevant_images, width=200)
            
            with tab2:
                if st.button("Generate Summary Slides"):
                    slides = self.generate_slides(processed_docs)
                    for slide in slides:
                        st.markdown(f"### Slide {slide['number']}")
                        st.write(slide['content'])
                        if slide['image']:
                            st.image(slide['image'], width=300)
            
            with tab3:
                if st.button("Generate Assessment"):
                    assessment = self.generate_assessment(processed_docs)
                    for q in assessment['questions']:
                        st.write(q['text'])
                        st.radio("Select answer", q['options'])
            
            with tab4:
                if st.button("Generate Audio Summary"):
                    audio_file = self.generate_audio_summary(processed_docs)
                    st.audio(audio_file, format='audio/mp3')

def main():
    assistant = ClinicalPDFAssistant()
    assistant.main()

if __name__ == "__main__":
    main()
    
# ValueError: Expected Embedings to be non-empty list or numpy array, got [] in upsert.