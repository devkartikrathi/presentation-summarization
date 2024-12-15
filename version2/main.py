import os
import streamlit as st
from dotenv import load_dotenv
import uuid
from models import ModelProvider, RAGModel
from doc_processor import DocumentProcessor
from utils import AudioGenerator, ImageUtils
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class ClinicalDocumentAssistant:
    def __init__(self):
        load_dotenv()
        self.setup_streamlit()
        self.model_provider = ModelProvider()
        self.rag_model = RAGModel()
        self.doc_processor = DocumentProcessor(self.output_path)
        self.audio_generator = AudioGenerator()
        self.image_utils = ImageUtils()

    def setup_streamlit(self):
        st.set_page_config(page_title="Clinical Document Assistant", page_icon="🩺")
        self.output_path = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_path, exist_ok=True)

    def add_documents_to_retriever(self, processed_docs):
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
        self.rag_model.retriever.vectorstore.add_documents(text_documents)
        self.rag_model.retriever.docstore.mset(list(zip(text_ids, text_docs)))

        if image_docs:
            image_documents = [
                Document(page_content=f"Image document {i+1}", metadata={"doc_id": doc_id}) 
                for i, doc_id in enumerate(image_ids)
            ]
            self.rag_model.retriever.vectorstore.add_documents(image_documents)
            self.rag_model.retriever.docstore.mset(list(zip(image_ids, image_docs)))

        return True

    def get_rag_response(self, query, processed_docs, selected_model_name):
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
            {"context": self.rag_model.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(query) if processed_docs['text_elements'] else "No context available to generate response."

        relevant_images = self.image_utils.find_most_relevant_images(
            response, processed_docs.get('image_elements', [])
        )

        return response, relevant_images

    def generate_audio_summary(self, processed_docs):
        all_content = "\n".join(processed_docs['text_elements'])
        summary_prompt = """Create a concise summary of the document's key medical concepts."""

        model_config = self.model_provider.get_model_configs()["Meta - Llama 3.3 70B Instruct Turbo"]
        llm = self.model_provider.create_model(model_config)

        text_summary = llm.invoke(summary_prompt + "\n" + all_content).content
        audio_filename = self.audio_generator.generate_audio_summary(text_summary, self.output_path)

        return text_summary, audio_filename

    def main(self):
        st.title("Clinical Document Assistant 🩺")

        uploaded_file = st.file_uploader("Upload Clinical Document (PDF or PPTX)", type=['pdf', 'pptx'])
        
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            with st.spinner("Processing document..."):
                processed_docs = self.doc_processor.process_document(uploaded_file, file_type)

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
                    for img in processed_docs['image_elements'][:3]:
                        self.image_utils.decode_and_display_image(img)

                with tabs[2]:
                    st.markdown("### Generate Slides from Processed Content")
                    for text in processed_docs['text_elements'][:5]:
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