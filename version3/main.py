import os
import streamlit as st
from models import ModelProvider, RAGModel
from doc_processor import DocumentProcessor
from utils import AudioGenerator, ImageUtils, SlideGenerator, SummaryProcessor
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class ClinicalDocumentAssistant:
    def __init__(self):
        self.setup_environment()
        self.initialize_components()

    def setup_environment(self):
        st.set_page_config(page_title="Clinical Document Assistant", page_icon="🩺")
        self.output_path = "outputs"
        os.makedirs(self.output_path, exist_ok=True)

    def initialize_components(self):
        self.model_provider = ModelProvider()
        self.rag_model = RAGModel()
        self.doc_processor = DocumentProcessor(self.output_path)
        self.audio_generator = AudioGenerator()
        self.image_utils = ImageUtils()
        self.slide_generator = SlideGenerator(self.model_provider)
        self.summary_processor = SummaryProcessor(self.model_provider, self.audio_generator)

    def run(self):
        st.title("Clinical Document Assistant 🩺")
        self.initialize_session_state()
        self.handle_file_upload()

    def initialize_session_state(self):
        if 'processed_docs' not in st.session_state:
            st.session_state.processed_docs = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def handle_file_upload(self):
        uploaded_file = st.sidebar.file_uploader(
            "Upload Medical Document", 
            type=['pdf', 'pptx']
        )

        if uploaded_file:
            try:
                self.process_uploaded_file(uploaded_file)
            except Exception as e:
                st.error(f"Error processing document: {e}")

    def process_uploaded_file(self, uploaded_file):
        file_type = uploaded_file.name.split('.')[-1].lower()
        st.session_state.processed_docs = self.doc_processor.process_document(
            uploaded_file, file_type
        )
        st.sidebar.success("Document processed successfully!")
        self.create_interface_tabs()

    def create_interface_tabs(self):
        tab1, tab2, tab3 = st.tabs(["Chat", "Slides", "Summary"])

        with tab1:
            self.render_chat_interface()

        with tab2:
            self.render_slides_interface()

        with tab3:
            self.render_summary_interface()

    def render_chat_interface(self):
        st.header("Chat with Document")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the document"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = self.get_rag_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    def render_slides_interface(self):
        st.header("Generate Slides")
        num_slides = st.slider("Number of Slides", 1, 10, 5)
        
        if st.button("Generate Slides"):
            with st.spinner("Generating slides..."):
                slides = self.slide_generator.generate_slides(
                    st.session_state.processed_docs['text_elements'],
                    num_slides
                )
                for slide in slides:
                    st.subheader(slide["title"])
                    for point in slide["points"]:
                        st.markdown(point)

    def render_summary_interface(self):
        st.header("Document Summary")
        
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary, audio_path = self.summary_processor.generate_summary(
                    st.session_state.processed_docs['text_elements'],
                    self.output_path
                )
                if summary:
                    st.markdown(summary)
                if audio_path:
                    st.audio(audio_path)

    def get_rag_response(self, query: str) -> str:
        try:
            model_config = self.model_provider.get_model_configs()["Meta - Llama 3.2 3B Instruct Turbo"]
            llm = self.model_provider.create_model(model_config)
            
            template = """Answer based on the following context:
{context}

Question: {question}

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = (
                {"context": self.rag_model.retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            return chain.invoke(query)
        except Exception as e:
            return f"Error generating response: {e}"

def main():
    assistant = ClinicalDocumentAssistant()
    assistant.run()

if __name__ == "__main__":
    main()