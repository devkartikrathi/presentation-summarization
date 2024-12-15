import streamlit as st
from models import ModelProvider, RAGModel
from doc_processor import DocumentProcessor
from utils import AudioGenerator, ImageUtils
import asyncio
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
        self.setup_streamlit()
        self.model_provider = ModelProvider()
        self.rag_model = RAGModel()
        self.doc_processor = DocumentProcessor(self.output_path)
        self.audio_generator = AudioGenerator()
        self.image_utils = ImageUtils()

    def setup_streamlit(self):
        st.set_page_config(page_title="Clinical Document Assistant", page_icon="🩺")
        self.output_path = "outputs"
        os.makedirs(self.output_path, exist_ok=True)

    def add_documents_to_retriever(self, processed_docs):
        if not processed_docs or not processed_docs['text_elements']:
            return False

        text_docs = processed_docs['text_elements']
        text_ids = [str(uuid.uuid4()) for _ in text_docs]

        text_documents = [
            Document(page_content=doc, metadata={"doc_id": doc_id, "type": "text"}) 
            for doc, doc_id in zip(text_docs, text_ids)
        ]
        
        self.rag_model.add_documents(text_documents)
        return True

    async def get_multimodal_rag_response(self, query, processed_docs, selected_model_name):
        if not processed_docs or not processed_docs.get('text_elements'):
            return "No valid document content found."

        await self.add_documents_to_retriever(processed_docs)

        model_configs = self.model_provider.get_model_configs()
        model_config = model_configs.get(selected_model_name, 
            model_configs["Meta - Llama 3.2 3B Instruct Turbo"])

        llm = self.model_provider.create_model(model_config)
        
        template = """Advanced Medical Knowledge Assistant:

COMPREHENSIVE CONTEXT: {context}

SPECIFIC QUERY: {question}

RESPONSE PROTOCOL:
1. Provide precise, evidence-based medical insights
2. Integrate contextual information from document
3. Use advanced clinical terminology
4. Ensure actionable, structured response

Detailed Medical Interpretation:"""

        prompt = ChatPromptTemplate.from_template(template)
        retriever = self.rag_model.retriever
        
        retrieval_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return await retrieval_chain.ainvoke(query)

    def generate_slides(self, processed_docs, num_slides):
        model_config = self.model_provider.get_model_configs()["Meta - Llama 3.3 70B Instruct Turbo"]
        llm = self.model_provider.create_model(model_config)

        full_context = " ".join(processed_docs['text_elements'])

        slide_template = """
Slide {slide_num}: {title}
   •    {point1}
   •    {point2}
   •    {point3}
"""

        slide_generation_prompt = f"""Generate {num_slides} precise medical slides following this exact format for each slide:

{slide_template}

Use this context: {full_context}

Requirements:
- Each slide must have exactly 3 bullet points
- Use medical terminology accurately
- Ensure logical flow between slides
- Focus on key clinical insights
- Be concise but comprehensive"""

        slides_content = llm.invoke(slide_generation_prompt).content
        return self._parse_slides(slides_content)

    def _parse_slides(self, content):
        slides = []
        current_slide = None
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Slide'):
                if current_slide:
                    slides.append(current_slide)
                current_slide = {"title": line, "points": []}
            elif line.startswith('•') and current_slide:
                current_slide["points"].append(line)
        
        if current_slide:
            slides.append(current_slide)
            
        return slides

    def run(self):
        st.title("Clinical Document Assistant 🩺")

        if 'processed_docs' not in st.session_state:
            st.session_state.processed_docs = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        uploaded_file = st.sidebar.file_uploader(
            "Upload Medical Document", 
            type=['pdf', 'pptx']
        )

        if uploaded_file:
            try:
                file_type = uploaded_file.name.split('.')[-1].lower()
                st.session_state.processed_docs = self.doc_processor.process_document(
                    uploaded_file, file_type
                )
                st.sidebar.success("Document processed successfully!")

                tab1, tab2 = st.tabs(["Chat Functionality", "Slides Content Generation"])

                with tab1:
                    st.header("Interactive Document Chat")
                    
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    model_options = list(self.model_provider.get_model_configs().keys())
                    selected_model = st.selectbox(
                        "Select AI Model", 
                        model_options, 
                        index=model_options.index("Meta - Llama 3.2 3B Instruct Turbo")
                    )

                    if prompt := st.chat_input("Ask a question about the document"):
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            with st.spinner("Generating response..."):
                                if st.session_state.processed_docs:
                                    response = asyncio.run(self.get_multimodal_rag_response(
                                        prompt, 
                                        st.session_state.processed_docs, 
                                        selected_model
                                    ))
                                    
                                    st.markdown(response)
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": response
                                    })

                with tab2:
                    st.header("Slide Content Generation")
                    num_slides = st.slider(
                        "Number of Slides", 
                        min_value=1, max_value=10, value=5
                    )
                    
                    if st.button("Generate Slides") and st.session_state.processed_docs:
                        with st.spinner("Generating slides..."):
                            slides = self.generate_slides(
                                st.session_state.processed_docs, 
                                num_slides
                            )
                            
                            for slide in slides:
                                st.subheader(slide["title"])
                                for point in slide["points"]:
                                    st.markdown(point)

            except Exception as e:
                st.error(f"Error processing document: {e}")

def main():
    assistant = ClinicalDocumentAssistant()
    assistant.run()

if __name__ == "__main__":
    main()