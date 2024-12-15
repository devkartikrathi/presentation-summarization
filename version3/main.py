import os
import streamlit as st
import base64
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
        self.output_path = os.path.join(os.getcwd(), "outputs")
        os.makedirs(self.output_path, exist_ok=True)

    def add_documents_to_retriever(self, processed_docs):
        if not processed_docs or not processed_docs['text_elements']:
            return False

        text_docs = processed_docs['text_elements']
        image_docs = processed_docs['image_elements']
        image_descriptions = processed_docs.get('image_descriptions', [])

        text_ids = [str(uuid.uuid4()) for _ in text_docs]
        image_ids = [str(uuid.uuid4()) for _ in image_docs]

        # Add text documents
        text_documents = [
            Document(page_content=doc, metadata={"doc_id": doc_id, "type": "text"}) 
            for doc, doc_id in zip(text_docs, text_ids)
        ]
        self.rag_model.retriever.vectorstore.add_documents(text_documents)
        self.rag_model.retriever.docstore.mset(list(zip(text_ids, text_docs)))

        # Add image documents
        if image_docs:
            image_documents = [
                Document(
                    page_content=desc or f"Image document {i+1}", 
                    metadata={"doc_id": doc_id, "type": "image"}
                ) 
                for i, (doc_id, desc) in enumerate(zip(image_ids, image_descriptions))
            ]
            self.rag_model.retriever.vectorstore.add_documents(image_documents)
            self.rag_model.retriever.docstore.mset(list(zip(image_ids, image_docs)))

        return True

    def get_multimodal_rag_response(self, query, processed_docs, selected_model_name):
        if not processed_docs or not processed_docs.get('text_elements'):
            return "No valid document content found.", []

        # Ensure documents are added to retriever
        self.add_documents_to_retriever(processed_docs)

        model_configs = self.model_provider.get_model_configs()
        model_config = model_configs.get(selected_model_name, 
            model_configs["Meta - Llama 3.2 3B Instruct Turbo"])

        llm = self.model_provider.create_model(model_config)

        # Template for generating detailed medical responses
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

        # Retrieve most relevant context
        retriever = self.rag_model.retriever
        retrieval_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Get text response
        response = retrieval_chain.invoke(query)

        # # Find relevant images
        # relevant_images = self.image_utils.find_most_relevant_images(
        #     response, 
        #     processed_docs.get('image_elements', [])
        # )

        return response #, relevant_images

    def generate_slides(self, processed_docs, num_slides):
        # Use more reliable model for slide generation
        model_configs = self.model_provider.get_model_configs()
        model_config = model_configs["Meta - Llama 3.3 70B Instruct Turbo"]  
        llm = self.model_provider.create_model(model_config)

        # Extract full context
        full_context = " ".join(processed_docs['text_elements'])

        # Enhanced slide generation prompt
        slide_generation_prompt = f"""Generate {num_slides} precise, knowledge-dense medical slides 
that capture the most critical information from the document:

DOCUMENT CONTEXT: {full_context}

ADVANCED SLIDE GENERATION REQUIREMENTS:
- Strict limit: 3 key points per slide
- Use exact, technical medical terminology
- Create a logical, sequential flow of information
- Eliminate redundancy
- Maximize knowledge density
- Focus on unique, high-impact insights
- Ensure clarity and immediate comprehensibility

OUTPUT FORMAT:
[Slide Title]
1. Primary Key Point
2. Supporting Technical Detail
3. Critical Clinical Insight"""

        # Generate slides with enhanced precision
        slides_content = llm.invoke(slide_generation_prompt).content

        # Robust slide parsing
        slides = []
        current_slide = {"content": "", "images": []}
        
        for line in slides_content.split('\n'):
            line = line.strip()
            
            # Detect new slide by title
            if line and not line.startswith('1.') and not line.startswith('2.') and not line.startswith('3.'):
                # Finalize previous slide if exists
                if current_slide["content"]:
                    slides.append(current_slide)
                
                # Start new slide
                current_slide = {"content": line + "\n", "images": []}
            
            # Add key points to current slide
            elif line.startswith(('1.', '2.', '3.')):
                current_slide["content"] += line + "\n"

        # Add final slide
        if current_slide["content"]:
            slides.append(current_slide)

        return slides

    def run(self):
        st.title("Clinical Document Assistant 🩺")

        # Initialize session state for chat and document
        if 'processed_docs' not in st.session_state:
            st.session_state.processed_docs = None
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Sidebar for document upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Medical Document", 
            type=['pdf', 'pptx']
        )

        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            # Process uploaded document
            try:
                st.session_state.processed_docs = self.doc_processor.process_document(
                    uploaded_file, file_type
                )
                st.sidebar.success("Document processed successfully!")

                # Create tabs
                tab1, tab2, tab3 = st.tabs([
                    "Chat Functionality", 
                    "Slides Content Generation", 
                    "Document Images"
                ])

                with tab1:
                    # Chat interface
                    st.header("Interactive Document Chat")
                    
                    # Display chat messages from history
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # Model selection
                    model_options = list(self.model_provider.get_model_configs().keys())
                    selected_model = st.selectbox(
                        "Select AI Model", 
                        model_options, 
                        index=model_options.index("Meta - Llama 3.2 3B Instruct Turbo")
                    )

                    # Chat input
                    if prompt := st.chat_input("Ask a question about the document"):
                        # Add user message to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            with st.spinner("Generating response..."):
                                if st.session_state.processed_docs:
                                    response = self.get_multimodal_rag_response(
                                        prompt, 
                                        st.session_state.processed_docs, 
                                        selected_model
                                    )
                                    
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
                            slides = self.generate_slides(st.session_state.processed_docs, num_slides)
                            
                            for i, slide in enumerate(slides, 1):
                                st.subheader(f"Slide {i}")
                                st.write(slide['content'])

                with tab3:
                    st.header("Document Images")
                    
                    if st.session_state.processed_docs:
                        image_paths = st.session_state.processed_docs.get('image_paths', [])
                        
                        st.write(f"Total Images: {len(image_paths)}")
                        
                        for i, img_path in enumerate(image_paths, 1):
                            st.subheader(f"Image {i}")
                            st.image(img_path, caption=f"Document Image {i}")

            except Exception as e:
                st.error(f"Error processing document: {e}")

def main():
    assistant = ClinicalDocumentAssistant()
    assistant.run()

if __name__ == "__main__":
    main()