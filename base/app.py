import streamlit as st
import os
import google.generativeai as genai
from document_processor import DocumentProcessor
from document_update_checker import DocumentUpdateChecker
import config
from together import Together

# genai.configure(api_key=config.GEMINI_API_KEY)
# model = genai.GenerativeModel('gemini-pro')

client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
model = client.chat.completions

def main():
    st.title("PDF Intelligence & Summarization Tool")

    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        file_path = os.path.join(config.DATA_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"File {uploaded_file.name} uploaded successfully!")

    query = st.text_input("Ask a question about your documents:")
    slides_number = st.slider("Number of slides to generate", min_value=3, max_value=10, value=5)

    if st.button("Process & Summarize"):
        with st.spinner("Processing documents..."):
            doc_checker = DocumentUpdateChecker()
            updated_files = doc_checker.check_and_update_documents()
            
            st.write("Updated files:", updated_files)
            doc_processor = DocumentProcessor(config.INPUT_PATH, config.OUTPUT_PATH)
            
            if updated_files:
                all_text = ""
                for file in updated_files:
                    file_path = os.path.join(config.DATA_FOLDER, file)
                    elements, _, _ = doc_processor.process_pdf(file_path)
                    all_text += " ".join(elements)

                slides_content = doc_processor.generate_slides(all_text, slides_number)
                
                st.subheader("Generated Slide Deck")
                st.markdown(slides_content)

    if query:
        with st.spinner("Generating response..."):
            response = client.chat.completions.create(
                model="meta-llama/Llama-3-70B-Instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": query}
                ]
            )
            st.write("Response:", response.choices[0].message.content)

if __name__ == "__main__":
    main()