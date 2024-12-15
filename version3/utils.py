import os
import base64
import io
from gtts import gTTS
from PIL import Image
import streamlit as st
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from together import Together

client = Together()

class ImgSum:
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def summarize_image(encoded_image):
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the contents of this image."}, 
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}",
                            }
                        }
                    ]
                }
            ]
        )
        return response

class AudioGenerator:
    def generate_audio_summary(self, text: str, output_path: str, language='en') -> Optional[str]:
        try:
            os.makedirs(output_path, exist_ok=True)
            audio_filename = os.path.join(output_path, f"summary_audio_{hash(text)}.mp3")
            tts = gTTS(text=text, lang=language)
            tts.save(audio_filename)
            return audio_filename
        except Exception as e:
            st.error(f"Audio generation error: {e}")
            return None

class ImageUtils:
    def process_image(self, image_data: bytes, max_width: int = 800) -> Optional[Image.Image]:
        try:
            image = Image.open(io.BytesIO(image_data))
            if max_width and image.width > max_width:
                aspect_ratio = image.height / image.width
                new_height = int(max_width * aspect_ratio)
                image = image.resize((max_width, new_height), Image.LANCZOS)
            return image
        except Exception as e:
            st.error(f"Image processing error: {e}")
            return None

    def save_image(self, image: Image.Image, output_path: str, filename: str) -> Optional[str]:
        try:
            os.makedirs(output_path, exist_ok=True)
            image_path = os.path.join(output_path, filename)
            image.save(image_path)
            return image_path
        except Exception as e:
            st.error(f"Image saving error: {e}")
            return None

class SlideGenerator:
    def __init__(self, model_provider):
        self.model_provider = model_provider

    def generate_slides(self, text_elements: List[str], num_slides: int) -> List[Dict[str, str]]:
        try:
            model_config = self.model_provider.get_model_configs()["Meta - Llama 3.3 70B Instruct Turbo"]
            llm = self.model_provider.create_model(model_config)
            return self._generate_slide_content(text_elements, num_slides, llm)
        except Exception as e:
            st.error(f"Slide generation error: {e}")
            return []

    def _generate_slide_content(self, text_elements: List[str], num_slides: int, llm):
        full_text = " ".join(text_elements)
        template = """Generate {num_slides} medical presentation slides following this exact format:

Slide [number]: [Title]
   •    [Key Point 1]
   •    [Key Point 2]
   •    [Key Point 3]

DOCUMENT CONTENT: {text}

REQUIREMENTS:
1. Follow the exact format shown above
2. Use medical terminology accurately
3. Ensure logical flow between slides
4. Include only the most important information
5. Keep bullet points concise and clear"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"text": RunnablePassthrough(), "num_slides": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        slides_content = chain.invoke({"text": full_text, "num_slides": num_slides})
        return self._parse_slides(slides_content)

    def _parse_slides(self, content: str) -> List[Dict[str, str]]:
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

class SummaryProcessor:
    def __init__(self, model_provider, audio_generator):
        self.model_provider = model_provider
        self.audio_generator = audio_generator
        self.executor = ThreadPoolExecutor(max_workers=2)

    def generate_summary(self, text_elements: List[str], output_path: str) -> Tuple[str, str]:
        """Generate text summary and audio summary."""
        try:
            model_config = self.model_provider.get_model_configs()["Meta - Llama 3.3 70B Instruct Turbo"]
            llm = self.model_provider.create_model(model_config)

            text_summary = self._generate_text_summary(text_elements, llm)
            audio_path = self.audio_generator.generate_audio_summary(text_summary, output_path)

            return text_summary, audio_path
        except Exception as e:
            st.error(f"Summary generation error: {e}")
            return None, None

    def _generate_text_summary(self, text_elements: List[str], llm) -> str:
        """Generate text summary using the LLM."""
        full_text = " ".join(text_elements)
        
        template = """Generate a concise yet comprehensive medical document summary:

DOCUMENT CONTENT: {text}

REQUIREMENTS:
1. Create a structured, hierarchical summary
2. Capture all key medical points and findings
3. Maintain clinical accuracy and terminology
4. Be concise but thorough
5. Format with clear sections and bullet points

SUMMARY FORMAT:
Key Findings:
• [Main medical findings]

Core Concepts:
• [Essential medical concepts]

Clinical Implications:
• [Important clinical considerations]"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"text": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain.invoke(full_text)