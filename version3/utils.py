import os
import base64
import io
from gtts import gTTS
from PIL import Image
import streamlit as st
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class AudioGenerator:
    @staticmethod
    def generate_audio_summary(text: str, output_path: str, language='en', slow=False) -> str:
        try:
            os.makedirs(output_path, exist_ok=True)
            audio_filename = os.path.join(output_path, f"summary_audio_{hash(text)}.mp3")
            tts = gTTS(text=text, lang=language, slow=slow)
            tts.save(audio_filename)
            return audio_filename
        except Exception as e:
            st.error(f"Audio generation error: {e}")
            return None

class ImageUtils:
    @staticmethod
    def decode_base64_image(encoded_string: str, max_width: int = None) -> Image:
        try:
            image_data = base64.b64decode(encoded_string)
            image = Image.open(io.BytesIO(image_data))
            
            if max_width and image.width > max_width:
                aspect_ratio = image.height / image.width
                new_height = int(max_width * aspect_ratio)
                image = image.resize((max_width, new_height), Image.LANCZOS)
            
            return image
        except Exception as e:
            st.error(f"Image decoding error: {e}")
            return None

    @staticmethod
    def image_to_base64(image_path: str) -> str:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            st.error(f"Image to base64 conversion error: {e}")
            return None

class SlideGenerator:
    def __init__(self, model_provider):
        self.model_provider = model_provider

    def generate_slides(self, text_elements: List[str], num_slides: int) -> List[Dict[str, str]]:
        try:
            model_config = self.model_provider.get_model_configs()["Meta - Llama 3.3 70B Instruct Turbo"]
            llm = self.model_provider.create_model(model_config)

            full_text = " ".join(text_elements)
            
            template = """Generate {num_slides} medical presentation slides following this exact format:

DOCUMENT CONTENT: {text}

REQUIRED FORMAT FOR EACH SLIDE:
Slide [number]: [Title]
   •    [Key Point 1]
   •    [Key Point 2]
   •    [Key Point 3]

REQUIREMENTS:
1. Follow the exact format shown above
2. Use medical terminology accurately
3. Ensure logical flow between slides
4. Include only the most important information
5. Keep bullet points concise and clear

Generate the slides:"""

            prompt = ChatPromptTemplate.from_template(template)
            
            chain = (
                {"text": RunnablePassthrough(), "num_slides": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            slides_content = chain.invoke({"text": full_text, "num_slides": num_slides})
            return self._parse_slides(slides_content)

        except Exception as e:
            st.error(f"Error generating slides: {e}")
            return []

    def _parse_slides(self, slides_content: str) -> List[Dict[str, str]]:
        slides = []
        current_slide = {"content": ""}
        
        for line in slides_content.split('\n'):
            line = line.strip()
            if line.startswith('Slide'):
                if current_slide["content"]:
                    slides.append(current_slide)
                current_slide = {"content": line + "\n"}
            elif line.startswith('   •'):
                current_slide["content"] += line + "\n"
        
        if current_slide["content"]:
            slides.append(current_slide)
        
        return slides

class SummaryProcessor:
    def __init__(self, model_provider, audio_generator):
        self.model_provider = model_provider
        self.audio_generator = audio_generator
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def generate_summary_async(self, text_elements: List[str], output_path: str) -> Tuple[str, str]:
        model_config = self.model_provider.get_model_configs()["Meta - Llama 3.3 70B Instruct Turbo"]
        llm = self.model_provider.create_model(model_config)

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
• [Important clinical considerations]

Please provide the summary:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"text": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        text_summary = await chain.ainvoke(full_text)
        
        import asyncio
        loop = asyncio.get_event_loop()
        audio_path = await loop.run_in_executor(
            self.executor,
            self.audio_generator.generate_audio_summary,
            text_summary,
            output_path
        )

        return text_summary, audio_path