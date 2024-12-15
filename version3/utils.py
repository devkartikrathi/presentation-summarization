import os
import base64
import io
from gtts import gTTS
from PIL import Image
import streamlit as st

class AudioGenerator:
    @staticmethod
    def generate_audio_summary(text, output_path, language='en', slow=False):
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
    def decode_base64_image(encoded_string, max_width=None):
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
    def image_to_base64(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            st.error(f"Image to base64 conversion error: {e}")
            return None