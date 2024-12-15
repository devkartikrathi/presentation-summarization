from gtts import gTTS
import os
from PIL import Image
import io
import base64
import streamlit as st

class AudioGenerator:
    @staticmethod
    def generate_audio_summary(text, output_path):
        """Generate audio file from text."""
        audio_filename = os.path.join(output_path, "summary_audio.mp3")
        tts = gTTS(text=text, lang='en')
        tts.save(audio_filename)
        return audio_filename

class ImageUtils:
    @staticmethod
    def decode_and_display_image(encoded_string):
        """Decode and display base64 encoded image."""
        try:
            image_data = base64.b64decode(encoded_string)
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption="Relevant Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error decoding image: {e}")

    @staticmethod
    def find_most_relevant_images(response, image_documents, top_k=3):
        """Find most relevant images based on textual similarity."""
        if not image_documents:
            return []

        def score_image(image_doc):
            keywords = response.lower().split()
            return sum(1 for keyword in keywords if keyword in image_doc.lower())

        scored_images = sorted(
            [(img, score_image(img)) for img in image_documents], 
            key=lambda x: x[1], 
            reverse=True
        )

        return [img for img, score in scored_images[:top_k]]