import os
import base64
import io
import streamlit as st
import torch
import numpy as np

from gtts import gTTS
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class AudioGenerator:
    @staticmethod
    def generate_audio_summary(text, output_path, language='en', slow=False):
        """
        Generate an audio file from text with configurable options.
        
        Args:
            text (str): Text to convert to speech
            output_path (str): Directory to save the audio file
            language (str, optional): Language code. Defaults to 'en'
            slow (bool, optional): Speak slowly. Defaults to False
        
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)
            
            # Generate unique filename
            audio_filename = os.path.join(output_path, f"summary_audio_{hash(text)}.mp3")
            
            # Generate text-to-speech
            tts = gTTS(text=text, lang=language, slow=slow)
            tts.save(audio_filename)
            
            return audio_filename
        except Exception as e:
            st.error(f"Audio generation error: {e}")
            return None

class ImageUtils:
    @staticmethod
    def decode_base64_image(encoded_string, max_width=None):
        """
        Decode a base64 encoded image with optional resizing.
        
        Args:
            encoded_string (str): Base64 encoded image
            max_width (int, optional): Maximum width for image
        
        Returns:
            PIL.Image.Image: Processed image or None
        """
        try:
            # Decode base64 string
            image_data = base64.b64decode(encoded_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Resize if needed
            if max_width and image.width > max_width:
                aspect_ratio = image.height / image.width
                new_height = int(max_width * aspect_ratio)
                image = image.resize((max_width, new_height), Image.LANCZOS)
            
            return image
        except Exception as e:
            st.error(f"Image decoding error: {e}")
            return None

    @staticmethod
    def display_base64_image(encoded_string, max_width=600, title=None):
        """
        Display base64 encoded image in Streamlit.
        
        Args:
            encoded_string (str): Base64 encoded image
            max_width (int, optional): Maximum display width
            title (str, optional): Image caption
        """
        image = ImageUtils.decode_base64_image(encoded_string, max_width)
        if image:
            st.image(image, caption=title, use_column_width='auto')

    @staticmethod
    def image_to_base64(image_path):
        """
        Convert an image file to base64 encoded string.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            st.error(f"Image to base64 conversion error: {e}")
            return None

    @staticmethod
    def find_most_relevant_images(response, image_documents, top_k=3):
        """
        Find most relevant images using CLIP model for semantic similarity.
        
        Args:
            response (str): Text to compare against images
            image_documents (list): List of base64 encoded images
            top_k (int, optional): Number of top images to return
        
        Returns:
            list: Most relevant image documents
        """
        if not image_documents:
            return []

        try:
            # Load CLIP model
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # Process text
            text_inputs = clip_processor(text=response, return_tensors="pt", padding=True)
            text_embeddings = clip_model.get_text_features(**text_inputs)

            def score_image(image_doc):
                """Calculate similarity between text and image"""
                try:
                    # Decode and process image
                    image_data = base64.b64decode(image_doc)
                    image = Image.open(io.BytesIO(image_data))
                    image_inputs = clip_processor(images=image, return_tensors="pt")
                    image_embeddings = clip_model.get_image_features(**image_inputs)
                    
                    # Calculate cosine similarity
                    similarity = torch.nn.functional.cosine_similarity(
                        text_embeddings, image_embeddings
                    ).item()
                    return similarity
                except Exception:
                    return 0

            # Score and sort images
            scored_images = sorted(
                [(img, score_image(img)) for img in image_documents], 
                key=lambda x: x[1], 
                reverse=True
            )

            return [img for img, score in scored_images[:top_k]]
        except Exception as e:
            st.error(f"Image relevance error: {e}")
            return image_documents[:top_k]

def get_images_from_context(context_list, max_width=600, display_method='streamlit'):
    """
    Display images from a list of base64 encoded context strings.
    
    Args:
        context_list (list): List of base64 encoded image strings
        max_width (int, optional): Maximum image width
        display_method (str, optional): Display method ('streamlit', 'print', 'return')
    
    Returns:
        list or None: List of PIL images if display_method is 'return'
    """
    if not context_list:
        return [] if display_method == 'return' else None

    processed_images = []
    for i, context in enumerate(context_list, 1):
        try:
            image = ImageUtils.decode_base64_image(context, max_width)
            
            if image:
                if display_method == 'streamlit':
                    st.subheader(f"Image {i}")
                    st.image(image, caption=f"Context Image {i}", use_column_width='auto')
                elif display_method == 'print':
                    print(f"Image {i}")
                    image.show()
                elif display_method == 'return':
                    processed_images.append(image)
        except Exception as e:
            st.error(f"Error processing image {i}: {e}")

    return processed_images if display_method == 'return' else None

def generate_placeholder_image(width=400, height=300, text="Placeholder"):
    """
    Generate a simple placeholder image.
    
    Args:
        width (int, optional): Image width
        height (int, optional): Image height
        text (str, optional): Text to display on image
    
    Returns:
        PIL.Image.Image: Generated placeholder image
    """
    try:
        # Create a new image with a white background
        image = Image.new('RGB', (width, height), color='white')
        
        # Use PIL's ImageDraw to add text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        # Try to use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Calculate text position
        text_width, text_height = draw.textsize(text, font=font)
        position = ((width-text_width)/2, (height-text_height)/2)
        
        # Draw text and border
        draw.rectangle([0, 0, width-1, height-1], outline='gray')
        draw.text(position, text, fill='gray', font=font)
        
        return image
    except Exception as e:
        st.error(f"Placeholder image generation error: {e}")
        return None