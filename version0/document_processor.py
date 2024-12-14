import os
import base64
from together import Together
import config

class DocumentProcessor:
    def __init__(self, input_path, output_path):
        self.client = Together(api_key=os.getenv('TOGETHER_API_KEY'))
        self.input_path = input_path
        self.output_path = output_path
        self.vision_model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
        self.text_model = "meta-llama/Llama-3-70B-Instruct"

    def process_pdf(self, filepath):
        from unstructured.partition.pdf import partition_pdf
        
        raw_pdf_elements = partition_pdf(
            filename=filepath,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=self.output_path,
        )

        text_elements = []
        table_elements = []
        image_elements = []

        for element in raw_pdf_elements:
            if 'CompositeElement' in str(type(element)):
                text_elements.append(element.text)
            elif 'Table' in str(type(element)):
                table_elements.append(element.text)

        for image_file in os.listdir(self.output_path):
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.output_path, image_file)
                encoded_image = self.encode_image(image_path)
                image_elements.append(encoded_image)

        return text_elements, table_elements, image_elements

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_slides(self, content, num_slides=5):
        """
        Generate slide content using Together AI
        """
        prompt = f"""Create a comprehensive slide deck outline with {num_slides} slides 
        based on the following content. For each slide, provide:
        1. A clear and engaging title
        2. 3-5 key bullet points
        3. A concise description explaining the slide's main message and key insights

        Content to summarize:
        {content}

        Format your response as a structured slide deck outline."""

        response = self.client.chat.completions.create(
            model=self.text_model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert presentation creator who can transform complex content into clear, concise slide decks."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=2000
        )

        return response.choices[0].message.content

    def image_analysis(self, image_path):

        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Describe the contents of this image in detail."},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        return response.choices[0].message.content