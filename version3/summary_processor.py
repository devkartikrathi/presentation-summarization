import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class SummaryProcessor:
    def __init__(self, model_provider, audio_generator):
        self.model_provider = model_provider
        self.audio_generator = audio_generator
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def generate_summary_async(self, text_elements, output_path):
        """Generate text and audio summary asynchronously"""
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

        # Generate text summary
        text_summary = await chain.ainvoke(full_text)

        # Generate audio summary in a separate thread
        loop = asyncio.get_event_loop()
        audio_path = await loop.run_in_executor(
            self.executor,
            self.audio_generator.generate_audio_summary,
            text_summary,
            output_path
        )

        return text_summary, audio_path