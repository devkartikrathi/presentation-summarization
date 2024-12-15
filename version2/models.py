from langchain_together import TogetherEmbeddings, ChatTogether
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class ModelProvider:
    @staticmethod
    def get_model_configs():
        """Define available model configurations."""
        return {
            "Meta - Llama 3.3 70B Instruct Turbo": {
                "provider": "together",
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "description": "Advanced instruction-tuned model for large-scale tasks",
                "type": "text"
            },
            "Meta - Llama 3.2 3B Instruct Turbo": {
                "provider": "together",
                "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
                "description": "Compact instruction-tuned model for standard tasks",
                "type": "text"
            },
            "Qwen - QwQ-32B-Preview": {
                "provider": "together",
                "model": "Qwen/QwQ-32B-Preview",
                "description": "Preview version of Qwen's powerful 32B model",
                "type": "text"
            },
            "Ollama - Llama 3.2 1B": {
                "provider": "ollama",
                "model": "llama3:1b",
                "description": "Lightweight model for quick responses",
                "type": "text"
            },
            "Ollama - Llama 3.2 3B": {
                "provider": "ollama",
                "model": "llama3:3b",
                "description": "Balanced model for general tasks",
                "type": "text"
            }
        }

    @staticmethod
    def create_model(model_config):
        """Create a language model based on the configuration."""
        if model_config['provider'] == 'together':
            return TogetherEmbeddings(
                model=model_config['model']
            ) if model_config['type'] == 'embeddings' else ChatTogether(
                model=model_config['model']
            )
        elif model_config['provider'] == 'ollama':
            return ChatOllama(
                model=model_config['model']
            )
        else:
            raise ValueError(f"Unsupported model provider: {model_config['provider']}")

class RAGModel:
    def __init__(self):
        self.embeddings = TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-32k-retrieval"
        )
        self.vectorstore = Chroma(
            collection_name="clinical_knowledge", 
            embedding_function=self.embeddings
        )
        self.store = InMemoryStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore, 
            docstore=self.store, 
            id_key="doc_id",
            top_k=5
        )