from langchain_together import TogetherEmbeddings, ChatTogether
from langchain_chroma import Chroma
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryStore

class ModelProvider:
    @staticmethod
    def get_model_configs():
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
            }
        }

    @staticmethod
    def create_model(model_config):
        if model_config['provider'] == 'together':
            return TogetherEmbeddings(
                model=model_config['model']
            ) if model_config['type'] == 'embeddings' else ChatTogether(
                model=model_config['model']
            )
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
            top_k=7,
            search_kwargs={
                "filter": {"domain": "medical"}
            }
        )

    def add_documents(self, documents):
        self.vectorstore.add_documents(documents)
        doc_pairs = [(doc.metadata["doc_id"], doc.page_content) for doc in documents]
        self.store.mset(doc_pairs)