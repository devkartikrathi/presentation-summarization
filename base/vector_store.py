import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
import config

class VectorStoreManager:
    def __init__(self, collection_name="summaries"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GEMINI_API_KEY
        )
        self.vectorstore = Chroma(
            collection_name=collection_name, 
            embedding_function=self.embeddings
        )
        self.store = InMemoryStore()
        self.id_key = "doc_id"
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore, 
            docstore=self.store, 
            id_key=self.id_key,
            top_k=1
        )

    def add_documents(self, summaries, original_contents):
        doc_ids = [str(uuid.uuid4()) for _ in summaries]
        summary_docs = [
            Document(page_content=s, metadata={self.id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, original_contents)))

    def get_retriever(self):
        return self.retriever