#We are using AstraDb and not chromadb because AstraDb provides a more scalable and managed solution for vector storage, which is essential for handling large datasets and ensuring high availability.
#Using Huggingface embeddings to convert text data into vector representations that can be efficiently stored and queried in AstraDb.
#The DataIngestor class is responsible for ingesting Flipkart review data into the AstraDb vector store.
# It initializes the embedding model and the vector store, and provides a method to ingest data either by loading existing data or converting new data using the DataConverter class.
# This setup allows for efficient storage and retrieval of Flipkart review data for use in LangChain applications.

from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from flipkart.data_converter import DataConverter
from flipkart.config import Config

# DataIngestor class to handle the ingestion of Flipkart review data into AstraDb vector store.
class DataIngestor:
    # Initializing the DataIngestor with embedding model and AstraDb vector store.
    # Setting up HuggingFace embeddings and AstraDB vector store with necessary configurations.
    # The collection name, API endpoint, token, and namespace are all sourced from the Config class.
    # Let's break down each parameter:
        
        # embedding=self.embedding
        # → Tell the storage: "Use THIS translator to understand reviews"
        
        # collection_name="flipkart_database"
        # → This is like naming your storage box "Flipkart Toys"
        # → All your product reviews go in this box
        
        # api_endpoint=Config.ASTRA_DB_API_ENDPOINT
        # → This is the "address" of your cloud storage
        # → Like: "123 Cloud Avenue, Internet City"
        
        # token=Config.ASTRA_DB_APPLICATION_TOKEN
        # → This is your SECRET PASSWORD to access storage
        # → Like the key to your locker
        
        # namespace=Config.ASTRA_DB_KEYSPACE
        # → This is like a "folder" inside your storage
        # → Like: "Put it in the 'Electronics' folder"

        # REAL-LIFE ANALOGY:
        # Imagine Google Drive for toys:
        # - You need the website address (api_endpoint)
        # - You need to login with password (token)
        # - You create a folder called "flipkart_database" (collection_name)
        # - Inside, you have sub-folders (namespace)
    
        
    def __init__(self):
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )
    # Ingest method to load existing data or convert and add new data to the vector store.
    def ingest(self,load_existing=True):
        if load_existing==True:
            return self.vstore
        # If not loading existing data, convert new data using DataConverter and add to vector store.
        docs = DataConverter("data/flipkart_product_review.csv").convert()
        # Adding the converted Document objects to the AstraDb vector store.
        self.vstore.add_documents(docs)

        return self.vstore
