import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any
import os

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MilvusLiteManager:
    """
    A class to manage Milvus-lite database operations with German semantic embeddings
    """
    
    def __init__(self, db_path: str = "./milvus_lite.db", collection_name: str = "german_documents"):
        """
        Initialize the Milvus-lite manager
        
        Args:
            db_path (str): Path to the Milvus-lite database file
            collection_name (str): Name of the collection to create/use
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.collection = None
        self.embedding_model = None
        self.embedding_dim = None
        
    def load_embedding_model(self):
        """
        Load the German Semantic V3b model from Hugging Face
        This model is specifically trained for German text and provides high-quality embeddings
        """
        
        logger.info("Loading German Semantic V3b embedding model from Hugging Face...")
        self.embedding_model = SentenceTransformer('aari1995/German_Semantic_V3b')
            
        # Get embedding dimension by encoding a test sentence
        test_embedding = self.embedding_model.encode("Test")
        self.embedding_dim = len(test_embedding)
            
        logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        return True
        
    
    def connect_to_milvus(self):
        """
        Establish connection to Milvus-lite database
        Milvus-lite is a lightweight version that runs locally without external dependencies
        """
        try:
            logger.info(f"Connecting to Milvus-lite database at: {self.db_path}")
            
            # Connect to Milvus-lite (local file-based database)
            connections.connect(
                alias="default",
                uri=self.db_path  # Local file path for Milvus-lite
            )
            
            logger.info("Successfully connected to Milvus-lite")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus-lite: {e}")
            return False
    
    def create_collection(self):
        """
        Create a collection with schema optimized for German document storage
        
        Schema includes:
        - id: Primary key (auto-generated)
        - text: Original document text
        - embedding: Vector embedding of the text
        - metadata: Additional information about the document
        """
        try:
            # Check if collection already exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists. Dropping it...")
                utility.drop_collection(self.collection_name)
            
            # Define the schema for our collection
            fields = [
                FieldSchema(
                    name="id", 
                    dtype=DataType.INT64, 
                    is_primary=True, 
                    auto_id=True,
                    description="Unique document identifier"
                ),
                FieldSchema(
                    name="text", 
                    dtype=DataType.VARCHAR, 
                    max_length=10000,
                    description="Original German document text"
                ),
                FieldSchema(
                    name="embedding", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=self.embedding_dim,
                    description="German semantic embedding vector"
                ),
                FieldSchema(
                    name="metadata", 
                    dtype=DataType.VARCHAR, 
                    max_length=1000,
                    description="Additional metadata in JSON format"
                )
            ]
            
            # Create collection schema
            schema = CollectionSchema(
                fields=fields, 
                description="German documents with semantic embeddings"
            )
            
            # Create the collection
            self.collection = Collection(
                name=self.collection_name, 
                schema=schema,
                using='default'
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of German texts
        
        Args:
            texts (List[str]): List of German text documents
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            
            # Generate embeddings using the German model
            # The model handles German text preprocessing and tokenization automatically
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=32  # Process in batches for better performance
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def insert_documents(self, docs: List[str], metadata_list: List[str] = None):
        """
        Insert documents into the Milvus collection with their embeddings
        
        Args:
            docs (List[str]): List of German document texts
            metadata_list (List[str]): Optional metadata for each document
        """
        try:
            if not docs:
                logger.warning("No documents provided for insertion")
                return False
            
            # Generate default metadata if not provided
            if metadata_list is None:
                metadata_list = [f'{{"doc_index": {i}, "language": "de"}}' for i in range(len(docs))]
            
            # Generate embeddings for all documents
            embeddings = self.create_embeddings(docs)
            if embeddings is None:
                return False
            
            # Prepare data for insertion
            data = {
                "text": docs,
                "embedding": embeddings.tolist(),  # Convert to list for Milvus
                "metadata": metadata_list
            }
            
            logger.info(f"Inserting {len(docs)} documents into collection...")
            
            # Insert data into collection
            insert_result = self.collection.insert(data)
            
            # Flush to ensure data is written to disk
            self.collection.flush()
            
            logger.info(f"Successfully inserted {len(docs)} documents. IDs: {insert_result.primary_keys[:5]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return False
    
    def create_index(self):
        """
        Create an index for efficient similarity search
        Using IVF_FLAT index which is good for accuracy and works well with moderate dataset sizes
        """
        try:
            logger.info("Creating index for similarity search...")
            
            # Define index parameters
            # IVF_FLAT: Inverted File with exact distance calculation
            index_params = {
                "metric_type": "COSINE",  # Cosine similarity for semantic search
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}  # Number of cluster units
            }
            
            # Create index on the embedding field
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info("Index created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def load_collection(self):
        """
        Load the collection into memory for search operations
        """
        try:
            self.collection.load()
            logger.info("Collection loaded into memory for searching")
            return True
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            return False


def main():
    """
    Main function demonstrating the complete workflow
    """
    # Sample German documents to insert into the database
    docs = [
        "Die deutsche Sprache ist eine der wichtigsten Sprachen in Europa und wird von über 100 Millionen Menschen gesprochen.",
        "Künstliche Intelligenz revolutioniert die Art und Weise, wie wir mit Computern interagieren und Probleme lösen.",
        "Berlin ist die Hauptstadt Deutschlands und ein wichtiges kulturelles und politisches Zentrum.",
        "Machine Learning Algorithmen können große Datenmengen analysieren und Muster erkennen, die für Menschen schwer zu entdecken sind.",
        "Die deutsche Wirtschaft ist eine der stärksten in der Welt und basiert auf Innovation und Technologie.",
        "Natural Language Processing ermöglicht es Computern, menschliche Sprache zu verstehen und zu verarbeiten.",
        "München ist bekannt für das Oktoberfest und seine reiche bayerische Kultur und Tradition.",
        "Deep Learning Modelle verwenden neuronale Netzwerke, um komplexe Aufgaben wie Bildererkennung zu lösen.",
        "Die deutsche Literatur hat viele berühmte Autoren wie Goethe, Schiller und Thomas Mann hervorgebracht.",
        "Datenbanken sind essentiell für die Speicherung und Verwaltung großer Mengen strukturierter Informationen."
    ]
    
    # Optional metadata for each document
    metadata_list = [
        '{"category": "language", "topic": "german"}',
        '{"category": "technology", "topic": "AI"}',
        '{"category": "geography", "topic": "berlin"}',
        '{"category": "technology", "topic": "machine_learning"}',
        '{"category": "economy", "topic": "german_economy"}',
        '{"category": "technology", "topic": "NLP"}',
        '{"category": "culture", "topic": "munich"}',
        '{"category": "technology", "topic": "deep_learning"}',
        '{"category": "literature", "topic": "german_authors"}',
        '{"category": "technology", "topic": "databases"}'
    ]
    
    # Initialize the Milvus manager
    milvus_manager = MilvusLiteManager()
    
    # Step 1: Load the German embedding model
    if not milvus_manager.load_embedding_model():
        logger.error("Failed to load embedding model. Exiting...")
        return
    
    # Step 2: Connect to Milvus-lite
    if not milvus_manager.connect_to_milvus():
        logger.error("Failed to connect to Milvus. Exiting...")
        return
    
    # Step 3: Create collection with appropriate schema
    if not milvus_manager.create_collection():
        logger.error("Failed to create collection. Exiting...")
        return
    
    # Step 4: Insert documents with embeddings
    if not milvus_manager.insert_documents(docs, metadata_list):
        logger.error("Failed to insert documents. Exiting...")
        return
    
    # Step 5: Create index for efficient searching
    if not milvus_manager.create_index():
        logger.error("Failed to create index. Exiting...")
        return
    
    # Step 6: Load collection for search operations
    if not milvus_manager.load_collection():
        logger.error("Failed to load collection. Exiting...")
        return
    
    logger.info("✅ Milvus-lite setup completed successfully!")
    logger.info(f"📚 Inserted {len(docs)} German documents into the database")
    logger.info("🔍 Database is ready for similarity search operations")
    
    # Display collection statistics
    print(f"\n📊 Collection Statistics:")
    print(f"Collection name: {milvus_manager.collection_name}")
    print(f"Number of entities: {milvus_manager.collection.num_entities}")
    print(f"Embedding dimension: {milvus_manager.embedding_dim}")


if __name__ == "__main__":
    main()