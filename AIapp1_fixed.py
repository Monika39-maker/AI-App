# AIapp1.py - Smart RAG System for Insurance Documents

from langchain_community.document_loaders import DirectoryLoader
from sentence_transformers.util import similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import uuid
from typing import List, Dict, Any
import os
import pickle
from langchain_core.documents import Document

# Simple text splitter function
def simple_text_splitter(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap if end < text_len else text_len
    
    return chunks

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for processing.
    
    Args:
        documents: List of Document objects
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked Document objects
    """
    split_docs = []
    
    for doc in documents:
        # Split the page content
        chunks = simple_text_splitter(doc.page_content, chunk_size, chunk_overlap)
        
        # Create new Document objects for each chunk
        for i, chunk in enumerate(chunks):
            # Copy metadata and add chunk info
            chunk_metadata = doc.metadata.copy()
            chunk_metadata['chunk_id'] = i
            chunk_metadata['chunk_count'] = len(chunks)
            
            chunk_doc = Document(
                page_content=chunk,
                metadata=chunk_metadata
            )
            split_docs.append(chunk_doc)
    
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show what a chunk looks like
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding manager
        Args:
            model_name (str): The name of the SentenceTransformer model to use.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            print(f"Loaded model: {self.model.get_sentence_embedding_dimension()} dimensions")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        Args:
            texts (List[str]): List of texts to embed.
        Returns:
            np.ndarray: Array of embeddings.
        """
        if not self.model:
            raise ValueError("Model not loaded.")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.
        Returns:
            int: Dimension of the embeddings.
        """
        if not self.model:
            raise ValueError("Model not loaded.")
        return self.model.get_sentence_embedding_dimension()

# initialize the embedding manager
embedding_manager = EmbeddingManager()

class VectorStore:
    """Manages document embeddings using FAISS"""
    def __init__(self, index_file: str = "AIdata/vector_store.faiss", 
                 data_file: str = "AIdata/vector_store_data.pkl"):
        """Initialize the vector store
        Args:
            index_file (str): Path to save/load FAISS index.
            data_file (str): Path to save/load document metadata.
        """
        self.index_file = index_file
        self.data_file = data_file
        self.index = None
        self.documents = []
        self.metadata = []
        self._initialize_store()

    def _initialize_store(self):
        """Initialize or load the FAISS index."""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.data_file):
                # Load existing index and data
                self.index = faiss.read_index(self.index_file)
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']
                print(f"Loaded existing FAISS index with {self.index.ntotal} documents")
            else:
                # Create new index
                embedding_dim = embedding_manager.get_embedding_dimension()
                self.index = faiss.IndexFlatL2(embedding_dim)
                print(f"Created new FAISS index with dimension {embedding_dim}")
        except Exception as e:
            print(f"Error initializing FAISS: {e}")
            raise
    
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """Add documents and their embeddings to the collection.
        Args:
            documents: List of document texts.
            embeddings (np.ndarray): Corresponding embeddings for the document.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        for doc in documents:
            self.documents.append(doc.page_content)
            self.metadata.append(doc.metadata)
        
        # Save to disk
        self._save_store()
        
        print(f"Successfully added {len(documents)} documents.")
        print(f"Total documents in store: {self.index.ntotal}")
    
    def _save_store(self):
        """Save the FAISS index and document data to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_file)
            
            # Save documents and metadata
            with open(self.data_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
                
            print(f"Vector store saved to {self.index_file}")
        except Exception as e:
            print(f"Error saving vector store: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Search for similar documents.
        Args:
            query_embedding (np.ndarray): Embedding of the query.
            k (int): Number of results to return.
        Returns:
            List of tuples: (document, metadata, distance)
        """
        if self.index.ntotal == 0:
            return []
        
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    self.metadata[idx],
                    float(dist)
                ))
        
        return results

# Load documents
dir_loader = DirectoryLoader(
   "c:/Users/monik/Downloads/AI-App/AIdata",
   glob="**/*.pdf",
   show_progress=True
)

documents = dir_loader.load()
print(f"Loaded {len(documents)} documents")

# initialize the vector store
vector_store = VectorStore()
print("Vector store initialized.")

# Split documents into chunks
chunks = split_documents(documents)

# Generate embeddings for chunks
embeddings = embedding_manager.generate_embeddings([chunk.page_content for chunk in chunks])

# Add documents to vector store
vector_store.add_documents(chunks, embeddings)

class RAGretrieve:
    """Retrieve relevant documents for a query using FAISS"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """Initialize the RAG retriever
        Args:
            vector_store: The FAISS vector store
            embedding_manager: The embedding manager
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query (str): The query to search for.
            k (int): Number of results to return.
        Returns:
            List of dictionaries containing retrieved documents, metadata, and distance.
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.search(query_embedding, k)
            
            # Process results
            retrieved_docs = []
            
            for doc_content, metadata, distance in results:
                retrieved_docs.append({
                    'content': doc_content,
                    'metadata': metadata,
                    'distance': distance
                })
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

rag_retriever = RAGretrieve(vector_store, embedding_manager)

print("RAG retriever initialized.")

# Initialize LLM first
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in .env file.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7, max_tokens=1000)

def detect_query_intent(query, retrieved_docs):
    """Analyze if retrieved docs match user intent"""
    query_lower = query.lower()
    
    # Check what user is looking for
    wants_coverage = any(word in query_lower for word in ['cover', 'include', 'benefit', 'service', 'provide'])
    wants_exclusions = any(word in query_lower for word in ['exclude', 'exclusion', 'not cover', 'not include'])
    
    # Check what we actually retrieved
    has_exclusion_content = any('exclusion' in doc['content'].lower() or 'exclusion' in str(doc['metadata']).lower() 
                               for doc in retrieved_docs)
    has_coverage_content = any(word in doc['content'].lower() 
                              for doc in retrieved_docs 
                              for word in ['benefit', 'cover', 'include', 'service', 'provide'])
    
    return {
        'user_wants': 'coverage' if wants_coverage else 'exclusions' if wants_exclusions else 'general',
        'found': 'exclusions' if has_exclusion_content else 'coverage' if has_coverage_content else 'general',
        'mismatch': (wants_coverage and has_exclusion_content) or (wants_exclusions and has_coverage_content)
    }

def generate_smart_response(query, retrieved_docs, llm):
    """Generate response that handles intent mismatches gracefully"""
    
    intent_analysis = detect_query_intent(query, retrieved_docs)
    context = "\n".join([doc['content'] for doc in retrieved_docs])
    
    if intent_analysis['mismatch']:
        # User wants X but we found Y
        if intent_analysis['user_wants'] == 'coverage' and intent_analysis['found'] == 'exclusions':
            prompt = f"""The user asked about coverage but I only found exclusion information. 
            Please acknowledge this limitation and provide the exclusion information that might be helpful.
            
            User Question: {query}
            Available Context (Exclusions): {context}
            
            Response: "I couldn't find specific coverage information in the documents, but here are the exclusions that might help you understand what's not covered: [exclusion details]"
            """
        elif intent_analysis['user_wants'] == 'exclusions' and intent_analysis['found'] == 'coverage':
            prompt = f"""The user asked about exclusions but I only found coverage information.
            Please acknowledge this and provide what coverage information is available.
            
            User Question: {query}
            Available Context (Coverage): {context}
            
            Response: "I couldn't find specific exclusion information, but here's what I found about coverage: [coverage details]"
            """
        else:
            prompt = f"""I found information that might not directly match your question. Please explain what's available.
            
            User Question: {query}
            Available Context: {context}
            
            Response: "I found some information that might be related to your question: [details]"
            """
    else:
        # Normal response - intent matches
        prompt = f"""Based on the context, answer the user's question accurately and concisely.
        
        User Question: {query}
        Context: {context}
        
        Answer:"""
    
    return llm.invoke(prompt).content

# Test the smart response system
print("\n=== Testing Smart RAG System ===")

# Test ambiguous query
test_query = "what are covered under care insurance policy"
results = rag_retriever.retrieve(test_query)
smart_answer = generate_smart_response(test_query, results, llm)

print(f"Query: {test_query}")
print(f"Smart Answer: {smart_answer}")
print("-" * 50)

# Test clear query
test_query2 = "What are the exclusions in care insurance"
results2 = rag_retriever.retrieve(test_query2)
smart_answer2 = generate_smart_response(test_query2, results2, llm)

print(f"Query: {test_query2}")
print(f"Smart Answer: {smart_answer2}")
