import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
   
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  meta {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
    
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks

def create_vector_store_with_rate_limiting(
    chunks, 
    persist_directory="db/chroma_db",
    batch_size=100,
    delay_seconds=60
):
    """Create and persist ChromaDB vector store with rate limiting"""
    print(f"Creating embeddings and storing in ChromaDB with rate limiting...")
    print(f"Batch size: {batch_size} chunks")
    print(f"Delay between batches: {delay_seconds} seconds\n")
        
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # Initialize empty vector store with first batch
    total_chunks = len(chunks)
    num_batches = (total_chunks + batch_size - 1) // batch_size
    
    print(f"Total chunks: {total_chunks}")
    print(f"Number of batches: {num_batches}\n")
    
    vectorstore = None
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_chunks)
        batch_chunks = chunks[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (chunks {start_idx + 1}-{end_idx})...")
        
        try:
            if vectorstore is None:
                # Create vector store with first batch
                print("--- Creating initial vector store ---")
                vectorstore = Chroma.from_documents(
                    documents=batch_chunks,
                    embedding=embedding_model,
                    persist_directory=persist_directory,
                    collection_metadata={"hnsw:space": "cosine"}
                )
                print(f"✅ Batch {batch_idx + 1} processed successfully")
            else:
                # Add subsequent batches to existing vector store
                print("--- Adding to existing vector store ---")
                vectorstore.add_documents(documents=batch_chunks)
                print(f"✅ Batch {batch_idx + 1} processed successfully")
            
            # Wait before processing next batch (except for last batch)
            if batch_idx < num_batches - 1:
                print(f"⏳ Waiting {delay_seconds} seconds before next batch...\n")
                time.sleep(delay_seconds)
            
        except Exception as e:
            print(f"❌ Error processing batch {batch_idx + 1}: {str(e)}")
            print(f"Waiting {delay_seconds} seconds before retrying...\n")
            time.sleep(delay_seconds)
            # Retry the same batch
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch_chunks,
                    embedding=embedding_model,
                    persist_directory=persist_directory,
                    collection_metadata={"hnsw:space": "cosine"}
                )
            else:
                vectorstore.add_documents(documents=batch_chunks)
    
    print(f"\n--- Finished creating vector store ---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    # Configuration for rate limiting
    BATCH_SIZE = 5  # Process 5 chunks at a time
    DELAY_SECONDS = 60  # Wait 60 seconds between batches
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("✅ Vector store already exists. No need to re-process documents.")
        
        embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_model, 
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
        return vectorstore
    
    print("Persistent directory does not exist. Initializing vector store...\n")
    
    # Step 1: Load documents
    documents = load_documents(docs_path)  

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Create vector store with rate limiting
    vectorstore = create_vector_store_with_rate_limiting(
        chunks, 
        persistent_directory,
        batch_size=BATCH_SIZE,
        delay_seconds=DELAY_SECONDS
    )
    
    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore

if __name__ == "__main__":
    main()



# documents = [
#    Document(
#        page_content="Google LLC is an American multinational corporation and technology company focusing on online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, consumer electronics, and artificial intelligence (AI).",
#        metadata={'source': 'docs/google.txt'}
#    ),
#    Document(
#        page_content="Microsoft Corporation is an American multinational corporation and technology conglomerate headquartered in Redmond, Washington.",
#        metadata={'source': 'docs/microsoft.txt'}
#    ),
#    Document(
#        page_content="Nvidia Corporation is an American technology company headquartered in Santa Clara, California.",
#        metadata={'source': 'docs/nvidia.txt'}
#    ),
#    Document(
#        page_content="Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American space technology company headquartered at the Starbase development site in Starbase, Texas.",
#        metadata={'source': 'docs/spacex.txt'}
#    ),
#    Document(
#        page_content="Tesla, Inc. is an American multinational automotive and clean energy company headquartered in Austin, Texas.",
#        metadata={'source': 'docs/tesla.txt'}
#    )
# ]

