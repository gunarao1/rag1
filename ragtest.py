"""
RAG System - Chat with Your Documents!
Upload any text file and ask questions about it.

RAG = Retrieval Augmented Generation
- Retrieval: Find relevant parts of document
- Augmented: Add that context to prompt
- Generation: AI answers based on YOUR data
"""
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# Your API key
API_KEY = "AIzaSyDZxmiP5mumEAqR9YM0YFr87GAQ40X-KV0"

# Configure Gemini (fixed import)
genai.configure(api_key=API_KEY)

class SimpleRAG:
    """Simple RAG system that chunks documents and finds relevant pieces"""
    
    def __init__(self):
        self.chunks = []  # Store document chunks
        self.embeddings = []  # Store vector embeddings
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction="You are a helpful assistant. Answer questions based ONLY on the provided context. If the answer isn't in the context, say so."
        )
        self.chat = self.model.start_chat(history=[])
    
    def load_document(self, file_path):
        """Load and chunk a text document"""
        print(f"\n📄 Loading document: {file_path}")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        #chunk_size = 800  # characters per chunk
        #chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        chunks = smart_chunk(text)

        # Split into chunks (simple approach: by paragraphs)
       # chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        
        # If no paragraph breaks, split by sentences
        if len(chunks) < 3:
            chunks = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        print(f"✅ Loaded {len(chunks)} chunks")
        
        # Create embeddings for each chunk
        print("🔢 Creating embeddings...")
        embeddings = []
        for i, chunk in enumerate(chunks):
            embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )
            embeddings.append(embedding['embedding'])
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(chunks)} chunks...")
        
        self.chunks = chunks
        self.embeddings = embeddings
        print(f"✅ Document ready! You can now ask questions.\n")
    
    def find_relevant_chunks(self, query, top_k=5):
        """Find the most relevant chunks for a query"""
        
        # Create embedding for the query
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Calculate similarity between query and all chunks
        similarities = cosine_similarity(
            [query_embedding],
            self.embeddings
        )[0]
        
        # Get top K most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = [self.chunks[i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        
        return relevant_chunks, scores
    
    def ask(self, question):
        """Ask a question about the document"""
        
        if not self.chunks:
            return "❌ No document loaded. Use load_document() first!"
        
        # Find relevant chunks
        relevant_chunks, scores = self.find_relevant_chunks(question, top_k=3)
        
        # Build context from relevant chunks
        context = "\n\n".join([f"[Chunk {i+1}]: {chunk}" 
                               for i, chunk in enumerate(relevant_chunks)])
        
        # Create prompt with context
        prompt = f"""Based on the following context from the document, answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (based only on the context above):"""
        
        # Get AI response with streaming
        print("\n🤖 AI: ", end='', flush=True)
        response = self.chat.send_message(prompt, stream=True)
        
        full_response = ""
        for chunk in response:
            print(chunk.text, end='', flush=True)
            full_response += chunk.text
        
        print("\n")
        
        # Show which chunks were used (for transparency)
        print(f"📊 Used {len(relevant_chunks)} relevant chunks (similarity: {scores[0]:.2f}, {scores[1]:.2f}, {scores[2]:.2f})")
        
        return full_response


def main():
    """Main interactive loop"""
    print("=" * 60)
    print("🚀 RAG System - Chat with Your Documents")
    print("=" * 60)
    
    rag = SimpleRAG()
    
    # Ask for document to load
    print("\nCommands:")
    print("  load <filename> - Load a document")
    print("  ask <question>  - Ask about the document")
    print("  quit           - Exit")
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.startswith('load '):
            file_path = user_input[5:].strip()
            try:
                rag.load_document(file_path)
            except Exception as e:
                print(f"❌ Error loading file: {e}")
        
        elif user_input.startswith('ask '):
            question = user_input[4:].strip()
            if question:
                try:
                    rag.ask(question)
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ Please provide a question!")
        
        else:
            print("❌ Unknown command. Use 'load <file>' or 'ask <question>'")
    
    print("\nGoodbye! 👋")

def smart_chunk(text, chunk_size=200, overlap=50):
    """Split text at sentence boundaries with overlap"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

if __name__ == "__main__":
    main()


"""
=== HOW TO RUN ===

STEP 1: INSTALL REQUIRED LIBRARIES
pip install google-generativeai numpy scikit-learn

STEP 2: CREATE A TEST DOCUMENT
Create a file called 'test.txt' with some content:

Python is a high-level programming language. It was created by Guido van Rossum
and released in 1991. Python emphasizes code readability with significant whitespace.

Python supports multiple programming paradigms including procedural, object-oriented,
and functional programming. It has a comprehensive standard library.

Python is widely used in web development, data science, artificial intelligence,
scientific computing, and automation.

STEP 3: RUN THE APP
python rag_app.py

STEP 4: TRY IT
> load test.txt
> ask Who created Python?
> ask What is Python used for?
> ask When was Python released?


=== HOW RAG WORKS - STEP BY STEP ===

1. LOADING DOCUMENT:
   Document → Split into chunks → Create embeddings for each chunk → Store

2. ASKING QUESTION:
   Question → Create embedding → Find similar chunks → Add to prompt → Ask AI

3. THE MAGIC - EMBEDDINGS:
   Text → [0.23, -0.45, 0.78, ...] (768 numbers)
   Similar text = similar numbers!
   
   "Python programming" → [0.2, 0.5, 0.1, ...]
   "Coding in Python"   → [0.3, 0.4, 0.2, ...]  ← SIMILAR!
   "Cooking recipes"    → [0.9, -0.8, 0.3, ...] ← DIFFERENT!


=== KEY CONCEPTS EXPLAINED ===

1. CHUNKING:
   Why? AI has token limits. Can't send entire book!
   Solution: Split into small pieces (chunks)
   
   Document (10,000 words) → Chunks (200 words each) → 50 chunks

2. EMBEDDINGS:
   Text → Vector (list of numbers)
   Similar meaning = similar vectors
   Model: "text-embedding-004" (Google's embedding model)

3. COSINE SIMILARITY:
   Measures how similar two vectors are
   Score 0.0 = completely different
   Score 1.0 = identical
   Score > 0.7 = very relevant

4. TOP-K RETRIEVAL:
   Find the K most relevant chunks
   We use top_k=3 (best 3 chunks)
   Send only these to AI (saves tokens + more focused)


=== WHAT THIS CODE DOES ===

SimpleRAG Class:
  ├─ load_document(): Read file, split chunks, create embeddings
  ├─ find_relevant_chunks(): Search for relevant pieces
  └─ ask(): Find context + ask AI

Flow:
User uploads doc → Chunks + embeddings created
User asks question → Find relevant chunks → Add to prompt → AI answers


=== TRY THESE EXPERIMENTS ===

1. ADJUST CHUNK SIZE:
   # In load_document(), change splitting:
   chunk_size = 500  # characters per chunk
   chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

2. CHANGE TOP-K (number of chunks used):
   relevant_chunks, scores = self.find_relevant_chunks(question, top_k=5)
   # More chunks = more context but uses more tokens

3. ADD SIMILARITY THRESHOLD:
   # Only use chunks above certain similarity
   relevant_chunks = [chunk for chunk, score in zip(relevant_chunks, scores) if score > 0.5]

4. SHOW CHUNK SOURCES:
   # After AI response, show which chunks were used:
   print("\n📚 Sources used:")
   for i, chunk in enumerate(relevant_chunks):
       print(f"\nChunk {i+1} (similarity: {scores[i]:.2f}):")
       print(chunk[:100] + "...")


=== ADVANCED FEATURES TO ADD ===

1. MULTIPLE DOCUMENTS:
   Store chunks with document names
   Search across all documents
   
2. BETTER CHUNKING:
   Use semantic chunking (LangChain)
   Keep sentences together
   Add overlap between chunks

3. VECTOR DATABASE:
   Replace in-memory storage with:
   - Pinecone (cloud)
   - ChromaDB (local)
   - Qdrant (self-hosted)

4. METADATA FILTERING:
   Add metadata to chunks (date, author, section)
   Filter before similarity search

5. HYBRID SEARCH:
   Combine embeddings (semantic) + keywords (BM25)


=== COMPARISON: RAG vs FINE-TUNING ===

RAG (what we built):
✓ Add new data anytime (just load document)
✓ No training needed
✓ Cheaper
✓ Can cite sources
✗ Needs good retrieval

FINE-TUNING:
✓ Knowledge "baked in" to model
✓ No retrieval needed
✗ Expensive ($100s-$1000s)
✗ Can't easily update
✗ Can't cite sources

For 90% of cases: USE RAG!


=== REAL-WORLD RAG APPLICATIONS ===

- Customer support (search knowledge base)
- Legal document analysis
- Research paper Q&A
- Code documentation chatbot
- Medical records analysis
- Company policy assistant


=== COMMON ISSUES & FIXES ===

1. "No relevant chunks found"
   FIX: Document too short or question too different
   Try: Lower similarity threshold

2. "Out of context" answers
   FIX: Increase top_k (use more chunks)
   
3. "Slow embedding creation"
   FIX: Normal for large docs. Add progress bar.
   
4. "AI hallucinates despite context"
   FIX: Improve system prompt, emphasize "ONLY use context"


=== COSTS ===

Gemini Embeddings (text-embedding-004):
- FREE for reasonable usage
- Rate limits apply

Gemini API calls:
- Same as before (FREE tier)


=== PRODUCTION-READY IMPROVEMENTS ===

For real apps, add:
1. Vector database (ChromaDB, Pinecone)
2. Better chunking (semantic, with overlap)
3. Caching (don't re-embed same text)
4. Error handling
5. Async processing
6. Web interface


=== NEXT LEVEL: LANGCHAIN ===

This is a "from scratch" RAG to understand concepts.
In production, use LangChain:

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 5 lines vs our 100+ lines!

But NOW you understand what's happening under the hood! 🎉


=== WHAT YOU LEARNED ===

✅ What embeddings are (text → numbers)
✅ Vector similarity search
✅ Chunking strategies
✅ Retrieval + Generation (RAG)
✅ Context-aware AI responses
✅ Building production AI features

This is the foundation of most AI products today!
"""