"""
Multi-Document RAG with ChromaDB Vector Database
Persistent storage, lightning-fast search, production-ready!
"""

import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime
import hashlib

# Try to import libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_SUPPORT = True
except ImportError:
    CHROMA_SUPPORT = False
    st.error("⚠️ ChromaDB not installed! Run: pip install chromadb")

# Page config
st.set_page_config(
    page_title="Multi-Document Chat with Vector DB",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = []


class VectorDBRAG:
    """RAG system with ChromaDB vector database for persistence and speed"""
    
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB (persistent storage)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction="You are a helpful assistant. Answer questions based on the provided context from documents. Always mention which document(s) you're referencing."
        )
        self.chat = self.model.start_chat(history=[])
    
    def smart_chunk_text(self, text, chunk_size=600, overlap=100):
        """Split text intelligently at sentence boundaries with overlap"""
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_words = ' '.join(words[-overlap//5:])
                current_chunk = overlap_words + ' ' + sentence
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def get_document_id(self, filename):
        """Generate unique ID for document"""
        return hashlib.md5(filename.encode()).hexdigest()
    
    def document_exists(self, filename):
        """Check if document already exists in database"""
        doc_id = self.get_document_id(filename)
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                limit=1
            )
            return len(results['ids']) > 0
        except:
            return False
    
    def add_document(self, file, file_type, filename):
        """Add document to vector database"""
        try:
            # Check if already exists
            if self.document_exists(filename):
                st.warning(f"⚠️ {filename} already exists in database")
                return False
            
            # Extract text
            if file_type == 'pdf':
                if not PDF_SUPPORT:
                    st.error("PDF support not available")
                    return False
                text = self.extract_text_from_pdf(file)
            else:
                text = file.read().decode('utf-8')
            
            if not text or len(text.strip()) < 10:
                st.error(f"Document {filename} is empty!")
                return False
            
            # Create chunks
            chunks = self.smart_chunk_text(text, chunk_size=600, overlap=100)
            
            # Prepare data for ChromaDB
            doc_id = self.get_document_id(filename)
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Create embeddings and add to ChromaDB
            progress_bar = st.progress(0, text=f"Processing {filename}...")
            
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = genai.embed_content(
                    model="models/text-embedding-004",
                    content=chunk,
                    task_type="retrieval_document"
                )
                
                # Add to ChromaDB
                self.collection.add(
                    ids=[chunk_ids[i]],
                    embeddings=[embedding['embedding']],
                    documents=[chunk],
                    metadatas=[{
                        "doc_name": filename,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "added_at": datetime.now().isoformat()
                    }]
                )
                
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
            return True
            
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return False
    
    def remove_document(self, filename):
        """Remove document from vector database"""
        try:
            doc_id = self.get_document_id(filename)
            
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return True
            return False
        except Exception as e:
            st.error(f"Error removing document: {e}")
            return False
    
    def get_loaded_documents(self):
        """Get list of all documents in database"""
        try:
            # Get all unique document names
            all_data = self.collection.get()
            if not all_data['metadatas']:
                return []
            
            doc_names = set()
            for metadata in all_data['metadatas']:
                doc_names.add(metadata['doc_name'])
            
            return sorted(list(doc_names))
        except:
            return []
    
    def find_relevant_chunks(self, query, top_k=5):
        """Find relevant chunks using vector database"""
        try:
            # Generate query embedding
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            
            # Query ChromaDB (SUPER FAST!)
            results = self.collection.query(
                query_embeddings=[query_embedding['embedding']],
                n_results=top_k
            )
            
            # Format results
            relevant_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    relevant_results.append({
                        'doc_name': results['metadatas'][0][i]['doc_name'],
                        'chunk': results['documents'][0][i],
                        'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'chunk_index': results['metadatas'][0][i]['chunk_index']
                    })
            
            return relevant_results
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def ask(self, question):
        """Ask question using vector database search"""
        # Find relevant chunks (INSTANT with ChromaDB!)
        relevant_results = self.find_relevant_chunks(question, top_k=5)
        
        if not relevant_results:
            return "No relevant information found. Please upload documents first.", []
        
        # Build context
        context_parts = []
        for i, result in enumerate(relevant_results):
            context_parts.append(
                f"[From: {result['doc_name']}]\n{result['chunk']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context from multiple documents, answer the question.
When referencing information, ALWAYS mention which document it came from.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (mention document names when citing information):"""
        
        # Get AI response
        response = self.chat.send_message(prompt)
        
        return response.text, relevant_results
    
    def get_stats(self):
        """Get database statistics"""
        try:
            all_data = self.collection.get()
            num_chunks = len(all_data['ids']) if all_data['ids'] else 0
            
            doc_names = set()
            if all_data['metadatas']:
                for metadata in all_data['metadatas']:
                    doc_names.add(metadata['doc_name'])
            
            return {
                'num_documents': len(doc_names),
                'total_chunks': num_chunks,
                'storage_path': './chroma_db'
            }
        except:
            return {
                'num_documents': 0,
                'total_chunks': 0,
                'storage_path': './chroma_db'
            }
    
    def clear_database(self):
        """Clear entire database"""
        try:
            # Delete collection and recreate
            self.client.delete_collection("documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            st.error(f"Error clearing database: {e}")
            return False


def main():
    # Header
    st.title("📚 Multi-Document Chat with Vector Database")
    st.markdown("Lightning-fast search with persistent storage!")
    
    # Check ChromaDB
    if not CHROMA_SUPPORT:
        st.error("⚠️ ChromaDB not installed. Run: pip install chromadb")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Get your free API key from https://aistudio.google.com/app/apikey"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Google API key")
            st.stop()
        
        # Initialize RAG system
        if st.session_state.rag_system is None:
            with st.spinner("Initializing vector database..."):
                st.session_state.rag_system = VectorDBRAG(api_key)
                # Load existing documents
                st.session_state.documents_loaded = st.session_state.rag_system.get_loaded_documents()
        
        st.divider()
        
        # File upload
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf', 'md'],
            accept_multiple_files=True,
            help="Upload multiple text files or PDFs"
        )
        
        if uploaded_files:
            if st.button("📤 Load Documents", use_container_width=True):
                success_count = 0
                
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    file_type = filename.split('.')[-1].lower()
                    
                    # Add to database
                    with st.spinner(f'Adding {filename} to database...'):
                        success = st.session_state.rag_system.add_document(
                            uploaded_file,
                            file_type,
                            filename
                        )
                    
                    if success:
                        success_count += 1
                
                if success_count > 0:
                    st.success(f"✅ Added {success_count} document(s) to database!")
                    st.session_state.documents_loaded = st.session_state.rag_system.get_loaded_documents()
                    st.balloons()
                    st.rerun()
        
        st.divider()
        
        # Document Management
        if st.session_state.documents_loaded:
            st.header("📋 Documents in Database")
            st.caption("✨ Persisted - survives refresh!")
            
            for doc_name in st.session_state.documents_loaded:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc_name}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc_name}"):
                        with st.spinner(f"Removing {doc_name}..."):
                            st.session_state.rag_system.remove_document(doc_name)
                            st.session_state.documents_loaded = st.session_state.rag_system.get_loaded_documents()
                        st.success(f"Removed {doc_name}")
                        st.rerun()
            
            st.divider()
            
            # Statistics
            stats = st.session_state.rag_system.get_stats()
            st.metric("📚 Documents", stats['num_documents'])
            st.metric("📊 Chunks", stats['total_chunks'])
            st.caption(f"💾 Stored at: {stats['storage_path']}")
        
        # Database management
        if st.session_state.documents_loaded:
            st.divider()
            if st.button("🗑️ Clear Database", use_container_width=True):
                if st.session_state.rag_system.clear_database():
                    st.session_state.documents_loaded = []
                    st.success("Database cleared!")
                    st.rerun()
        
        # Clear chat
        if st.session_state.messages:
            st.divider()
            if st.button("💬 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Main interface
    if not st.session_state.documents_loaded:
        st.info("👈 Upload documents to get started!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### ⚡ Lightning Fast")
            st.markdown("Vector database provides instant search")
        
        with col2:
            st.markdown("### 💾 Persistent")
            st.markdown("Documents survive browser refresh")
        
        with col3:
            st.markdown("### 📈 Scalable")
            st.markdown("Handle thousands of documents")
        
        st.markdown("---")
        st.markdown("### 🎯 ChromaDB Features:")
        st.markdown("""
        - ✅ Instant search (no re-embedding)
        - ✅ Persistent storage (survives refresh)
        - ✅ Efficient memory usage
        - ✅ Production-ready
        - ✅ Scales to 1000s of documents
        """)
        
    else:
        # Display chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        docs_used = {}
                        for result in message["sources"]:
                            doc_name = result['doc_name']
                            if doc_name not in docs_used:
                                docs_used[doc_name] = []
                            docs_used[doc_name].append(result)
                        
                        for doc_name, results in docs_used.items():
                            st.markdown(f"**📄 {doc_name}**")
                            for i, result in enumerate(results):
                                st.markdown(f"*Relevance: {result['score']:.2%}*")
                                st.text(result['chunk'][:200] + "..." if len(result['chunk']) > 200 else result['chunk'])
                                if i < len(results) - 1:
                                    st.markdown("---")
                            st.markdown("")
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching database..."):
                    response, sources = st.session_state.rag_system.ask(prompt)
                
                st.markdown(response)
                
                if sources:
                    with st.expander("📚 View Sources"):
                        docs_used = {}
                        for result in sources:
                            doc_name = result['doc_name']
                            if doc_name not in docs_used:
                                docs_used[doc_name] = []
                            docs_used[doc_name].append(result)
                        
                        for doc_name, results in docs_used.items():
                            st.markdown(f"**📄 {doc_name}**")
                            for i, result in enumerate(results):
                                st.markdown(f"*Relevance: {result['score']:.2%}*")
                                st.text(result['chunk'][:200] + "..." if len(result['chunk']) > 200 else result['chunk'])
                                if i < len(results) - 1:
                                    st.markdown("---")
                            st.markdown("")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })


if __name__ == "__main__":
    main()


