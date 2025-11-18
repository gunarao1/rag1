"""
Multi-Document RAG System with Streamlit
Upload multiple documents and search across ALL of them!
"""

import streamlit as st
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

# Try to import PDF library
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Page config
st.set_page_config(
    page_title="Multi-Document Chat",
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


class MultiDocumentRAG:
    """RAG system that handles multiple documents"""
    
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Store chunks with document metadata
        self.documents = []  # List of {name, chunks, embeddings}
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
    
    def add_document(self, file, file_type, filename):
        """Add a new document to the system"""
        try:
            # Extract text
            if file_type == 'pdf':
                if not PDF_SUPPORT:
                    st.error("PDF support not available. Install PyPDF2")
                    return False
                text = self.extract_text_from_pdf(file)
            else:  # text file
                text = file.read().decode('utf-8')
            
            if not text or len(text.strip()) < 10:
                st.error(f"Document {filename} is empty or too short!")
                return False
            
            # Create chunks
            chunks = self.smart_chunk_text(text, chunk_size=600, overlap=100)
            
            # Create embeddings
            embeddings = []
            progress_bar = st.progress(0, text=f"Processing {filename}...")
            
            for i, chunk in enumerate(chunks):
                embedding = genai.embed_content(
                    model="models/text-embedding-004",
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.append(embedding['embedding'])
                progress_bar.progress((i + 1) / len(chunks))
            
            progress_bar.empty()
            
            # Store document with metadata
            self.documents.append({
                'name': filename,
                'chunks': chunks,
                'embeddings': embeddings,
                'added_at': datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            
            return True
            
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return False
    
    def remove_document(self, doc_name):
        """Remove a document from the system"""
        self.documents = [doc for doc in self.documents if doc['name'] != doc_name]
    
    def find_relevant_chunks(self, query, top_k=5):
        """Find most relevant chunks across ALL documents"""
        if not self.documents:
            return [], []
        
        # Create query embedding
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Search across all documents
        all_results = []
        
        for doc in self.documents:
            # Calculate similarities for this document
            similarities = cosine_similarity(
                [query_embedding],
                doc['embeddings']
            )[0]
            
            # Get best chunks from this document
            for i, score in enumerate(similarities):
                all_results.append({
                    'doc_name': doc['name'],
                    'chunk': doc['chunks'][i],
                    'score': score,
                    'doc_index': self.documents.index(doc)
                })
        
        # Sort all results by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Get top K results
        top_results = all_results[:top_k]
        
        return top_results
    
    def ask(self, question):
        """Ask question across all documents"""
        if not self.documents:
            return "Please upload at least one document first!", []
        
        # Find relevant chunks across all documents
        relevant_results = self.find_relevant_chunks(question, top_k=5)
        
        if not relevant_results:
            return "No relevant information found in the documents.", []
        
        # Build context with document sources
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
        """Get statistics about loaded documents"""
        total_chunks = sum(len(doc['chunks']) for doc in self.documents)
        total_embeddings = sum(len(doc['embeddings']) for doc in self.documents)
        
        return {
            'num_documents': len(self.documents),
            'total_chunks': total_chunks,
            'total_embeddings': total_embeddings
        }


def main():
    # Header
    st.title("📚 Multi-Document Chat System")
    st.markdown("Upload multiple documents and search across all of them!")
    
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
            st.session_state.rag_system = MultiDocumentRAG(api_key)
        
        st.divider()
        
        # File upload (multiple files!)
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf', 'md'],
            accept_multiple_files=True,  # KEY: Allow multiple files!
            help="Upload multiple text files or PDFs"
        )
        
        if uploaded_files:
            if st.button("📤 Load All Documents", use_container_width=True):
                success_count = 0
                
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    
                    # Check if already loaded
                    if filename in st.session_state.documents_loaded:
                        st.info(f"⏭️ {filename} already loaded")
                        continue
                    
                    file_type = filename.split('.')[-1].lower()
                    
                    # Load document
                    with st.spinner(f'Loading {filename}...'):
                        success = st.session_state.rag_system.add_document(
                            uploaded_file,
                            file_type,
                            filename
                        )
                    
                    if success:
                        st.session_state.documents_loaded.append(filename)
                        success_count += 1
                
                if success_count > 0:
                    st.success(f"✅ Loaded {success_count} new document(s)!")
                    st.balloons()
                    st.rerun()
        
        st.divider()
        
        # Document Management
        if st.session_state.documents_loaded:
            st.header("📋 Loaded Documents")
            
            for doc_name in st.session_state.documents_loaded:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {doc_name}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc_name}"):
                        st.session_state.rag_system.remove_document(doc_name)
                        st.session_state.documents_loaded.remove(doc_name)
                        st.rerun()
            
            st.divider()
            
            # Statistics
            stats = st.session_state.rag_system.get_stats()
            st.metric("📚 Documents", stats['num_documents'])
            st.metric("📊 Total Chunks", stats['total_chunks'])
            st.metric("🔢 Embeddings", stats['total_embeddings'])
        
        # Clear all
        if st.session_state.documents_loaded:
            st.divider()
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                st.session_state.rag_system.documents = []
                st.session_state.documents_loaded = []
                st.success("All documents cleared!")
                st.rerun()
        
        # Clear chat
        if st.session_state.messages:
            st.divider()
            if st.button("💬 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    if not st.session_state.documents_loaded:
        # Welcome screen
        st.info("👈 Upload documents from the sidebar to get started!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📚 Multiple Docs")
            st.markdown("Search across all your documents at once")
        
        with col2:
            st.markdown("### 🎯 Smart Search")
            st.markdown("Finds relevant info from the right document")
        
        with col3:
            st.markdown("### 🔍 Source Tracking")
            st.markdown("Always shows which document answered")
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions:")
        st.markdown("""
        - "Compare the approaches mentioned in all documents"
        - "Which document talks about Python?"
        - "Summarize the key points from each document"
        - "What are the differences between document A and B?"
        """)
        
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources with document names
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        # Group by document
                        docs_used = {}
                        for result in message["sources"]:
                            doc_name = result['doc_name']
                            if doc_name not in docs_used:
                                docs_used[doc_name] = []
                            docs_used[doc_name].append(result)
                        
                        # Display by document
                        for doc_name, results in docs_used.items():
                            st.markdown(f"**📄 {doc_name}**")
                            for i, result in enumerate(results):
                                st.markdown(f"*Relevance: {result['score']:.2%}*")
                                st.text(result['chunk'][:200] + "..." if len(result['chunk']) > 200 else result['chunk'])
                                if i < len(results) - 1:
                                    st.markdown("---")
                            st.markdown("")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Searching across all documents..."):
                    response, sources = st.session_state.rag_system.ask(prompt)
                
                st.markdown(response)
                
                # Show sources grouped by document
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
            
            # Save assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })


if __name__ == "__main__":
    main()


