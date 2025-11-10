"""
Beautiful Web UI for RAG System using Streamlit
Chat with your documents in a professional interface!
"""

import streamlit as st
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os

# Try to import PDF library
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Page config
st.set_page_config(
    page_title="Chat with Documents",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'document_loaded' not in st.session_state:
    st.session_state.document_loaded = False


class StreamlitRAG:
    """RAG system optimized for Streamlit"""
    
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.chunks = []
        self.embeddings = []
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction="You are a helpful assistant. Answer questions based ONLY on the provided context. If the answer isn't in the context, say so clearly."
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
    
    def load_document(self, file, file_type):
        """Load document from uploaded file"""
        try:
            if file_type == 'pdf':
                if not PDF_SUPPORT:
                    st.error("PDF support not available. Install PyPDF2: pip install PyPDF2")
                    return False
                text = self.extract_text_from_pdf(file)
            else:  # text file
                text = file.read().decode('utf-8')
            
            if not text or len(text.strip()) < 10:
                st.error("Document is empty or too short!")
                return False
            
            # Create chunks
            with st.spinner('Creating smart chunks...'):
                self.chunks = self.smart_chunk_text(text, chunk_size=600, overlap=100)
            
            # Create embeddings
            with st.spinner(f'Creating embeddings for {len(self.chunks)} chunks...'):
                self.embeddings = []
                progress_bar = st.progress(0)
                
                for i, chunk in enumerate(self.chunks):
                    embedding = genai.embed_content(
                        model="models/text-embedding-004",
                        content=chunk,
                        task_type="retrieval_document"
                    )
                    self.embeddings.append(embedding['embedding'])
                    progress_bar.progress((i + 1) / len(self.chunks))
                
                progress_bar.empty()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading document: {e}")
            return False
    
    def find_relevant_chunks(self, query, top_k=3):
        """Find most relevant chunks for query"""
        query_embedding = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = [self.chunks[i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        
        return relevant_chunks, scores
    
    def ask(self, question):
        """Ask question about the document"""
        if not self.chunks:
            return "Please upload a document first!", []
        
        # Find relevant chunks
        relevant_chunks, scores = self.find_relevant_chunks(question, top_k=3)
        
        # Build context
        context = "\n\n".join([f"[Source {i+1}]: {chunk}" 
                               for i, chunk in enumerate(relevant_chunks)])
        
        # Create prompt
        prompt = f"""Based on the following context from the document, answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (based only on the context above):"""
        
        # Get AI response
        response = self.chat.send_message(prompt)
        
        return response.text, list(zip(relevant_chunks, scores))


def main():
    # Header
    st.title("📚 Chat with Your Documents")
    st.markdown("Upload a document and ask questions about it using AI!")
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key input
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Get your free API key from https://aistudio.google.com/app/apikey"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Google API key to continue")
            st.stop()
        
        st.divider()
        
        # File upload
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'md'],
            help="Upload a text file or PDF to chat with"
        )
        
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if st.button("📤 Load Document", use_container_width=True):
                # Initialize RAG system
                if st.session_state.rag_system is None:
                    st.session_state.rag_system = StreamlitRAG(api_key)
                
                # Load document
                with st.spinner('Loading document...'):
                    success = st.session_state.rag_system.load_document(
                        uploaded_file, 
                        file_type
                    )
                
                if success:
                    st.session_state.document_loaded = True
                    st.success(f"✅ Loaded {len(st.session_state.rag_system.chunks)} chunks!")
                    st.balloons()
        
        # Document info
        if st.session_state.document_loaded and st.session_state.rag_system:
            st.divider()
            st.metric("📊 Chunks", len(st.session_state.rag_system.chunks))
            st.metric("🔢 Embeddings", len(st.session_state.rag_system.embeddings))
        
        # Clear chat button
        if st.session_state.messages:
            st.divider()
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Main chat interface
    if not st.session_state.document_loaded:
        # Welcome message
        st.info("👈 Upload a document from the sidebar to get started!")
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🚀 Fast")
            st.markdown("Get answers in seconds using advanced AI")
        
        with col2:
            st.markdown("### 🎯 Accurate")
            st.markdown("Answers based only on your documents")
        
        with col3:
            st.markdown("### 🔒 Private")
            st.markdown("Your documents stay secure")
        
        # Example questions
        st.markdown("---")
        st.markdown("### 💡 Example Questions You Can Ask:")
        st.markdown("""
        - "What is the main topic of this document?"
        - "Summarize the key points"
        - "Who is mentioned in this document?"
        - "What are the conclusions?"
        """)
        
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources if available
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, (chunk, score) in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}** (relevance: {score:.2%})")
                            st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                            st.divider()
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources = st.session_state.rag_system.ask(prompt)
                
                st.markdown(response)
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, (chunk, score) in enumerate(sources):
                        st.markdown(f"**Source {i+1}** (relevance: {score:.2%})")
                        st.text(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                        st.divider()
            
            # Save assistant message with sources
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })


if __name__ == "__main__":
    main()

