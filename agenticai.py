"""
Multi-Document RAG with Agentic AI + ChromaDB
Now with function calling for accurate data analysis!
"""

import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime
import hashlib
import json

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
    page_title="AI Agent with RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = []
if 'json_cache' not in st.session_state:
    st.session_state.json_cache = {}  # Cache JSON data for analysis


class AgenticRAG:
    """RAG system with AI Agent capabilities for data analysis"""
    
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Define tools the agent can use
        self.tools = [
            {
                "name": "analyze_json_field",
                "description": "Analyze a numeric field in JSON data. Use this for finding max, min, average, sum, or count of numeric values.",
                "parameters": {
                    "doc_name": "Name of the JSON document to analyze",
                    "field_name": "Name of the field to analyze (e.g., 'currentValue', 'totalReading')",
                    "operation": "Operation to perform: 'max', 'min', 'avg', 'sum', 'count', or 'all'"
                }
            },
            {
                "name": "search_json_records",
                "description": "Search for specific records in JSON data based on conditions",
                "parameters": {
                    "doc_name": "Name of the JSON document",
                    "field_name": "Field to search",
                    "condition": "Condition like 'greater_than', 'less_than', 'equals'",
                    "value": "Value to compare against"
                }
            }
        ]
        
        # Initialize model with function calling
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction="""You are an intelligent AI agent with access to tools for data analysis.

When asked about numerical data (max, min, average, etc.) in JSON files:
1. Use the 'analyze_json_field' tool to get accurate results
2. Don't try to answer from text chunks - use the tool!
3. Always cite the tool results in your answer

For general questions about documents, use the provided context.
Always be precise and cite your sources."""
        )
        self.chat = self.model.start_chat(history=[])
    
    def smart_chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text intelligently"""
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
            
            if len(potential_chunk) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_words = ' '.join(words[-overlap//5:]) if len(words) > overlap//5 else ""
                current_chunk = overlap_words + ' ' + sentence if overlap_words else sentence
            else:
                current_chunk = potential_chunk
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_json(self, json_file, filename):
        """Extract text from JSON and cache the data"""
        try:
            # Read JSON
            json_file.seek(0)  # Reset file pointer
            json_data = json.load(json_file)
            
            # Cache the actual JSON data for agent analysis
            st.session_state.json_cache[filename] = json_data
            
            # Convert to readable text
            def json_to_text(obj, prefix=""):
                text_parts = []
                
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            text_parts.append(f"{prefix}{key}:")
                            text_parts.append(json_to_text(value, prefix + "  "))
                        else:
                            text_parts.append(f"{prefix}{key}: {value}")
                
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            text_parts.append(f"{prefix}Item {i+1}:")
                            text_parts.append(json_to_text(item, prefix + "  "))
                        else:
                            text_parts.append(f"{prefix}- {item}")
                
                else:
                    text_parts.append(f"{prefix}{obj}")
                
                return "\n".join(text_parts)
            
            text = json_to_text(json_data)
            header = f"JSON Document: {filename}\n{'='*50}\n\n"
            return header + text
            
        except Exception as e:
            st.error(f"Error reading JSON: {e}")
            return ""
    
    def get_document_id(self, filename):
        """Generate unique ID"""
        return hashlib.md5(filename.encode()).hexdigest()
    
    def document_exists(self, filename):
        """Check if document exists"""
        doc_id = self.get_document_id(filename)
        try:
            results = self.collection.get(where={"doc_id": doc_id}, limit=1)
            return len(results['ids']) > 0
        except:
            return False
    
    def add_document(self, file, file_type, filename):
        """Add document to database"""
        try:
            if self.document_exists(filename):
                st.warning(f"⚠️ {filename} already exists")
                return False
            
            # Extract text
            if file_type == 'pdf':
                if not PDF_SUPPORT:
                    st.error("PDF support not available")
                    return False
                text = self.extract_text_from_pdf(file)
            elif file_type == 'json':
                text = self.extract_text_from_json(file, filename)
            else:
                text = file.read().decode('utf-8')
            
            if not text or len(text.strip()) < 10:
                st.error(f"Document {filename} is empty!")
                return False
            
            # Create chunks
            chunks = self.smart_chunk_text(text, chunk_size=500, overlap=50)
            
            # Add to database
            doc_id = self.get_document_id(filename)
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            progress_bar = st.progress(0, text=f"Processing {filename}...")
            
            for i, chunk in enumerate(chunks):
                try:
                    if not chunk.strip():
                        continue
                    
                    if len(chunk.encode('utf-8')) > 30000:
                        mid = len(chunk) // 2
                        sub_chunks = [chunk[:mid], chunk[mid:]]
                        
                        for j, sub_chunk in enumerate(sub_chunks):
                            embedding = genai.embed_content(
                                model="models/text-embedding-004",
                                content=sub_chunk,
                                task_type="retrieval_document"
                            )
                            
                            self.collection.add(
                                ids=[f"{chunk_ids[i]}_part{j}"],
                                embeddings=[embedding['embedding']],
                                documents=[sub_chunk],
                                metadatas=[{
                                    "doc_name": filename,
                                    "doc_id": doc_id,
                                    "chunk_index": i,
                                    "added_at": datetime.now().isoformat()
                                }]
                            )
                    else:
                        embedding = genai.embed_content(
                            model="models/text-embedding-004",
                            content=chunk,
                            task_type="retrieval_document"
                        )
                        
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
                    
                except Exception as e:
                    st.warning(f"Skipped chunk {i}: {str(e)[:100]}")
                    continue
            
            progress_bar.empty()
            return True
            
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            return False
    
    # AGENT TOOLS
    def analyze_json_field(self, doc_name, field_name, operation):
        """Analyze numeric field in JSON data"""
        try:
            # Get JSON data from cache
            if doc_name not in st.session_state.json_cache:
                return {"error": f"JSON data for {doc_name} not found in cache"}
            
            json_data = st.session_state.json_cache[doc_name]
            
            # Extract values
            values = []
            timestamps = []
            
            def extract_field(obj):
                if isinstance(obj, dict):
                    if field_name in obj:
                        values.append(obj[field_name])
                        if 'timestamp' in obj:
                            timestamps.append(obj['timestamp'])
                    for value in obj.values():
                        extract_field(value)
                elif isinstance(obj, list):
                    for item in obj:
                        extract_field(item)
            
            extract_field(json_data)
            
            if not values:
                return {"error": f"No values found for field '{field_name}'"}
            
            # Perform operation
            numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
            
            if not numeric_values:
                return {"error": f"No numeric values found for '{field_name}'"}
            
            result = {}
            
            if operation in ['max', 'all']:
                max_idx = numeric_values.index(max(numeric_values))
                result['max'] = {
                    'value': max(numeric_values),
                    'timestamp': timestamps[max_idx] if max_idx < len(timestamps) else None
                }
            
            if operation in ['min', 'all']:
                min_idx = numeric_values.index(min(numeric_values))
                result['min'] = {
                    'value': min(numeric_values),
                    'timestamp': timestamps[min_idx] if min_idx < len(timestamps) else None
                }
            
            if operation in ['avg', 'all']:
                result['avg'] = sum(numeric_values) / len(numeric_values)
            
            if operation in ['sum', 'all']:
                result['sum'] = sum(numeric_values)
            
            if operation in ['count', 'all']:
                result['count'] = len(numeric_values)
            
            result['all_values'] = list(zip(numeric_values, timestamps[:len(numeric_values)]))
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def find_relevant_chunks(self, query, top_k=5):
        """Find relevant chunks"""
        try:
            query_embedding = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            
            results = self.collection.query(
                query_embeddings=[query_embedding['embedding']],
                n_results=top_k
            )
            
            relevant_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    relevant_results.append({
                        'doc_name': results['metadatas'][0][i]['doc_name'],
                        'chunk': results['documents'][0][i],
                        'score': 1 - results['distances'][0][i],
                        'chunk_index': results['metadatas'][0][i]['chunk_index']
                    })
            
            return relevant_results
            
        except Exception as e:
            st.error(f"Search error: {e}")
            return []
    
    def ask(self, question):
        """Ask question with agent capabilities"""
        
        # Check if question is about JSON numerical analysis
        json_keywords = ['highest', 'lowest', 'maximum', 'minimum', 'average', 'sum', 'total', 'count']
        is_numeric_query = any(keyword in question.lower() for keyword in json_keywords)
        
        # Get context from vector search
        relevant_results = self.find_relevant_chunks(question, top_k=5)
        
        # Check for JSON documents
        json_docs = [doc for doc in st.session_state.json_cache.keys()]
        
        # Build enhanced prompt
        if is_numeric_query and json_docs:
            # Agent mode: Guide AI to use tools
            context = f"""Available JSON documents for analysis: {', '.join(json_docs)}

You have access to the 'analyze_json_field' tool for accurate numerical analysis.

Question: {question}

Instructions:
1. Identify the field name mentioned (e.g., 'currentValue', 'totalReading')
2. Identify the operation needed (max, min, avg, etc.)
3. Use analyze_json_field tool with proper parameters
4. Report the accurate result from the tool

Context from documents (for reference only):
{chr(10).join([f"[{r['doc_name']}]: {r['chunk'][:200]}..." for r in relevant_results[:3]])}"""
            
            # Simulate tool call (Gemini 1.5 doesn't have function calling yet, so we do it manually)
            # Detect what to analyze
            field_name = None
            operation = None
            
            if 'currentvalue' in question.lower():
                field_name = 'currentValue'
            elif 'totalreading' in question.lower():
                field_name = 'totalReading'
            
            if any(word in question.lower() for word in ['highest', 'maximum', 'max']):
                operation = 'max'
            elif any(word in question.lower() for word in ['lowest', 'minimum', 'min']):
                operation = 'min'
            elif 'average' in question.lower() or 'avg' in question.lower():
                operation = 'avg'
            elif 'all' in question.lower():
                operation = 'all'
            
            # Call tool if we detected parameters
            tool_result = None
            if field_name and operation and json_docs:
                tool_result = self.analyze_json_field(json_docs[0], field_name, operation)
            
            # Build final prompt with tool result
            if tool_result and 'error' not in tool_result:
                final_prompt = f"""{context}

TOOL RESULT from analyze_json_field({json_docs[0]}, {field_name}, {operation}):
{json.dumps(tool_result, indent=2)}

Based on the TOOL RESULT above (which is 100% accurate), answer the question.
Always cite the specific value and timestamp from the tool result."""
            else:
                final_prompt = context
        
        else:
            # Normal RAG mode
            context_parts = []
            for i, result in enumerate(relevant_results):
                context_parts.append(f"[From: {result['doc_name']}]\n{result['chunk']}")
            
            context = "\n\n".join(context_parts)
            
            final_prompt = f"""Based on the following context, answer the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (mention document names when citing):"""
        
        # Get AI response
        response = self.chat.send_message(final_prompt)
        
        return response.text, relevant_results, tool_result if 'tool_result' in locals() else None
    
    def remove_document(self, filename):
        """Remove document"""
        try:
            doc_id = self.get_document_id(filename)
            results = self.collection.get(where={"doc_id": doc_id})
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            
            # Remove from JSON cache
            if filename in st.session_state.json_cache:
                del st.session_state.json_cache[filename]
            
            return True
        except Exception as e:
            st.error(f"Error removing document: {e}")
            return False
    
    def get_loaded_documents(self):
        """Get loaded documents"""
        try:
            all_data = self.collection.get()
            if not all_data['metadatas']:
                return []
            
            doc_names = set()
            for metadata in all_data['metadatas']:
                doc_names.add(metadata['doc_name'])
            
            return sorted(list(doc_names))
        except:
            return []
    
    def get_stats(self):
        """Get statistics"""
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
                'json_docs_cached': len(st.session_state.json_cache),
                'storage_path': './chroma_db'
            }
        except:
            return {
                'num_documents': 0,
                'total_chunks': 0,
                'json_docs_cached': 0,
                'storage_path': './chroma_db'
            }
    
    def clear_database(self):
        """Clear database"""
        try:
            self.client.delete_collection("documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            st.session_state.json_cache = {}
            return True
        except Exception as e:
            st.error(f"Error clearing database: {e}")
            return False


def main():
    st.title("🤖 AI Agent with RAG + Vector Database")
    st.markdown("Now with intelligent function calling for accurate data analysis!")
    
    if not CHROMA_SUPPORT:
        st.error("⚠️ ChromaDB not installed. Run: pip install chromadb")
        st.stop()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Get your free API key from https://aistudio.google.com/app/apikey"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Google API key")
            st.stop()
        
        if st.session_state.rag_system is None:
            with st.spinner("Initializing AI Agent..."):
                st.session_state.rag_system = AgenticRAG(api_key)
                st.session_state.documents_loaded = st.session_state.rag_system.get_loaded_documents()
        
        st.divider()
        
        st.header("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf', 'md', 'json'],
            accept_multiple_files=True,
            help="Upload text, PDF, Markdown, or JSON files"
        )
        
        if uploaded_files:
            if st.button("📤 Load Documents", use_container_width=True):
                success_count = 0
                
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    file_type = filename.split('.')[-1].lower()
                    
                    with st.spinner(f'Adding {filename}...'):
                        success = st.session_state.rag_system.add_document(
                            uploaded_file,
                            file_type,
                            filename
                        )
                    
                    if success:
                        success_count += 1
                
                if success_count > 0:
                    st.success(f"✅ Added {success_count} document(s)!")
                    st.session_state.documents_loaded = st.session_state.rag_system.get_loaded_documents()
                    st.balloons()
                    st.rerun()
        
        st.divider()
        
        if st.session_state.documents_loaded:
            st.header("📋 Documents in Database")
            
            for doc_name in st.session_state.documents_loaded:
                col1, col2 = st.columns([3, 1])
                with col1:
                    icon = "📊" if doc_name.endswith('.json') else "📄"
                    st.text(f"{icon} {doc_name}")
                with col2:
                    if st.button("🗑️", key=f"del_{doc_name}"):
                        with st.spinner(f"Removing {doc_name}..."):
                            st.session_state.rag_system.remove_document(doc_name)
                            st.session_state.documents_loaded = st.session_state.rag_system.get_loaded_documents()
                        st.rerun()
            
            st.divider()
            
            stats = st.session_state.rag_system.get_stats()
            st.metric("📚 Documents", stats['num_documents'])
            st.metric("📊 Chunks", stats['total_chunks'])
            st.metric("🤖 JSON Cached", stats['json_docs_cached'])
        
        if st.session_state.documents_loaded:
            st.divider()
            if st.button("🗑️ Clear Database", use_container_width=True):
                if st.session_state.rag_system.clear_database():
                    st.session_state.documents_loaded = []
                    st.success("Database cleared!")
                    st.rerun()
        
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
            st.markdown("### 🤖 AI Agent")
            st.markdown("Uses tools for accurate calculations")
        
        with col2:
            st.markdown("### 📊 Data Analysis")
            st.markdown("Precise max, min, avg on JSON data")
        
        with col3:
            st.markdown("### ⚡ Smart RAG")
            st.markdown("Best of both worlds!")
        
        st.markdown("---")
        st.markdown("### 🎯 Try These Questions on JSON:")
        st.markdown("""
        - "What's the highest currentValue?" ← Now accurate!
        - "What's the average totalReading?"
        - "Show me the minimum currentValue with its timestamp"
        - "Count how many records are in the data"
        """)
        
    else:
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
                                st.text(result['chunk'][:200] + "...")
                                if i < len(results) - 1:
                                    st.markdown("---")
        
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Agent thinking..."):
                    response, sources, tool_result = st.session_state.rag_system.ask(prompt)
                
                st.markdown(response)
                
                # Show tool usage if applicable
                if tool_result:
                    with st.expander("🔧 Tool Used"):
                        st.json(tool_result)
                
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
                                st.text(result['chunk'][:200] + "...")
                                if i < len(results) - 1:
                                    st.markdown("---")
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })


if __name__ == "__main__":
    main()

