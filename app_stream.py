import streamlit as st
import os
import hashlib
from app_start import MultimodalRAG
import traceback
import json
from datetime import datetime


def get_file_hash(file_content):
    """Calculate SHA-256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()


def load_file_metadata():
    """Load processed files metadata from JSON"""
    metadata_path = os.path.join(os.getcwd(), 'processed_files.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.sidebar.warning(f"Error loading metadata: {str(e)}")
            return {}
    return {}


def save_file_metadata(metadata):
    """Save processed files metadata to JSON"""
    metadata_path = os.path.join(os.getcwd(), 'processed_files.json')
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        st.sidebar.warning(f"Error saving metadata: {str(e)}")


def check_file_exists(file_hash, filename):
    """Check if file exists and is valid"""
    # Load metadata
    metadata = load_file_metadata()

    # Check if file exists in metadata
    if file_hash in metadata:
        file_info = metadata[file_hash]
        file_path = file_info.get('file_path')

        # Verify file exists and matches metadata
        if (os.path.exists(file_path) and
                os.path.basename(file_path) == filename and
                os.path.getsize(file_path) == file_info.get('size')):
            return True, file_path, file_info

    return False, None, None


def initialize_rag():
    """Initialize RAG system with persistent database"""
    try:
        persist_directory = os.path.join(os.getcwd(), 'chroma_db')
        os.makedirs(persist_directory, exist_ok=True)
        return MultimodalRAG(input_path=os.getcwd(), persist_directory=persist_directory)
    except Exception as e:
        st.sidebar.error(f"Error initializing RAG system: {str(e)}")
        st.sidebar.code(traceback.format_exc())
        return None


def process_document(rag, file_path):
    """Process document with error handling"""
    try:
        return rag.process_document(file_path)
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.code(traceback.format_exc())
        return None


def save_uploaded_file(uploaded_file, file_hash):
    """Save uploaded file to disk and return the file path"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(upload_dir, exist_ok=True)

        # Save the file
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            file_content = uploaded_file.getvalue()
            f.write(file_content)

        # Update metadata
        metadata = load_file_metadata()
        metadata[file_hash] = {
            'filename': uploaded_file.name,
            'file_path': file_path,
            'size': len(file_content),
            'upload_time': datetime.now().isoformat(),
            'last_processed': None
        }
        save_file_metadata(metadata)

        return file_path
    except Exception as e:
        st.sidebar.error(f"Error saving uploaded file: {str(e)}")
        return None


def update_processing_status(file_hash, stats):
    """Update processing status in metadata"""
    try:
        metadata = load_file_metadata()
        if file_hash in metadata:
            metadata[file_hash]['last_processed'] = datetime.now().isoformat()
            metadata[file_hash]['processing_stats'] = stats
            save_file_metadata(metadata)
    except Exception as e:
        st.sidebar.warning(f"Error updating processing status: {str(e)}")


def sidebar_upload():
    """Handle file upload in sidebar"""
    st.sidebar.markdown("### Document Upload")

    # Disable file upload during processing
    disabled = st.session_state.get('processing_status', False)

    if disabled:
        st.sidebar.warning("⚠️ Please wait while processing the current document...")

    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        key="pdf_uploader",
        disabled=disabled
    )

    if uploaded_file is not None:
        # Calculate file hash
        file_content = uploaded_file.getvalue()
        file_hash = get_file_hash(file_content)

        # Check if file already exists and is processed
        exists, existing_path, file_info = check_file_exists(file_hash, uploaded_file.name)

        if exists:
            st.session_state.current_file = uploaded_file.name
            st.session_state.file_path = existing_path
            st.session_state.file_hash = file_hash

            # Check if file was previously processed
            if file_info.get('last_processed'):
                st.session_state.doc_processed = True
                st.session_state.processing_status = False
                st.session_state.processing_stats = file_info.get('processing_stats', {})
                st.sidebar.success("File already processed - ready for questions!")
            else:
                st.session_state.doc_processed = False
                st.session_state.processing_status = False
                st.sidebar.info("File found but needs processing")

        else:
            # New file upload
            if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
                st.session_state.current_file = uploaded_file.name
                st.session_state.doc_processed = False
                st.session_state.processing_status = False
                st.session_state.file_hash = file_hash

                file_path = save_uploaded_file(uploaded_file, file_hash)
                if file_path:
                    st.session_state.file_path = file_path
                    st.sidebar.success(f"File uploaded successfully: {uploaded_file.name}")

        # Show process button if needed (disabled during processing)
        if not st.session_state.processing_status and not st.session_state.doc_processed:
            if st.sidebar.button("Process Document",
                                 type="primary",
                                 disabled=disabled):
                st.session_state.processing_status = True

    # Display current file info
    if hasattr(st.session_state, 'current_file'):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Current Document")
        st.sidebar.info(f"📄 {st.session_state.current_file}")

        if st.session_state.get('doc_processed', False):
            st.sidebar.success("✅ Processing completed")
        elif st.session_state.get('processing_status', False):
            st.sidebar.warning("⏳ Processing...")
        else:
            st.sidebar.warning("⚠️ Not processed")


def qa_interface():
    """Question and answer interface with clear history feature"""
    st.markdown("### Ask a Question")

    # Create columns for input and submit button
    cols = st.columns([6, 1])

    with cols[0]:
        question = st.text_input(
            "",
            placeholder="e.g., 我需要对比不同模型在三个数据集（Boston、Energy、Yacht）的性能，请给出建议和示意图",
            key="question_input",
            label_visibility="collapsed"
        )

    with cols[1]:
        st.markdown("""
            <style>
            div.stButton > button {
                height: 42px;
                padding-top: 0px;
                padding-bottom: 0px;
            }
            </style>
            """, unsafe_allow_html=True)
        submit_button = st.button("Submit", type="primary", use_container_width=True)

    # Create a key for the current question in session state
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None

    # Create a key for the response container in session state
    if 'response_container' not in st.session_state:
        st.session_state.response_container = st.container()

    if submit_button and question:
        # Check if this is a new question
        if question != st.session_state.current_question:
            # Clear previous response by creating a new container
            st.session_state.response_container = st.container()
            st.session_state.current_question = question

            # Process the question in the new container
            with st.session_state.response_container:
                with st.spinner("Processing your question..."):
                    try:
                        result = st.session_state.rag.ask_question(question)

                        # Display images first if any
                        if result.get("images") or result.get('image_summaries'):
                            st.markdown("#### Retrieved Images")
                            for idx, (image_path, summary) in enumerate(zip(
                                    result.get("images", []),
                                    result.get("image_summaries", [])
                            )):
                                cols = st.columns([1, 1])
                                with cols[0]:
                                    if os.path.exists(image_path):
                                        st.image(
                                            image_path,
                                            caption=f"Image {idx + 1}",
                                            use_column_width=True
                                        )
                                    else:
                                        st.error(f"Image not found: {image_path}")
                                with cols[1]:
                                    st.markdown(f"**Image Summary:**\n{summary}")

                            # Add some spacing
                            st.markdown("---")

                        # Then display the answer
                        st.markdown("#### Answer")
                        st.write(result["answer"])

                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        st.code(traceback.format_exc())
    elif submit_button:
        st.warning("Please enter a question.")


def display_processing_results():
    """Display document processing results"""
    if (st.session_state.get('processing_status', False) and
            not st.session_state.get('doc_processed', False) and
            st.session_state.rag is not None and
            hasattr(st.session_state, 'file_path')):

        st.markdown("### Processing Document")
        with st.spinner("Processing document... This may take a few minutes."):
            stats = process_document(st.session_state.rag, st.session_state.file_path)
            if stats is not None:
                st.session_state.doc_processed = True
                st.session_state.processing_status = False
                st.session_state.processing_stats = stats

                # Update metadata with processing status
                if hasattr(st.session_state, 'file_hash'):
                    update_processing_status(st.session_state.file_hash, stats)

                st.success("Document processed successfully!")
            else:
                st.session_state.processing_status = False
                st.error("Failed to process document. Please try again.")

    # Always display stats if they exist and document is processed
    if st.session_state.get('doc_processed', False) and hasattr(st.session_state, 'processing_stats'):
        st.markdown("### Document Analysis Results")
        stats = st.session_state.processing_stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Text Elements", stats.get("text_elements", 0))
        with col2:
            st.metric("Table Elements", stats.get("table_elements", 0))
        with col3:
            st.metric("Image Elements", stats.get("image_elements", 0))
        with col4:
            st.metric("Image Summaries", stats.get("image_summaries", 0))


def main():
    st.set_page_config(
        page_title="Multimodal RAG System",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
            <h1 style='text-align: center; color: #2e4053;'>
                Multimodal RAG System
            </h1>
            """, unsafe_allow_html=True)

    # Initialize RAG system
    if 'rag' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag = initialize_rag()
            if st.session_state.rag is not None:
                st.sidebar.success("RAG system initialized successfully!")

    # Initialize session state variables
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = False
    if 'doc_processed' not in st.session_state:
        st.session_state.doc_processed = False

    # Handle file upload in sidebar
    sidebar_upload()

    # Main content area
    if not hasattr(st.session_state, 'current_file'):
        st.info("👈 Please upload a document using the sidebar")
        st.stop()

    # Display processing results (if any)
    display_processing_results()

    # Add separator before QA interface
    if st.session_state.get('doc_processed', False):
        st.markdown("---")
        # Clear the response container when switching documents
        if 'previous_file' not in st.session_state or st.session_state.previous_file != st.session_state.current_file:
            if 'response_container' in st.session_state:
                st.session_state.response_container = st.container()
            st.session_state.previous_file = st.session_state.current_file
            st.session_state.current_question = None

        qa_interface()


if __name__ == "__main__":
    main()