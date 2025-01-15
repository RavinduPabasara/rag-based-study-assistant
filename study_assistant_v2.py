import os
from typing import List, Dict, Union, Optional
from pathlib import Path
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain.prompts import PromptTemplate
import streamlit as st
import tempfile
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class DocumentProcessor:
    """Handle different types of document processing including handwritten notes."""

    def __init__(self):
        """Initialize document processor with necessary configurations."""
        # Configure Tesseract path if needed
        if os.getenv('TESSERACT_PATH'):
            pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

        # Configure OpenCV parameters for image preprocessing
        self.image_processing_params = {
            'blur_kernel': (5, 5),
            'threshold_block_size': 11,
            'threshold_constant': 2
        }

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better OCR results.

        Args:
            image: Input image as numpy array
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.image_processing_params['threshold_block_size'],
                self.image_processing_params['threshold_constant']
            )

            # Denoise
            denoised = cv2.medianBlur(binary, 3)

            # Dilation to connect broken text
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(denoised, kernel, iterations=1)

            return dilated
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def process_handwritten_image(self, image_path: str) -> str:
        """
        Process handwritten notes from images using OCR.

        Args:
            image_path: Path to the image file
        Returns:
            Extracted text from the image
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Perform OCR with custom configuration
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(processed_image, config=custom_config)

            if not text.strip():
                logger.warning(f"No text extracted from image: {image_path}")
                return "No text could be extracted from this image."

            return text.strip()

        except Exception as e:
            logger.error(f"Error processing handwritten image {image_path}: {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF files with enhanced error handling.

        Args:
            pdf_path: Path to the PDF file
        Returns:
            Extracted text content
        """
        try:
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                if not pdf.pages:
                    raise ValueError("PDF file contains no pages")

                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        text_content.append(text.strip())

            if not text_content:
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return "No text could be extracted from this PDF."

            return "\n\n".join(text_content)

        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise


class StudyAssistant:
    def __init__(self, openai_api_key: str):
        """Initialize Study Assistant with enhanced capabilities."""
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")

        self.openai_api_key = openai_api_key
        self.doc_processor = DocumentProcessor()

        # Initialize embeddings with error handling
        try:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)  # Use openai_api_key instead of api_key
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
            raise

        # Configure advanced text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Initialize language model with advanced configuration
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-4o",  # Use correct model name
            openai_api_key=openai_api_key,  # Use openai_api_key consistently
            request_timeout=60,
            max_retries=3
        )

        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )

        self.vector_store = None
        self.qa_chain = None
        self.processed_docs_cache = {}

    def process_file(self, file_path: str, file_type: str) -> str:
        """
        Process different types of files with appropriate handlers.

        Args:
            file_path: Path to the file
            file_type: Type of file ('pdf', 'image', etc.)
        Returns:
            Extracted text content
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path in self.processed_docs_cache:
            return self.processed_docs_cache[file_path]

        try:
            if file_type == 'pdf':
                text = self.doc_processor.extract_text_from_pdf(file_path)
            elif file_type in ['jpg', 'jpeg', 'png']:
                text = self.doc_processor.process_handwritten_image(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            if not text or text.isspace():
                raise ValueError(f"No text could be extracted from {file_path}")

            self.processed_docs_cache[file_path] = text
            return text

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    def process_documents(self, text_content: str):
        """Process documents with enhanced error handling and chunking."""
        if not text_content or text_content.isspace():
            raise ValueError("No text content to process")

        try:
            # Split text into chunks
            texts = self.text_splitter.split_text(text_content)

            if not texts:
                raise ValueError("No text chunks created during splitting")

            # Create vector store with progress tracking
            self.vector_store = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=[{"chunk": i} for i in range(len(texts))]
            )

            if not self.vector_store:
                raise ValueError("Failed to create vector store")

            # Initialize QA chain with custom prompt
            template = """
            Use the following context to answer the question. If you cannot find
            the answer in the context, say so clearly and suggest related questions
            that might be more relevant.

            Context: {context}

            Question: {question}

            Answer: Let me help you understand this topic clearly and thoroughly.
            """

            CUSTOM_PROMPT = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance
                    search_kwargs={"k": 5}
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
                return_source_documents=True
            )

            logger.info("Successfully processed documents and created QA chain")

        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def generate_memory_map(self, topic: str) -> Dict[str, Union[str, List[str]]]:
        """
        Generate an enhanced memory map with citations and confidence scores.

        Args:
            topic: The topic to create a memory map for
        Returns:
            Dictionary containing the memory map and metadata
        """
        if not self.qa_chain:
            return {"error": "Please upload documents first."}

        if not topic or topic.isspace():
            return {"error": "Please provide a valid topic."}

        prompt = f"""
        Create a comprehensive memory map for: {topic}

        Structure the response as follows:
        1. Core Concepts (with examples and relationships)
        2. Key Relationships (cause-effect, dependencies)
        3. Practical Applications (real-world examples)
        4. Study Focus Points (important areas, common test topics)
        5. Common Misconceptions and Clarifications
        6. Related Topics and Prerequisites

        For each point, provide specific references to the source material
        and indicate confidence level in the information.
        """

        try:
            response = self.qa_chain({"question": prompt})

            if not response or 'answer' not in response:
                raise ValueError("Failed to generate memory map response")

            return {
                "content": response['answer'],
                "sources": [doc.page_content for doc in response.get('source_documents', [])],
                "timestamp": datetime.now().isoformat(),
                "topic": topic
            }

        except Exception as e:
            logger.error(f"Error generating memory map: {str(e)}")
            return {"error": f"Failed to generate memory map: {str(e)}"}


def create_streamlit_app():
    """Create an enhanced Streamlit interface with additional features."""
    st.set_page_config(page_title="Advanced Study Assistant", layout="wide")

    st.title("Advanced Study Assistant - Memory Map Generator")

    # Secure API key handling
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    if api_key:
        try:
            study_assistant = StudyAssistant(api_key)

            # File upload section with multiple file types
            st.subheader("Upload Your Study Materials")
            uploaded_files = st.file_uploader(
                "Upload PDFs or images of handwritten notes",
                type=['pdf', 'jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )

            if uploaded_files:
                with st.spinner('Processing your documents... Please wait.'):
                    all_text = []

                    # Process files in parallel
                    with ThreadPoolExecutor() as executor:
                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False,
                                                             suffix=f'.{uploaded_file.type.split("/")[-1]}') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                file_type = uploaded_file.type.split('/')[-1]

                                try:
                                    text = study_assistant.process_file(tmp_file.name, file_type)
                                    if text and not text.isspace():
                                        all_text.append(text)
                                except Exception as e:
                                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                                finally:
                                    os.unlink(tmp_file.name)

                    if not all_text:
                        st.error("No text could be extracted from any of the uploaded files.")
                        return

                    combined_text = "\n\n".join(all_text)
                    study_assistant.process_documents(combined_text)
                    st.success("✅ Documents processed successfully!")

            # Memory map generation with advanced options
            st.subheader("Generate Memory Map")
            col1, col2 = st.columns(2)

            with col1:
                topic = st.text_input("Enter your topic:")

            with col2:
                detail_level = st.select_slider(
                    "Select detail level:",
                    options=["Basic", "Intermediate", "Advanced"]
                )

            if topic:
                with st.spinner('Creating your memory map...'):
                    memory_map = study_assistant.generate_memory_map(topic)

                    if "error" in memory_map:
                        st.error(memory_map["error"])
                    else:
                        st.markdown(memory_map["content"])

                        if memory_map.get("sources"):
                            with st.expander("View Sources"):
                                for i, source in enumerate(memory_map["sources"], 1):
                                    st.text(f"Source {i}:\n{source[:200]}...")

            # Interactive Q&A section
            st.subheader("Ask Questions About Your Content")
            question = st.text_input("What would you like to know about your materials?")

            if question:
                with st.spinner('Finding your answer...'):
                    try:
                        response = study_assistant.qa_chain({"question": question})
                        st.markdown(response["answer"])

                        if response.get("source_documents"):
                            with st.expander("View Related Context"):
                                for i, doc in enumerate(response["source_documents"], 1):
                                    st.text(f"Reference {i}:\n{doc.page_content[:200]}...")
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

        except Exception as e:
            st.error(f"An error occurred initializing the application: {str(e)}")
            logger.error(f"Application error: {str(e)}")

            # Add session state management for better user experience
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()

        if 'last_error' not in st.session_state:
            st.session_state.last_error = None

if __name__ == "__main__":
    try:
        create_streamlit_app()
    except Exception as e:
        st.error("Fatal application error. Please refresh the page and try again.")
        logger.critical(f"Fatal application error: {str(e)}")