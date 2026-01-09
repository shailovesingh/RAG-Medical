import os
from app.components.pdf_loader import load_pdf_files, create_text_chunks

from app.components.vector_store import save_vector_store
from app.config.config import DB_FAISS_PATH

from app.components.logger import get_logger
from app.common.custom_exceptions import CustomException

logger = get_logger(__name__)

def process_and_store_pdfs():
    try:
        logger.info("Making the vectorestore...")

        documents = load_pdf_files()
        
        text_chunks = create_text_chunks(documents)

        save_vector_store(text_chunks)

        logger.info("vectorstore created successfully.")

    except Exception as e:
        error_message = CustomException("Failed to create vectorstore")
        logger.error(str(error_message))

if __name__ == "__main__":
    process_and_store_pdfs()
