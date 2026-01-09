from langchain_hugginface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embedding_model():
    try:
        logger.info("Intializing Our Hugginface embedding model")

        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        logger.info("Successfully Intialized Hugginface embedding model")
        return model
    
    except Exception as e:
        error_message = CustomException("Failed to load Hugginface embedding model", e)
        logger.error(str(error_message))
        return None