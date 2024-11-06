import logging

logger = logging.Logger("gen-ai")
logger.setLevel("INFO")
logger.addHandler(logging.StreamHandler())

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level="INFO",
)
