from waitress import serve
from main import app
from loguru import logger

if __name__ == '__main__':
    logger.info("Starting Waitress server...")
    serve(app, host='0.0.0.0', port=5000)
    logger.info("Waitress server stopped.")