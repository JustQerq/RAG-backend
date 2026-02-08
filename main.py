import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import dns
# from dotenv import load_dotenv
from loguru import logger
from prometheus_flask_exporter import PrometheusMetrics

from rag import RAGManager, DatabaseManager


log_folder = "logs"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
logger.add(f"{log_folder}/app_{{time}}.log", rotation="10 MB", retention="10 days", level="INFO", backtrace=True, diagnose=True)

app = Flask(__name__)
CORS(app)
metrics = PrometheusMetrics(app) # Metrics can be accessed via /metrics endpoint

# load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.critical("MongoDB connection string not present in environment!")
else:
    logger.info(f"Connecting to MongoDB database with {MONGO_URI} connection string")

db_manager = DatabaseManager(MONGO_URI, db_name='rag_discord')
rag_manager = RAGManager(db_manager)

@app.route('/')
def home():
    logger.info("Home endpoint accessed")
    return "Backend is in working order!"


@app.route('/rag-query', methods=['POST'])
def generate_answer():
    data = request.get_json()
    app.logger.info(f"Received RAG query request")
    if not data:
        logger.warning("No data provided in a rag-query request")
        return jsonify({'error': 'no data provided'}), 400
    app.logger.info(f"Request data: {data}")
    
    try:
        session_id = int(data['session_id'])
        user_query = str(data['query'])
        verbose = bool(data['verbose']) if isinstance(data['verbose'], bool) else data['verbose'].lower() == 'true'
    except (KeyError, ValueError, AttributeError) as e:
        logger.error(f"Error parsing rag-query request data: {e}")
        return jsonify({'error': 'invalid data types'}), 400
    
    logger.info(f"Generating answer for session_id={session_id}, query='{user_query}', verbose={verbose}")
    result = rag_manager.generate_answer(session_id, user_query, verbose)
    logger.info(f"Answer generated for session_id={session_id}, query='{user_query}', answer='{result['answer']}'")
    return jsonify(result), 200


@app.route('/feedback', methods=['PUT'])
def save_feedback():
    data = request.get_json()
    logger.info(f"Received feedback")
    if not data:
        logger.warning("No data provided in a feedback request")
        return jsonify({'error': 'no data provided'}), 400
    logger.info(f"Feedback data: {data}")
    
    try:
        answer_id = str(data['answer_id'])
        helpful = bool(data['helpful'])
    except (KeyError, ValueError, AttributeError) as e:
        logger.error(f"Error parsing feedback request data: {e}")
        return jsonify({'error': 'invalid data types'}), 400
    
    db_manager.save_feedback(answer_id, helpful, 'chat_history')
    
    return jsonify({'message': 'Feedback successfully saved'}), 200


@app.route('/add', methods=['POST'])
def add_document():
    data = request.get_json()
    if not data:
        logger.warning("No data provided in add document request")
        return jsonify({'error': 'no data provided'}), 400
    
    documents = data['documents']
    if not isinstance(documents, list):
        documents = [documents]
    inserted_ids = rag_manager.add_documents(documents, collection_name=data['collection'])
    logger.info(f"Inserted document chunks with ids: {inserted_ids}")
    
    return jsonify({'message': 'documents successfully added', 'ids': inserted_ids}), 201

if __name__ == '__main__':
    app.run(debug=True)