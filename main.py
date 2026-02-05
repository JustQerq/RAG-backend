import os
from flask import Flask, jsonify, request
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

from rag import RAGManager, DatabaseManager

app = Flask(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
db_manager = DatabaseManager(MONGO_URI, db_name='rag_discord')
rag_manager = RAGManager(db_manager)

@app.route('/')
def home():
    return "Welcome to the Flask MongoDB API!"


@app.route('/rag-query', methods=['POST'])
def generate_answer():
    data = request.json
    if not data:
        return jsonify({'error': 'no data provided'}), 400
    
    try:
        session_id = int(data['session_id'])
        user_query = str(data['query'])
        verbose = bool(data['verbose']) if isinstance(data['verbose'], bool) else data['verbose'].lower() == 'true'
    except (KeyError, ValueError, AttributeError):
        return jsonify({'error': 'invalid data types'}), 400
    
    result = rag_manager.generate_answer(session_id, user_query, verbose)
    return jsonify(result), 200


@app.route('/add', methods=['POST'])
def add_document():
    data = request.json
    if not data:
        return jsonify({'error': 'no data provided'}), 400
    
    inserted_ids = db_manager.add_documents(data['documents'], collection_name=data['collection'])
    return jsonify({'message': 'documents successfully added', 'ids': inserted_ids}), 201

if __name__ == '__main__':
    app.run(debug=True)