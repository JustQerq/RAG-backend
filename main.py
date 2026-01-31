from flask import Flask, jsonify, request
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))

db = client['DiscordBot']
collection = db['tf2']

@app.route('/')
def home():
    return "Welcome to the Flask MongoDB API!"

@app.route('/add', methods=['POST'])
def add_document():
    data = request.json
    if not data:
        return jsonify({'error': 'no data provided'}), 400
    result = collection.insert_one(data)
    return jsonify({'message': 'document inserted', 'id': str(result.inserted_id)}), 201

@app.route('/documents', methods=['GET'])
def get_documents():
    documents = collection.find({}, {'_id': 0}).to_list()
    return jsonify(documents), 200

if __name__ == '__main__':
    app.run(debug=True)