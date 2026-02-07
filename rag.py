import os
import json
import time
from datetime import datetime
from utils import create_index, check_index_ready
from pymongo import MongoClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, List
from voyageai.client import Client as VoClient
import voyageai.error as error
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger


class RAGManager:
    def __init__(self, db_manager, document_collection_name='articles', history_collection_name='chat_history') -> None:
        self.voyage_client = VoClient()
        self.openai_client = OpenAI()
        self.db_manager = db_manager
        self.document_collection_name = document_collection_name
        self.history_collection_name = history_collection_name
    
    def add_documents(self, documents: List[Dict], collection_name: str):
        embedded_docs = []
        for doc in documents:
            chunks = self.embed_document(doc)
            embedded_docs.extend(chunks)
        inserted_ids = self.db_manager.add_documents(embedded_docs, collection_name)
        return inserted_ids
        
    def get_chunks(self, doc: Dict, text_field: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4", 
            chunk_size=200, 
            chunk_overlap=0
            )
        text = doc[text_field]
        chunks = text_splitter.split_text(text)
        return chunks
    
    def get_embeddings(self, content: List[str], input_type: str):
        logger.info(f"Generating embeddings for a {input_type}")
        try:
            embds_obj = self.voyage_client.contextualized_embed(
                inputs=[content], 
                model='voyage-context-3', 
                input_type=input_type,
                )
        except error.APIConnectionError as e:
            logger.error(f"Error connecting to embedding API: {e}")
        
        if input_type == "document":
            embeddings = [emb for r in embds_obj.results for emb in r.embeddings]
        if input_type == "query":
            embeddings = embds_obj.results[0].embeddings[0]
        return embeddings
    
    def embed_document(self, document: Dict) -> List[Dict]:
        embedded_chunks = []
        chunks = self.get_chunks(document, 'body')
        chunk_embeddings = self.get_embeddings(chunks, 'document')
        for chunk, embedding in zip(chunks, chunk_embeddings):
            document_chunk = document.copy()
            document_chunk['body'] = chunk
            document_chunk['embedding'] = embedding
            embedded_chunks.append(document_chunk)
        return embedded_chunks
    
    def store_chat_message(self, session_id: int, role: str, content: str):
        message = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        message_ids = self.db_manager.add_documents([message], self.history_collection_name)
        
        return message_ids[0]
    
    def generate_answer(self, session_id: int, user_query: str, verbose: bool = False):
        messages = []
        system_prompt = "Answer the questions based only on the provided context. If the context is empty, say I don't know"
        messages.append({"role": "developer", "content": system_prompt})
        
        user_query_embedding = self.get_embeddings([user_query], "query")
        logger.info(f"Performing vector search for query='{user_query}', session_id={session_id}")
        context = self.db_manager.vector_search(self.document_collection_name, user_query_embedding)
        context_str = "\n\n".join([chunk.get("body", "") for chunk in context])
        messages.append({"role": "user", "content": context_str})
        
        message_history = self.db_manager.retrieve_session_history(session_id, self.history_collection_name)
        messages.extend(message_history)
        
        user_message = {"role": "user", "content": user_query}
        messages.append(user_message)
        
        logger.info("Sending augmented query to OpenAI LLM")
        try:
            response = self.openai_client.responses.create(
                model="gpt-5-nano",
                input=messages
            )
            logger.info("Received a response from LLM")
        except Exception as e:
            logger.error("Error getting a response from LLM: {e}")
        
        self.store_chat_message(session_id, "user", user_query)
        answer_id = self.store_chat_message(session_id, "assistant", response.output_text)
        
        result = {'answer_id': answer_id, 'answer': response.output_text}
        if verbose:
            result['chat_history'] = message_history
            result['context'] = context_str
        
        return result
        

class DatabaseManager:
    def __init__(self, mongodb_uri, db_name='rag_discord'):
        self.db_client = MongoClient(mongodb_uri)
        self.db = self.db_client[db_name]
    
    def add_documents(self, documents: List[Dict], collection_name: str):
        collection = self.db[collection_name]
        result = collection.insert_many(documents)
        return result.inserted_ids
    
    def create_vector_search_index(self, collection_name, index_name='vector_index'):
        model = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024,
                        "similarity": "cosine"
                    }
                ]
            }
        }
        create_index(self.db[collection_name], index_name, model)
    
    def check_index_ready(self, collection_name, index_name='vector_index'):
        return check_index_ready(self.db[collection_name], index_name)
    
    def vector_search(self, collection_name, query_embedding, index_name='vector_index'):
        while True:
            index_ready, _ = self.check_index_ready(collection_name, index_name)
            if not index_ready:
                time.sleep(5)
                continue
            else:
                break
            
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 20,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "topic": 1,
                    "body": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = self.db[collection_name].aggregate(pipeline)
        return results.to_list()
    
    def retrieve_session_history(self, session_id: int, collection_name: str):
        logger.info(f"Retrieving chat history for session_id={session_id}")
        try:
            cursor = self.db[collection_name].find({"session_id": session_id}).sort("timestamp", 1)
        except Exception as e:
            logger.error(f"Error retrieving chat history: {e}")
        
        if cursor:
            messages = [{"role": message['role'], "content": message['content']} for message in cursor]
        else:
            messages = []
        
        return messages


if __name__ == "__main__":
    load_dotenv()
    MONGODB_URI = os.getenv("MONGO_URI")
    db_manager = DatabaseManager(mongodb_uri=MONGODB_URI)
    rag = RAGManager(db_manager)
    
    with open("dataset\\clean_data\\S08\\questions.json", 'r') as f:
        questions = json.load(f)
    
    question = questions[10]
    query = f"This question is about {question['topic']}. {question['question']}"
    rag_result = rag.generate_answer(4, query, verbose=True)
    print("\n-------------Chat history-------------\n")
    print(rag_result['chat_history'])
    print("\n-------------User query-------------\n")
    print(query)
    print("\n-------------Context-------------\n")
    print(rag_result['context'])
    print("\n-------------Answer-------------\n")
    print(rag_result['answer'])