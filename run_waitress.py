from waitress import serve
from main import app

if __name__ == '__main__':
    print("Starting Waitress server...")
    serve(app, host='127.0.0.1', port=5000)
    print("Waitress server stopped.")