# db_utils.py
from pymongo import MongoClient
import urllib.parse

db_username = 'XXXX'
db_password = 'XXXXX'

# URL encode your username and password
username = urllib.parse.quote_plus(db_username)
password = urllib.parse.quote_plus(db_password)

# Connection string
connection_url = f"mongodb+srv://{username}:{password}@cluster0.9zjej.mongodb.net/your_database_name?retryWrites=true&w=majority"

def get_db_connection():
    try:
        client = MongoClient(connection_url)
        database = client['air_quality']
        return database
    except Exception as e:
        print(f"Database connection error: {e}")
        return None
