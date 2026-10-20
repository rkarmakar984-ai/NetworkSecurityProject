import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch the URI safely from the .env file
uri = os.getenv("MONGO_DB_URL")

client = MongoClient(uri)

try:
    client.admin.command("ping")
    print("Connected successfully using environment variables!")
    client.close()
except Exception as e:
    raise Exception(f"The following error occurred: {e}")