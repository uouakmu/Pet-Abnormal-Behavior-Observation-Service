import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from minio import Minio
from minio.error import S3Error

# --- MongoDB Configuration ---
# Use environment variable or fallback to localhost for direct running
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
mongo_client = None
db = None

def init_mongodb():
    global mongo_client, db
    try:
        mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        # Check connection
        mongo_client.admin.command('ping')
        print("Connected to MongoDB successfully.")
        
        # Select database
        db = mongo_client["pet_observation_db"]
    except ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")

def get_db():
    if db is None:
        init_mongodb()
    return db

# --- MinIO Configuration ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

minio_client = None
DAILY_BEHAVIOR_BUCKET = "daily-behavior-clips"
USER_PROFILE_BUCKET = "user-profiles"

def init_minio():
    global minio_client
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False # Set to True if using HTTPS
        )
        
        # Ensure buckets exist
        buckets_to_create = [DAILY_BEHAVIOR_BUCKET, USER_PROFILE_BUCKET]
        for bucket in buckets_to_create:
            if not minio_client.bucket_exists(bucket):
                minio_client.make_bucket(bucket)
                print(f"Created MinIO bucket: {bucket}")
            else:
                print(f"MinIO bucket '{bucket}' already exists.")
                
        print("Connected to MinIO successfully.")
    except S3Error as e:
        print(f"MinIO Error: {e}")
    except Exception as e:
        print(f"Could not connect to MinIO: {e}")

def get_minio_client():
    if minio_client is None:
        init_minio()
    return minio_client
