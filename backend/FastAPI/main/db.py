import os
from minio import Minio
from minio.error import S3Error

# --- MinIO Configuration ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

minio_client = None
DAILY_BEHAVIOR_BUCKET = "daily-behavior-clips"
USER_PROFILE_BUCKET = "user-profiles"

import json

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
                
            # Set public read policy for the buckets so Flutter can fetch images
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": ["s3:GetBucketLocation", "s3:ListBucket"],
                        "Resource": [f"arn:aws:s3:::{bucket}"]
                    },
                    {
                        "Effect": "Allow",
                        "Principal": "*",
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{bucket}/*"]
                    }
                ]
            }
            minio_client.set_bucket_policy(bucket, json.dumps(policy))
                
        print("Connected to MinIO successfully and applied public policies.")
    except Exception as e:
        print(f"MinIO Initialization Error: {e}")

def get_minio_client():
    if minio_client is None:
        init_minio()
    return minio_client
