import os
import cv2
import requests
import time
import datetime
import argparse

# --- CONFIGURATION (DEFAULTS) ---
VIDEO_PATH = "sample1.mp4" 
API_BASE = "http://localhost:8080"
API_UPLOAD = f"{API_BASE}/api/daily-behavior"
# ---------------------

def get_args():
    parser = argparse.ArgumentParser(description="Simulate pet analysis uploads.")
    parser.add_argument("--user", type=str, default="admin", help="User ID")
    parser.add_argument("--password", type=str, default="1234", help="Password")
    parser.add_argument("--pet", type=str, default="dog", help="Pet type (dog/cat)")
    parser.add_argument("--video", type=str, default=VIDEO_PATH, help="Path to sample video")
    return parser.parse_args()

args = get_args()
USER_ID = args.user
PASSWORD = args.password
PET_TYPE = args.pet
VIDEO_PATH = args.video
NUM_CHUNKS = 24


def register_user():
    print(f"Ensuring user {USER_ID} exists...")
    try:
        requests.post(f"{API_BASE}/signup/", json={'user_id': USER_ID, 'password': PASSWORD})
        requests.post(f"{API_BASE}/user-input/{USER_ID}", json={
            "pet_name": "바둑이", "pet_type": "강아지", "pet_gender": "남아", "pet_birthday": "2020-01-01"
        })
    except Exception as e:
        print(f"Warning on user creation: {e}")

def prepare_images():
    """Extracts 24 unique frames evenly distributed across the video."""
    print(f"Opening video: {VIDEO_PATH} to extract {NUM_CHUNKS} unique frames...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source video frames: {total_frames}")

    # Calculate stride to get 24 frames
    stride = max(1, total_frames // NUM_CHUNKS)
    
    image_names = []
    for i in range(NUM_CHUNKS):
        image_name = f"frame_sim_{i:03d}.jpg"
        image_names.append(image_name)
        
        if os.path.exists(image_name):
            continue
            
        frame_idx = i * stride
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"Extracting {image_name} (frame {frame_idx})...")
        cv2.imwrite(image_name, frame)
            
    cap.release()
    print("Frame extraction complete.")
    return image_names

def simulate_historical_uploads():
    #register_user()
    images = prepare_images()
    if not images:
        return
        
    print(f"\nStarting {NUM_CHUNKS} distinct image uploads for today...")

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    for chunk_idx, image_name in enumerate(images):
        # Space them 1 hour apart for TODAY
        chunk_time = datetime.datetime.now().replace(hour=chunk_idx % 24, minute=0, second=0, microsecond=0)

        print(f"[{chunk_idx+1}/{NUM_CHUNKS}] Uploading {image_name} for timestamp: {chunk_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            with open(image_name, 'rb') as f:
                # Backend now handles images in /api/daily-behavior
                files = {'file': (image_name, f, 'image/jpeg')}
                data = {
                    'user_id': USER_ID,
                    'pet_type': PET_TYPE,
                    'timestamp': chunk_time.isoformat()
                }
                
                response = requests.post(API_UPLOAD, data=data, files=files)
                
                if response.status_code == 200:
                    resp_data = response.json()
                    status = resp_data.get('status')
                    if status == "success":
                        ai_info = resp_data.get('ai_inference', {}).get('summary', '')
                        print(f" -> AI Result: {ai_info}")
                    else:
                        print(f" -> Backend Warning: {resp_data.get('message')}")
                else:
                    print(f" -> Upload Failed: HTTP {response.status_code}")
                    print(response.text)
                    
        except Exception as e:
            print(f"Error uploading image {chunk_idx}: {e}")
            
        time.sleep(0.5) # small delay
        
    print("\n--- Triggering LLM Diary Generation ---")
    try:
        diary_url = f"{API_BASE}/api/daily-diary/{USER_ID}?date={today}"
        print(f"Calling: {diary_url}")
        response = requests.get(diary_url)
        if response.status_code == 200:
            diary_data = response.json()
            if diary_data.get("status") == "success":
                print("\n--- Generated Diary ---\n")
                print(diary_data.get("diary"))
            else:
                print(f"Diary Error: {diary_data.get('message')}")
        else:
            print(f"Diary API Failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error triggering diary: {e}")

    print("\nSimulation complete! You can now check the Flutter app.")

if __name__ == "__main__":
    simulate_historical_uploads()

