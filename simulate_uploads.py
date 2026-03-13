import os
import cv2
import requests
import time
import datetime

# --- CONFIGURATION ---
VIDEO_PATH = "sample2.mp4" 
API_BASE = "http://localhost:8080"
API_UPLOAD = f"{API_BASE}/api/daily-behavior"
USER_ID = "admin"  
PASSWORD = "1234"
PET_TYPE = "dog"   
NUM_CHUNKS = 24    # Number of simulation uploads (4 days * 6 chunks per day)
# ---------------------

def register_user():
    print(f"Ensuring user {USER_ID} exists...")
    try:
        requests.post(f"{API_BASE}/signup/", json={'user_id': USER_ID, 'password': PASSWORD})
        requests.post(f"{API_BASE}/user-input/{USER_ID}", json={
            "pet_name": "바둑이", "pet_type": "강아지", "pet_gender": "남아", "pet_birthday": "2020-01-01"
        })
    except Exception as e:
        print(f"Warning on user creation: {e}")

def prepare_clips():
    """Extracts 24 unique 15-second clips evenly distributed across the video."""
    print(f"Opening video: {VIDEO_PATH} to extract {NUM_CHUNKS} unique 15s clips...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open {VIDEO_PATH}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Source video duration: {duration:.2f} seconds")

    # Calculate stride so 24 clips fit exactly within (duration - 15) seconds
    stride = max(1, (duration - 15) / max(1, NUM_CHUNKS - 1))
    
    clip_names = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i in range(NUM_CHUNKS):
        clip_name = f"chunk_sim_{i:03d}.mp4"
        clip_names.append(clip_name)
        
        if os.path.exists(clip_name):
            continue
            
        start_time = i * stride
        start_frame = int(start_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out = cv2.VideoWriter(clip_name, fourcc, fps, (width, height))
        
        frames_to_read = int(fps * 15)
        print(f"Extracting {clip_name} (from {start_time:.1f}s)...")
        for _ in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
        out.release()
        
    cap.release()
    print("Clip extraction complete.")
    return clip_names

def simulate_historical_uploads():
    #register_user()
    clips = prepare_clips()
    if not clips:
        return
        
    print(f"\nStarting {NUM_CHUNKS} distinct chunk uploads for 4 days...")

    # For 24 chunks spanning 4 days: 6 chunks per day
    for chunk_idx, clip_name in enumerate(clips):
        # Calculate historical time (4 days back to today)
        days_ago = 3 - (chunk_idx // 6)
        chunk_time = datetime.datetime.now() - datetime.timedelta(days=days_ago)
        
        # Space them 2 hours apart starting from 10:00 AM
        hour_offset = 10 + (chunk_idx % 6) * 2
        chunk_time = chunk_time.replace(hour=hour_offset if hour_offset < 24 else 23, minute=0, second=0, microsecond=0)

        print(f"[{chunk_idx+1}/{NUM_CHUNKS}] Uploading {clip_name} for timestamp: {chunk_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            with open(clip_name, 'rb') as f:
                files = {'file': (clip_name, f, 'video/mp4')}
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
            print(f"Error uploading chunk {chunk_idx}: {e}")
            
        time.sleep(1) # delay to not overwhelm server
        
    print("\n✅ Simulation complete! You can now check the Flutter app photo gallery.")

if __name__ == "__main__":
    simulate_historical_uploads()

