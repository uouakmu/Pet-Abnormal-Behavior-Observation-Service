"""
Test script: Saves dummy behavior/emotion/sound data to Firebase RTDB
and generates an LLM diary via llm_diary.generate_daily_diary().
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env if present
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"[env] Loaded .env from {env_path}")
    else:
        print("[env] No .env found — rely on system env vars")
except ImportError:
    print("[env] python-dotenv not installed, using system env vars")

import firebase_admin
from firebase_admin import credentials, db as firebase_db
from datetime import datetime

KEY_PATH = r"c:\Users\hsj20\OneDrive\문서\GitHub\Pet-Abnormal-Behavior-Observation-Service\backend\FastAPI\key\testApi.json"

if not firebase_admin._apps:
    cred = credentials.Certificate(KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://test-25cac-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })

USER_ID   = "test_user_diary"
PET_TYPE  = "dog"
TODAY     = datetime.now().strftime("%Y-%m-%d")
NOW_ISO   = datetime.now().isoformat()

DUMMY_ANALYSIS = {
    "status": "success",
    "behavior_analysis": {
        "detected_behavior": "walking",
        "emotion": "happy"
    },
    "audio_analysis": {
        "detected_sound": "barking"
    },
    "patella_analysis": {
        "status": "Normal"
    }
}

print(f"[1] Saving behavior/emotion/sound data to Firebase...")
print(f"    Path: users/{USER_ID}/day/{TODAY}/{{push_key}}")

ref = firebase_db.reference(f'users/{USER_ID}/day/{TODAY}')
doc = {
    "user_id": USER_ID,
    "pet_type": PET_TYPE,
    "timestamp": NOW_ISO,
    "video_url": "",
    "analysis_result": DUMMY_ANALYSIS
}
pushed = ref.push(doc)
print(f"    [OK] Saved! key={pushed.key}, userId={USER_ID}, date={TODAY}")

print()
print(f"[2] Generating LLM diary via Groq API...")
groq_key = os.getenv("GROQ_API_KEY")
print(f"    GROQ_API_KEY present: {bool(groq_key)}")

try:
    from FastAPI.main.llm_diary import generate_daily_diary
    diary = generate_daily_diary(USER_ID, TODAY)
    print()
    print("=" * 60)
    print("[OK] Generated diary:")
    print("=" * 60)
    print(diary)
    print("=" * 60)
except Exception as e:
    print(f"    [ERR] Diary generation error: {e}")
    import traceback; traceback.print_exc()

print()
print(f"[3] Verifying Firebase: users/{USER_ID}/LLM_diary/{TODAY}")
diary_ref = firebase_db.reference(f'users/{USER_ID}/LLM_diary/{TODAY}')
saved = diary_ref.get()
if saved and isinstance(saved, dict):
    content = saved.get('content', '')
    print(f"    [OK] Diary saved! First 80 chars: {content[:80]}...")
else:
    print("    [WARN] Diary not found in Firebase under LLM_diary key.")
    print(f"    Raw: {saved}")
