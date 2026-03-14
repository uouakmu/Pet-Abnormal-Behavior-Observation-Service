from fastapi import FastAPI, File, UploadFile, Form
from contextlib import asynccontextmanager
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from fastapi.middleware.cors import CORSMiddleware
import io
import datetime
import uuid

# AI Inference module import
from FastAPI.main.model_inference import ai_engine
# Minio DB Imports
from FastAPI.main.db import get_minio_client, DAILY_BEHAVIOR_BUCKET
from FastAPI.main.daily_behavior_inference import daily_behavior_engine

# LLM Diary & Statistics Imports
from FastAPI.main.llm_diary import generate_daily_diary, get_diary_list
from FastAPI.main.statistics import get_weekly_statistics

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load AI models
    print("Initializing AI Models...")
    ai_engine.load_models()
    
    # Initialize Daily Behavior Engine
    daily_behavior_engine.load_models()
    
    # Initialize DB Connections
    get_minio_client()
    
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# 1. 파이어베이스 초기화
import os as _os
_KEY_PATH = _os.path.join(_os.path.dirname(__file__), '..', 'key', 'testApi.json')
cred = credentials.Certificate(_os.path.abspath(_KEY_PATH))
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://test-25cac-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 도메인 허용
    allow_methods=["*"],
    allow_headers=["*"],
)

# 사용자 로그인
class User(BaseModel):
    user_id: str
    password: str

@app.post("/login/")
def login(user: User):
    try:
        ref = firebase_db.reference(f'users/{user.user_id}')
        user_data = ref.get()

        if not user_data:
            return {"status": "error", "message": "아이디가 존재하지 않습니다."}

        # 비밀번호 체크 (딕셔너리 또는 문자열 구조 모두 대응)
        fb_password = ""
        if isinstance(user_data, dict):
            fb_password = user_data.get('password')
        else:
            fb_password = user_data # 단순 문자열인 경우

        if fb_password == user.password:
            # Check if pet_info exists in Firebase
            has_pet_info = False
            if isinstance(user_data, dict):
                has_pet_info = 'pet_info' in user_data

            return {
                "status": "success",
                "message": "로그인 성공",
                "user_id": user.user_id,
                "has_pet_info": has_pet_info
            }
        else:
            return {"status": "error", "message": "아이디 또는 비밀번호가 틀렸습니다."}
            
    except Exception as e:
        print(f"Login error: {str(e)}")
        return {"status": "error", "message": f"서버 오류가 발생했습니다: {str(e)}"}

@app.post("/signup/")
async def signup(user: User):
    # 회원가입 (중복 체크 생략)
    ref = firebase_db.reference(f'users/{user.user_id}')
    # 이미 가입된 아이디인지 확인
    if ref.get() is not None:
        return {"status": "error", "message": "이미 존재하는 아이디입니다."}

    ref.set({
        "password": user.password
    })
    
    return {"status": "success"}

# 테스트 용
class Log(BaseModel):
    pet_name: str
    behavior: str # 예: "짖음", "이상 보행" 등
    timestamp: str

@app.post("/log/")
def save_log(log: Log):
    # 'pet_logs'라는 경로에 데이터 저장
    ref = firebase_db.reference('pet_logs')
    new_log_ref = ref.push(log.dict())
    return {"status": "success", "id": new_log_ref.key}
    
@app.get("/logs/")
def get_logs():
    ref = firebase_db.reference('pet_logs')
    return ref.get()

# 사용자 입력 : 반려동물 기본 정보 PetRegistrationPage
class PetInfo(BaseModel):
    pet_name: str
    pet_type: str
    pet_gender: str
    pet_birthday : str
    
@app.post("/user-input/{user_id}")
def save_pet_info(user_id: str, data: PetInfo):
    # 파이어베이스에 pet_info 저장
    ref = firebase_db.reference(f'users/{user_id}')
    ref.update({"pet_info": data.model_dump()})
    return {"status": "success", "user_id": user_id}

@app.get("/user-pet-info/{user_id}")
def get_all_pet_info(user_id: str):
    ref = firebase_db.reference(f'users/{user_id}/pet_info')
    pet_info = ref.get()

    if pet_info:
        return {
            "status": "success",
            "data": pet_info  # pet_name, pet_type, pet_gender, pet_birthday
        }
    else:
        return {"status": "error", "message": "반려동물 정보가 없습니다."}

# ─────────────────────────── 직접 로그 저장 (Flutter → Firebase) ───────────────────────────
class DirectLogRequest(BaseModel):
    user_id: str
    pet_type: str
    timestamp: str  # ISO 8601
    analysis_result: dict  # AI 분석 결과 JSON (behavior, audio, patella)
    video_url: str = ""  # optional

@app.post("/api/save-log")
def save_log_direct(req: DirectLogRequest):
    """
    Saves a pre-computed analysis result to Firebase RTDB under
    users/{user_id}/day/{YYYY-MM-DD}/{push_key}/.
    Used by the Flutter app to save test or real analysis data without uploading video.
    """
    try:
        log_time = datetime.datetime.fromisoformat(req.timestamp)
        date_str = log_time.strftime("%Y-%m-%d")
        time_str = log_time.strftime("%H:%M:%S")  # 분석 시각 (HH:MM:SS)

        ref = firebase_db.reference(f'users/{req.user_id}/day/{date_str}/{time_str}')
        ref.set({
            "image_url": req.video_url,
            "analysis_result": req.analysis_result
        })
        return {"status": "success", "time": time_str, "date": date_str, "user_id": req.user_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# AI 질환 분석 API
@app.post("/api/analyze-disease")
async def analyze_disease(
    pet_type: str = Form(...),
    disease_type: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        
        # ai_engine 추론 로직 호출
        result = ai_engine.analyze(
            image_bytes=contents,
            pet_type=pet_type,
            disease_type=disease_type
        )
        
        return result
    except Exception as e:
        return {"status": "error", "message": f"Server processing error: {str(e)}"}

# ─────────────────────────── HYBRID NEW ───────────────────────────
# AI 일상 행동 분석 API (Video Clip -> Frame Image Upload)
@app.post("/api/daily-behavior")
def analyze_daily_behavior(
    user_id: str = Form(...),
    pet_type: str = Form(...),
    file: UploadFile = File(...),
    timestamp: str = Form(None)
):
    import tempfile
    import os
    import cv2
    import traceback
    try:
        contents = file.file.read()
        filename = file.filename.lower()
        is_image = filename.endswith(('.jpg', '.jpeg', '.png'))
        
        print(f"Received file: {filename}, len={len(contents)} timestamp={timestamp}", flush=True)
        
        if is_image:
            # 1. Image Inference
            print("Starting image AI inference...", flush=True)
            ai_result = daily_behavior_engine.analyze_image(contents, pet_type)
            image_bytes = contents
            print("Completed image AI inference.", flush=True)
        else:
            # 1. Video Clip Inference
            print("Starting video AI inference...", flush=True)
            ai_result = daily_behavior_engine.analyze_clip(contents, pet_type)
            print("Completed video AI inference.", flush=True)

            # 2. Extract ONLY ONE frame for video
            print("Extracting frame from video...", flush=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", mode="wb") as tmp_file:
                tmp_file.write(contents)
                temp_video_path = tmp_file.name

            cap = cv2.VideoCapture(temp_video_path)
            ret, frame = cap.read()
            cap.release()
            os.remove(temp_video_path)
            
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()
            else:
                image_bytes = b""

        # 3. Upload image to MinIO
        minio_client = get_minio_client()
        object_name = f"{user_id}/{uuid.uuid4()}.jpg"
        
        minio_client.put_object(
            DAILY_BEHAVIOR_BUCKET,
            object_name,
            io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type="image/jpeg"
        )
        
        image_url = f"http://localhost:9000/{DAILY_BEHAVIOR_BUCKET}/{object_name}"
        
        # Determine timestamp
        if timestamp:
            try:
                log_time = datetime.datetime.fromisoformat(timestamp)
            except ValueError:
                log_time = datetime.datetime.now()
        else:
            log_time = datetime.datetime.now()

        # 4. Save to Firebase RTDB
        date_str = log_time.strftime("%Y-%m-%d")
        time_str = log_time.strftime("%H:%M:%S")
        ref = firebase_db.reference(f'users/{user_id}/day/{date_str}/{time_str}')
        ref.set({
            "image_url": image_url,
            "analysis_result": ai_result
        })
        
        return {
            "status": "success",
            "message": "Daily analysis saved successfully",
            "video_url": image_url,
            "ai_inference": ai_result
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Daily behavior processing error: {str(e)}"}

@app.post("/api/simulate-full-day")
async def simulate_full_day(
    user_id: str = Form(...),
    pet_type: str = Form(...)
):
    """
    SIMULATION ONLY: Extracts 24 frames from sample1.mp4, analyzes them,
    saves to DB, and returns a generated LLM diary.
    """
    import os
    import cv2
    import uuid
    import tempfile
    import datetime
    import io
    
    # Use a unique local name to avoid shadowing global VIDEO_PATH
    SIM_SAMPLE_FILE = "sample1.mp4" 
    sim_possible_paths = [
        SIM_SAMPLE_FILE,
        os.path.join(os.path.dirname(__file__), SIM_SAMPLE_FILE),
        os.path.join(os.path.dirname(__file__), "..", "..", SIM_SAMPLE_FILE),
        "/app/sample1.mp4"
    ]
    
    sim_found_path = None
    for p in sim_possible_paths:
        if os.path.exists(p):
            sim_found_path = p
            break
            
    if not sim_found_path:
        return {"status": "error", "message": f"Sample video not found. Tried: {sim_possible_paths}"}
    
    target_video = sim_found_path

    try:
        print(f"Starting simulation for user {user_id} using {target_video}")
        sim_cap = cv2.VideoCapture(target_video)
        if not sim_cap.isOpened():
            print(f"Error: Could not open {target_video}")
            return {"status": "error", "message": "Could not open sample video"}

        sim_total_frames = int(sim_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {sim_total_frames}")
        NUM_SIM_CHUNKS = 24
        sim_stride = max(1, sim_total_frames // NUM_SIM_CHUNKS)
        
        sim_today = datetime.datetime.now().strftime("%Y-%m-%d")
        minio_client = get_minio_client()

        for i in range(NUM_SIM_CHUNKS):
            f_idx = i * sim_stride
            sim_cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            success, sim_frame = sim_cap.read()
            if not success: 
                print(f"Warning: Could not read frame at index {f_idx}")
                continue
            
            # Encode frame
            _, sim_buffer = cv2.imencode('.jpg', sim_frame)
            sim_image_bytes = sim_buffer.tobytes()
            
            # 1. AI Analysis
            print(f"Analyzing frame {i+1}/24 (pet_type={pet_type})...")
            sim_ai_result = daily_behavior_engine.analyze_image(sim_image_bytes, pet_type)
            
            # 2. MinIO Upload
            object_name = f"{user_id}/sim_{uuid.uuid4()}.jpg"
            minio_client.put_object(
                DAILY_BEHAVIOR_BUCKET,
                object_name,
                io.BytesIO(sim_image_bytes),
                length=len(sim_image_bytes),
                content_type="image/jpeg"
            )
            image_url = f"http://localhost:9000/{DAILY_BEHAVIOR_BUCKET}/{object_name}"
            
            # 3. Firebase Save
            chunk_time = datetime.datetime.now().replace(hour=i % 24, minute=0, second=0, microsecond=0)
            time_str = chunk_time.strftime("%H:%M:%S")
            full_timestamp = chunk_time.isoformat()
            
            ref = firebase_db.reference(f'users/{user_id}/day/{sim_today}/{time_str}')
            ref.set({
                "video_url": image_url, # Gallery currently looks for video_url
                "image_url": image_url,
                "timestamp": full_timestamp,
                "analysis_result": sim_ai_result
            })
            
        sim_cap.release()
        print(f"Simulation loop successfully finished for {user_id}")
        
        # 4. Generate Diary
        diary_content = generate_daily_diary(user_id, sim_today)
        
        return {
            "status": "success",
            "message": "Full day simulation completed and diary generated",
            "diary": diary_content
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Simulation failed: {str(e)}"}

# ─────────────────────────── PHASE 3: AI DIARY & STATISTICS ───────────────────────────

@app.get("/api/daily-diary/{user_id}")
async def get_daily_diary(user_id: str, date: str = None):
    """
    Returns a daily diary generated by Groq (LLaMA3) based on the user's pet's daily logs.
    date format: YYYY-MM-DD
    """
    try:
        diary_content = generate_daily_diary(user_id, date)
        if any(keyword in diary_content for keyword in ["오류", "어렵습니다", "초기화되지 않았습니다", "실패"]):
            return {"status": "error", "message": diary_content}
        
        return {
            "status": "success",
            "user_id": user_id,
            "date": date if date else datetime.datetime.now().strftime("%Y-%m-%d"),
            "diary": diary_content
        }
    except Exception as e:
        return {"status": "error", "message": f"Diary generation error: {str(e)}"}

@app.get("/api/statistics/{user_id}")
async def get_pet_statistics(user_id: str, pet_type: str):
    """
    Returns weekly aggregated statistics for the pet based on daily logs.
    Includes emotion charts and patella warnings (if dog).
    """
    try:
        stats = get_weekly_statistics(user_id, pet_type)
        return {
            "status": "success",
            "user_id": user_id,
            "pet_type": pet_type,
            "statistics": stats
        }
    except Exception as e:
        return {"status": "error", "message": f"Statistics aggregation error: {str(e)}"}

@app.get("/api/daily-diaries/{user_id}")
def fetch_diary_list(user_id: str, limit: int = 0):
    """
    Fetches a list of generated diaries for the dashboard (limit=5) or total view (limit=0).
    """
    try:
        diaries = get_diary_list(user_id, limit)
        return {
            "status": "success",
            "user_id": user_id,
            "data": diaries
        }
    except Exception as e:
        return {"status": "error", "message": f"Fetching diaries error: {str(e)}"}

@app.get("/api/gallery/{user_id}")
def get_video_gallery(user_id: str):
    """
    Fetches video URLs from users/{user_id}/day/* to display in the Photo Gallery.
    """
    try:
        day_ref = firebase_db.reference(f'users/{user_id}/day')
        all_days = day_ref.get() or {}
        
        gallery_items = []
        for date_key, logs_on_day in all_days.items():
            if not isinstance(logs_on_day, dict):
                continue
            for push_key, doc in logs_on_day.items():
                if isinstance(doc, dict):
                    # Safely get the analysis results
                    beh_info = doc.get("analysis_result", {})
                    if isinstance(beh_info, dict) and beh_info.get("status") == "success":
                        beh_data = beh_info.get("behavior_analysis", {})
                        emotion = beh_data.get("emotion", "Unknown") if isinstance(beh_data, dict) else "Unknown"
                    else:
                        emotion = "Unknown"
                        
                    timestamp_str = doc.get("timestamp", "")
                    
                    # Format timestamp for display
                    display_time = "Unknown"
                    if timestamp_str:
                        try:
                            dt = datetime.datetime.fromisoformat(timestamp_str)
                            display_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            display_time = timestamp_str

                    url = doc.get("video_url") or doc.get("image_url", "")
                    url = url.replace("minio:9000", "localhost:9000")
                    
                    gallery_items.append({
                        "timestamp": display_time,
                        "video_url": url,
                        "emotion": emotion,
                        "is_image": "video_url" not in doc,
                        "_raw_time": timestamp_str
                    })
                
        # Sort by timestamp descending
        gallery_items.sort(key=lambda x: x.get("_raw_time", ""), reverse=True)
        
        # Remove sorting key
        for item in gallery_items:
            item.pop("_raw_time", None)
            
        return {
            "status": "success",
            "user_id": user_id,
            "data": gallery_items
        }
    except Exception as e:
        return {"status": "error", "message": f"Gallery fetching error: {str(e)}"}
