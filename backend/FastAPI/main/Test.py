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
# Hybrid DB Imports
from FastAPI.main.db import get_db, get_minio_client, DAILY_BEHAVIOR_BUCKET
from FastAPI.main.daily_behavior_inference import daily_behavior_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load AI models
    print("Initializing AI Models...")
    ai_engine.load_models()
    
    # Initialize Daily Behavior Engine
    daily_behavior_engine.load_models()
    
    # Initialize DB Connections
    get_db()
    get_minio_client()
    
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# 1. 파이어베이스 초기화 (다운로드한 키 파일 이름을 입력하세요)
cred = credentials.Certificate("key/testApi.json")
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
async def login(user: User):
    ref = firebase_db.reference(f'users/{user.user_id}')
    user_data = ref.get()

    if user_data and user_data.get('password') == user.password:
        # Check MongoDB for pet info
        db_client = get_db()
        user_record = db_client["users"].find_one({"user_id": user.user_id})
        has_pet_info = bool(user_record and "pet_info" in user_record)

        return {
            "status": "success",
            "message": "로그인 성공",
            "user_id": user.user_id,
            "has_pet_info": has_pet_info # 플러터에 알려줌
        }
    else:
        return {"status": "error", "message": "아이디 또는 비밀번호가 틀렸습니다."}

@app.post("/signup/")
async def signup(user: User):
    # 회원가입 (중복 체크 생략)
    ref = firebase_db.reference(f'users/{user.user_id}')
    # 이미 가입된 아이디인지 확인
    if ref.get() is not None:
        return {"status": "error", "message": "이미 존재하는 아이디입니다."}

    ref.set({"password": user.password})
    
    # Create empty user record in MongoDB
    db_client = get_db()
    db_client["users"].insert_one({"user_id": user.user_id, "created_at": datetime.datetime.now()})
    
    return {"status": "success"}

# 테스트 용
class Log(BaseModel):
    pet_name: str
    behavior: str # 예: "짖음", "이상 보행" 등
    timestamp: str

@app.post("/log/")
async def save_log(log: Log):
    # 'pet_logs'라는 경로에 데이터 저장
    ref = firebase_db.reference('pet_logs')
    new_log_ref = ref.push(log.dict())
    return {"status": "success", "id": new_log_ref.key}
    
@app.get("/logs/")
async def get_logs():
    ref = firebase_db.reference('pet_logs')
    return ref.get()

# 사용자 입력 : 반려동물 기본 정보 PetRegistrationPage
class PetInfo(BaseModel):
    pet_name: str
    pet_type: str
    pet_gender: str
    pet_birthday : str
    
@app.post("/user-input/{user_id}")
async def save_pet_info(user_id: str, data: PetInfo):
    db_client = get_db()
    # Update user document with pet_info in MongoDB
    result = db_client["users"].update_one(
        {"user_id": user_id},
        {"$set": {"pet_info": data.model_dump()}},
        upsert=True
    )
    return {"status": "success", "user_id": user_id}

@app.get("/user-pet-info/{user_id}")
async def get_all_pet_info(user_id: str):
    db_client = get_db()
    user_data = db_client["users"].find_one({"user_id": user_id})

    if user_data and "pet_info" in user_data:
        return {
            "status": "success",
            "data": user_data["pet_info"]  # pet_name, pet_type, pet_gender, pet_birthday
        }
    else:
        return {"status": "error", "message": "반려동물 정보가 없습니다."}

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
# AI 일상 행동 분석 API (Video Clip)
@app.post("/api/daily-behavior")
async def analyze_daily_behavior(
    user_id: str = Form(...),
    pet_type: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        
        # 1. Upload to MinIO
        minio_client = get_minio_client()
        file_extension = file.filename.split(".")[-1] if "." in file.filename else "mp4"
        object_name = f"{user_id}/{uuid.uuid4()}.{file_extension}"
        
        minio_client.put_object(
            DAILY_BEHAVIOR_BUCKET,
            object_name,
            io.BytesIO(contents),
            length=len(contents),
            content_type=file.content_type
        )
        
        video_url = f"http://localhost:9000/{DAILY_BEHAVIOR_BUCKET}/{object_name}"
        
        # 2. Daily Behavior & Sound Inference
        ai_result = daily_behavior_engine.analyze_clip(contents, pet_type)
        
        # 3. Save resulting metadata to MongoDB
        db_client = get_db()
        doc = {
            "user_id": user_id,
            "pet_type": pet_type,
            "timestamp": datetime.datetime.now(),
            "video_url": video_url,
            "analysis_result": ai_result
        }
        db_client["daily_logs"].insert_one(doc)
        
        return {
            "status": "success",
            "message": "Routine behavior analyzed successfully",
            "video_url": video_url,
            "ai_inference": ai_result
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Daily behavior processing error: {str(e)}"}

