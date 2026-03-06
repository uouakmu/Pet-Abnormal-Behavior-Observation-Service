from fastapi import FastAPI, File, UploadFile, Form
from contextlib import asynccontextmanager
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, db
from fastapi.middleware.cors import CORSMiddleware
import io

# AI Inference module import
from FastAPI.main.model_inference import ai_engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load AI models
    print("Initializing AI Models...")
    ai_engine.load_models()
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
    ref = db.reference(f'users/{user.user_id}')
    user_data = ref.get()

    if user_data and user_data.get('password') == user.password:
        # pet_info가 존재하는지 확인 (T/F)
        has_pet_info = "pet_info" in user_data

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
    ref = db.reference(f'users/{user.user_id}')
    # 이미 가입된 아이디인지 확인
    if ref.get() is not None:
        return {"status": "error", "message": "이미 존재하는 아이디입니다."}

    ref.set({"password": user.password})
    return {"status": "success"}

# 테스트 용
class Log(BaseModel):
    pet_name: str
    behavior: str # 예: "짖음", "이상 보행" 등
    timestamp: str

@app.post("/log/")
async def save_log(log: Log):
    # 'pet_logs'라는 경로에 데이터 저장
    ref = db.reference('pet_logs')
    new_log_ref = ref.push(log.dict())
    return {"status": "success", "id": new_log_ref.key}
@app.get("/logs/")
async def get_logs():
    ref = db.reference('pet_logs')
    return ref.get()

# 사용자 입력 : 반려동물 기본 정보 PetRegistrationPage
class PetInfo(BaseModel):
    pet_name: str
    pet_type: str
    pet_gender: str
    pet_birthday : str
@app.post("/user-input/{user_id}")
async def save_pet_info(user_id: str, data: PetInfo):
    # users > {user_id} > pet_info 경로에 바로 저장
    ref = db.reference(f'users/{user_id}/pet_info')
    ref.set(data.model_dump()) # 전체 데이터를 해당 경로에 덮어씀

    return {"status": "success", "user_id": user_id}

@app.get("/user-pet-info/{user_id}")
async def get_all_pet_info(user_id: str):
    # 해당 유저의 pet_info 노드 전체를 가져옵니다.
    ref = db.reference(f'users/{user_id}/pet_info')
    pet_data = ref.get()

    if pet_data:
        return {
            "status": "success",
            "data": pet_data  # pet_name, pet_type, pet_gender, pet_birthday가 모두 포함됨
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

