from datetime import datetime, timedelta
from firebase_admin import db as firebase_db

def get_weekly_statistics(user_id: str, pet_type: str) -> dict:
    """
    Fetches and aggregates weekly statistics for a pet, 
    including Emotion Index (Stress vs Happy) and Patella Warning count (dogs only).
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Fetch logs from Firebase RTDB: users/{user_id}/day/{YYYY-MM-DD}/{push_key}/
    day_ref = firebase_db.reference(f'users/{user_id}/day')
    all_days = day_ref.get() or {}
    
    logs = []
    # Iterate over each date, then each push_key log within the date
    for date_key, logs_on_day in all_days.items():
        if not isinstance(logs_on_day, dict):
            continue
        for push_key, log in logs_on_day.items():
            if isinstance(log, dict) and log.get("pet_type") == pet_type:
                log_time_str = log.get("timestamp")
                if log_time_str:
                    try:
                        log_time = datetime.fromisoformat(log_time_str)
                        if start_time <= log_time <= end_time:
                            log["timestamp"] = log_time  # replace string with datetime object for later use
                            logs.append(log)
                    except ValueError:
                        pass

    # Sort by timestamp ascending
    logs.sort(key=lambda x: x["timestamp"])
    
    # Initialize statistics containers
    stats = {
        "total_logs_analyzed": len(logs),
        "period": f"{start_time.strftime('%Y-%m-%d')} ~ {end_time.strftime('%Y-%m-%d')}",
        "emotion_index": {
            "happy_ratio": 0.0,
            "stress_ratio": 0.0,
            "neutral_ratio": 0.0,
            "daily_trend": [] # Array of {date, happy_count, stress_count}
        },
        "behavior_summary": {}
    }
    
    if pet_type.lower() == "dog":
        stats["patella_warnings"] = {
            "total_abnormal_count": 0,
            "message": ""
        }
        
    if not logs:
        return stats
        
    # Variables for aggregating
    emotion_counts = {"positive": 0, "negative": 0, "neutral": 0}
    daily_emotions = {}
    
    positive_emotions = ["happy", "excited", "relaxed", "dog_happy", "cat_happy", "cat_relaxed", "dog_relaxed"]
    negative_emotions = ["anxious", "scared", "sad", "angry", "stress", "dog_anxious", "dog_scared", "cat_anxious", "cat_scared"]
    
    patella_alert_count = 0
    
    for log in logs:
        date_str = log["timestamp"].strftime("%Y-%m-%d")
        if date_str not in daily_emotions:
            daily_emotions[date_str] = {"happy_count": 0, "stress_count": 0}
            
        behavior_info = log.get("analysis_result", {})
        
        if behavior_info.get("status") == "success":
            # 1. Emotion Processing
            emotion = behavior_info.get("behavior_analysis", {}).get("emotion", "").lower()
            
            is_positive = any(pe in emotion for pe in positive_emotions)
            is_negative = any(ne in emotion for ne in negative_emotions)
            
            if is_positive:
                emotion_counts["positive"] += 1
                daily_emotions[date_str]["happy_count"] += 1
            elif is_negative:
                emotion_counts["negative"] += 1
                daily_emotions[date_str]["stress_count"] += 1
            else:
                emotion_counts["neutral"] += 1
                
            # 2. General Behavior tally
            beh = behavior_info.get("behavior_analysis", {}).get("detected_behavior", "Unknown")
            stats["behavior_summary"][beh] = stats["behavior_summary"].get(beh, 0) + 1
            
            # 3. Patella (Dog only)
            if pet_type.lower() == "dog":
                pat = behavior_info.get("patella_analysis", {}).get("status", "").lower()
                # Assuming 'abnormal', 'warning', 'limping' etc indicate issues
                if pat in ["abnormal", "warning", "limping", "이상", "비정상"]:
                    patella_alert_count += 1
                    
    # Calculate Ratios
    total_emotions = sum(emotion_counts.values())
    if total_emotions > 0:
        stats["emotion_index"]["happy_ratio"] = round((emotion_counts["positive"] / total_emotions) * 100, 1)
        stats["emotion_index"]["stress_ratio"] = round((emotion_counts["negative"] / total_emotions) * 100, 1)
        stats["emotion_index"]["neutral_ratio"] = round((emotion_counts["neutral"] / total_emotions) * 100, 1)
        
    for date, counts in sorted(daily_emotions.items()):
        stats["emotion_index"]["daily_trend"].append(
            {"date": date, "happy_count": counts["happy_count"], "stress_count": counts["stress_count"]}
        )
        
    if pet_type.lower() == "dog":
        stats["patella_warnings"]["total_abnormal_count"] = patella_alert_count
        if patella_alert_count >= 5:
            stats["patella_warnings"]["message"] = f"최근 일주일간 슬개골 이상 모션이 총 {patella_alert_count}회 발견되었습니다. 가까운 동물병원 방문을 강력히 권장합니다."
        elif patella_alert_count > 0:
            stats["patella_warnings"]["message"] = f"최근 일주일간 슬개골 이상 모션이 총 {patella_alert_count}회 발견되었습니다. 아이의 걸음걸이를 주의 깊게 지켜보세요."
        else:
            stats["patella_warnings"]["message"] = "최근 일주일간 슬개골 이상 모션이 발견되지 않았습니다. 건강한 관절 상태를 유지하고 있습니다!"
            
    return stats
