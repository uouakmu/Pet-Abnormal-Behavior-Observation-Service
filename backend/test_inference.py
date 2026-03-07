import traceback
from FastAPI.main.daily_behavior_inference import daily_behavior_engine

try:
    daily_behavior_engine.load_models()
    print('models loaded')
    with open('sample1.mp4','rb') as f:
        data = f.read()
    res = daily_behavior_engine.analyze_clip(data, 'dog')
    print('RESULT:', res)
except Exception as e:
    print('ERROR:', e)
    traceback.print_exc()
