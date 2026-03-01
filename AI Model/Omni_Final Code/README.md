## Folder Overview
- 실제 가중치 파일 추출에 사용한 코드 폴더입니다.
- 코드 목록
    - dog_nomral_omni_train.py
    - cat_nomral_omni_train.py
    - dog_abnomral_omni_train.py
    - cat_abnomral_omni_train.py

- 코드 설명
    - 기본적으로 20 epoch마다 WarmRestart가 적용되어 있으며, 각 epoch에 따른 학습곡선 그래프를 실시간으로 저장합니다.
    - 최종적으로 가장 높은 성능의 정확도를 보인 가중치 파일이 선택되고, 이를 통해 서비스에 사용합니다.
    - 파일명이 AAA_BBB_omni_CCC.py로 되어 있으며, 그 의미는 다음과 같습니다.
        - AAA: 반려동물(Dog & Cat)
        - BBB: 정상 분류용 / 이상 증상 분류용
        - CCC: Train목적 / Test 목적