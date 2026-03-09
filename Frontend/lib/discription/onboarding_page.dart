import 'package:flutter/material.dart';
import 'package:pet_diary/discription/onboarding_page2.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class OnboardingPage extends StatelessWidget {
  const OnboardingPage({super.key});

  @override
  Widget build(BuildContext context) {
    // 오렌지 포인트 컬러 설정
    const Color pointColor = Color(0xFFFF7A00);
    const Color backgroundColor = Color(0xFFF5F5F5);

    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            children: [
              SizedBox(height: 40),
              // 1. 이미지 영역 (나중에 이미지를 바꿀 수 있는 박스)
              Container(
                width: double.infinity,
                height: MediaQuery.of(context).size.height * 0.45,
                decoration: BoxDecoration(
                  color: backgroundColor,
                  borderRadius: BorderRadius.circular(30.r),
                ),
                child: Center(
                  // 이 부분을 Image.asset('경로')로 바꾸시면 됩니다.
                  child: Icon(
                      Icons.pets_rounded,
                      size: 100,
                      color: pointColor.withOpacity(0.5)
                  ),
                ),
              ),
              SizedBox(height: 30),

              // 2. 인디케이터 (현재 2번째 페이지 활성화)
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: List.generate(5, (index) => Container(
                  margin: EdgeInsets.symmetric(horizontal: 4),
                  width: index == 0 ? 40 : 30, // 1번째 바를 길게 표시
                  height: 6,
                  decoration: BoxDecoration(
                    color: index == 0 ? pointColor : Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(3.r),
                  ),
                )),
              ),
              SizedBox(height: 40),

              // 3. 텍스트 영역
              Text(
                "첫 번째 페이지입니다.",
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
              ),
              SizedBox(height: 12),
              Text(
                "우리 아이의 평소와 다른 움직임을\n실시간으로 분석하여 알려드려요.",
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.grey.shade600,
                  height: 1.5,
                ),
              ),

              const Spacer(),

              // 4. 하단 버튼
              // 4. 하단 버튼
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: () {
                    // 다음 페이지(기존 대시보드)로 이동 로직 추가
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const OnboardingPage2(),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: pointColor,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12.r),
                    ),
                    elevation: 0,
                  ),
                  child: Text(
                    "다음으로",
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
              ),
              SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}