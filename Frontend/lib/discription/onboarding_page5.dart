import 'package:flutter/material.dart';
import 'package:pet_diary/pet_name_input_page.dart';
import 'package:http/http.dart' as http; // 상단에 추가
import 'dart:convert'; // JSON 변환을 위해 추가
import 'package:pet_diary/main.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';

class OnboardingPage5 extends StatelessWidget {
  const OnboardingPage5({super.key});

  // 1. 서버와 통신하는 실제 로직 (여기에 추가)
  Future<void> _handleAuth(BuildContext context, String id, String pw, bool isSignup) async {
    final String endpoint = isSignup ? '/signup/' : '/login/';
    final Uri url = Uri.parse('http://localhost:8080$endpoint');

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'user_id': id, 'password': pw}),
      );

      final result = jsonDecode(response.body);
      if (response.statusCode == 200 && result['status'] == 'success') {
        if (!context.mounted) return;
        Navigator.pop(context); // 다이얼로그 닫기

        // 회원가입인 경우 무조건 등록 페이지로 이동
        if (isSignup) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => PetNameInputPage(userId: id)),
          );
        }
        // 로그인인 경우 데이터 존재 여부에 따라 분기
        else {
          bool hasPet = result['has_pet_info'] ?? false;

          if (hasPet) {
            // 1. 데이터가 이미 있으므로 바로 대시보드로 이동
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (context) => PetHealthDashboard(userId: result['user_id']),
              ),
            );
          } else {
            // 2. 데이터가 없으므로 이름 입력(등록) 페이지로 이동
            Navigator.pushReplacement(
              context,
              MaterialPageRoute(
                builder: (context) => PetNameInputPage(userId: id),
              ),
            );
          }
        }
        // Navigator.pushReplacement(
        //   context,
        //   MaterialPageRoute(
        //     builder: (context) => PetNameInputPage(userId: result['user_id']), // 실제 서버에서 받은 ID 전달
        //   ),
        // );
      } else {
        if (!context.mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("로그인 실패: ${result['message'] ?? '알 수 없는 오류'}")),
        );
      }
    } catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("서버 연결 실패: 백엔드가 켜져있는지 확인해주세요.")),
      );
    }
  }

  // 2. 질문하신 다이얼로그 함수 (여기에 추가)
  void _showEmailAuthDialog(BuildContext context) {
    final TextEditingController idController = TextEditingController();
    final TextEditingController pwController = TextEditingController();

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text("이메일 로그인/가입", style: TextStyle(fontWeight: FontWeight.bold)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(controller: idController, decoration: const InputDecoration(hintText: "아이디")),
            const SizedBox(height: 10),
            TextField(controller: pwController, obscureText: true, decoration: const InputDecoration(hintText: "비밀번호")),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => _handleAuth(context, idController.text, pwController.text, false),
            child: const Text("로그인", style: TextStyle(color: Colors.orange)),
          ),
          ElevatedButton(
            onPressed: () => _handleAuth(context, idController.text, pwController.text, true),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.orange, elevation: 0),
            child: const Text("회원가입", style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }
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
                  width: index == 4 ? 40 : 30, // 2번째 바를 길게 표시
                  height: 6,
                  decoration: BoxDecoration(
                    color: index == 4 ? pointColor : Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(3.r),
                  ),
                )),
              ),
              SizedBox(height: 40),

              // 3. 텍스트 영역
              Text(
                "마지막 페이지입니다.",
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
              // 4. 하단 버튼
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: () {
                    // 닫기 버튼 클릭 시 아래에서 로그인 창(Bottom Sheet)이 올라옴
                    showModalBottomSheet(
                      context: context,
                      isScrollControlled: true, // 높이 조절을 위해 true 설정
                      backgroundColor: Colors.transparent, // 모서리 곡률을 위해 투명 설정
                      builder: (context) {
                        return _buildLoginSheet(context);
                      },
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
                    "닫기",
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

  // --- 로그인 바텀 시트 위젯 ---
  Widget _buildLoginSheet(BuildContext context) {

    return Container(
      height: MediaQuery.of(context).size.height * 0.50,
      padding: EdgeInsets.symmetric(horizontal: 24.w, vertical: 20.h),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(20.r)),
      ),
      child: Column(
        children: [
          // 상단 바 (Handle)
          Container(
            width: 40.w,
            height: 4.h,
            margin: EdgeInsets.only(bottom: 24.h), // 이 코드로 교체하세요
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(2.r),
            ),
          ),
          Text(
            "로그인 및 시작하기",
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 20.h),

          // 카카오 로그인 버튼
          _buildSNSButton(
            icon: Icons.chat_bubble,
            label: "카카오로 시작하기",
            color: const Color(0xFFFEE500),
            textColor: Colors.black,
            onTap: () {
              // 실제 로그인 로직 구현부
            },
          ),
          SizedBox(height: 10.h),

          // 구글 로그인 버튼
          _buildSNSButton(
            icon: Icons.g_mobiledata,
            label: "Google로 시작하기",
            color: Colors.white,
            textColor: Colors.black,
            onTap: () {
              // 실제 로그인 로직 구현부
            },
            isBorder: true,
          ),
          SizedBox(height: 10.h),

          // SNS 버튼 리스트 아래에 추가할 '이메일 로그인' 버튼
          _buildSNSButton(
            icon: Icons.email,
            label: "이메일로 시작하기",
            color: Colors.grey[200]!,
            textColor: Colors.black,
            onTap: () {
              Navigator.pop(context); // 바텀 시트를 먼저 닫음
              _showEmailAuthDialog(context); // 다이얼로그 호출
            },
          ),

          // 비회원 로그인 서브메뉴
          TextButton(
            onPressed: () {
              // 1. 바텀 시트 닫기
              Navigator.pop(context);
              // 2. 메인 대시보드 화면으로 이동
              // 2. 메인 대시보드 화면으로 이동
              Navigator.pushReplacement(
                context,
                MaterialPageRoute(
                  builder: (context) => const PetNameInputPage(
                    userId: "guest_user", // [수정] 비회원용 임시 ID 전달
                  ),
                ),
              );
            },
            child: Text(
              "로그인 없이 둘러보기 (비회원)",
              style: TextStyle(
                color: Colors.grey[500],
                decoration: TextDecoration.underline,
                fontSize: 14,
              ),
            ),
          ),
        ],
      ),
    );
  }



  // SNS 버튼 공통 디자인 위젯
  Widget _buildSNSButton({
    required IconData icon,
    required String label,
    required Color color,
    required Color textColor,
    required VoidCallback onTap,
    bool isBorder = false,
  }) {
    return InkWell(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        height: 48.h,
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(12.r),
          border: isBorder ? Border.all(color: Colors.grey.shade300) : null,
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: textColor, size: 22),
            SizedBox(width: 10.w),
            Text(
              label,
              style: TextStyle(color: textColor, fontWeight: FontWeight.w600, fontSize: 15),
            ),
          ],
        ),
      ),
    );
  }
}