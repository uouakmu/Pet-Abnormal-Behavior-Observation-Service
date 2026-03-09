import 'package:flutter/material.dart';
import 'package:pet_diary/mainPage/total_diary.dart';
import 'package:pet_diary/mainPage/odd_pet.dart';
import 'package:pet_diary/mainPage/daily_pet.dart';
import 'package:pet_diary/discription/onboarding_page.dart';
import 'package:pet_diary/mainPage/examination_page.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:pet_diary/mainPage/mypage.dart';
import 'package:pet_diary/mainPage/photo_gallery.dart'; // Phase 4 Photo Gallery
import 'package:pet_diary/mainPage/diary_detail.dart'; // Phase 4.1 Detail View

import 'package:flutter_screenutil/flutter_screenutil.dart';

void main() async {
  await ScreenUtil.ensureScreenSize();
  runApp(
    ScreenUtilInit(
      designSize: const Size(390, 844), // Standard standard mobile size
      minTextAdapt: true,
      splitScreenMode: true,
      builder: (context, child) {
        return const MaterialApp(
          home: OnboardingPage(),
          debugShowCheckedModeBanner: false,
        );
      },
    ),
  );
}

class PetHealthDashboard extends StatefulWidget {
  final String userId;
  const PetHealthDashboard({super.key, required this.userId});

  @override
  State<PetHealthDashboard> createState() => _PetHealthDashboardState();
}

class _PetHealthDashboardState extends State<PetHealthDashboard> {
  int _selectedIndex = 2;

  // 1. 단일 반려동물 정보를 담을 변수로 변경
  Map<String, dynamic>? petData;
  bool isLoading = true;

  // 2. Fetch recent diaries
  List<dynamic> recentDiaries = [];
  bool isDiaryLoading = true;
  final String baseUrl = "http://localhost:8080"; // !IMPORTANT: 안드로이드 실기기 IP 입력 부분 (예: 192.168.0.X:8080)

  @override
  void initState() {
    super.initState();
    _fetchPetInfo();
    _fetchRecentDiaries();
  }
  Future<void> _fetchPetInfo() async {
    final url = Uri.parse('http://localhost:8080/user-pet-info/${widget.userId}');

    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final Map<String, dynamic> result = json.decode(response.body);

        setState(() {
          if (result['status'] == 'success') {
            // 3. 서버 응답의 'data' 부분을 할당
            petData = result['data'];
          } else {
            petData = null;
          }
          isLoading = false;
        });
      }
    } catch (e) {
      print('연결 실패: $e');
      setState(() => isLoading = false);
    }
  }

  Future<void> _fetchRecentDiaries() async {
    final url = Uri.parse('$baseUrl/api/daily-diaries/${widget.userId}?limit=5');
    try {
      final response = await http.get(url);
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded['status'] == 'success') {
          setState(() {
            recentDiaries = decoded['data'] ?? [];
            isDiaryLoading = false;
          });
          return;
        }
      }
    } catch (e) {
      print('Diary fetch error: $e');
    }
    setState(() => isDiaryLoading = false);
  }
  // 하단 탭 클릭 시 상태 변경 함수
  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: const Icon(Icons.menu, color: Colors.black),
        title: Text(
            _selectedIndex == 1 ? 'AI 검진' : _selectedIndex == 2 ? 'Daily Behavior Diary' : (_selectedIndex == 4 ? '마이페이지' : '준비 중'),
            style: const TextStyle(color: Colors.black, fontSize: 14, fontWeight: FontWeight.bold)
        ),
        centerTitle: true,
        actions: [IconButton(icon: const Icon(Icons.share, color: Colors.blue), onPressed: () {})],
      ),

      // 선택된 탭 인덱스에 따라 홈 화면 또는 준비중 화면 표시
      body: _selectedIndex == 1
          ? ExaminationPage(petData: petData) // 새로 만든 AI 검진 페이지 연결
          : _selectedIndex == 2
          ? _buildDashboardHome() // 홈 대시보드
          : _selectedIndex == 3
          ? PhotoGalleryPage(userId: widget.userId) // 사진첩 연동 (Phase 4)
          : _selectedIndex == 4
          ? MyPage(petData: petData)    // 마이페이지 (새로 만든 파일 연결)
          : Center(child: Text('준비 중인 페이지입니다.', style: TextStyle(color: Colors.grey[400], fontSize: 16))),

      // 하단 내비게이션 바
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        backgroundColor: Colors.white,
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        selectedItemColor: Colors.green[700], // 현재 페이지 노란색(강조)
        unselectedItemColor: Colors.grey[400],
        selectedFontSize: 11,
        unselectedFontSize: 11,
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.stars), label: '모니터링'),
          BottomNavigationBarItem(icon: Icon(Icons.health_and_safety), label: '검진'),
          BottomNavigationBarItem(icon: Icon(Icons.home), label: '홈'),
          BottomNavigationBarItem(icon: Icon(Icons.favorite), label: '사진첩'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: '마이페이지'),
        ],
      ),
    );
  }

  // --- 메인 홈 대시보드 UI ---
  Widget _buildDashboardHome() {
    return SingleChildScrollView(
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeaderCard(),
          SizedBox(height: 12),
          Row(
            children: [
              _buildActionButton(Icons.book, '일상 일기', '기분 & 활동량', Colors.blue,
                      () => Navigator.push(context, MaterialPageRoute(builder: (context) => daily_pet()))),
              SizedBox(width: 8),
              _buildActionButton(Icons.error_outline, '이상 행동', '건강 체크', Colors.orange,
                      () => Navigator.push(context, MaterialPageRoute(builder: (context) => PageB()))),
            ],
          ),
          SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '최근 일기',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              GestureDetector(
                onTap: () {
                  // DiaryListPage로 이동
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => DiaryListPage(userId: widget.userId)),
                  );
                },
                child: Text(
                  '전체보기 →',
                  style: TextStyle(
                    color: Colors.purple[300],
                    fontSize: 12,
                    fontWeight: FontWeight.w500, // 약간의 두께감을 주면 더 버튼 같습니다.
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 12),
          isDiaryLoading
              ? const Center(child: CircularProgressIndicator())
              : recentDiaries.isEmpty
                  ? const Center(child: Text('최근 일기 기록이 없습니다.'))
                  : Column(
                      children: recentDiaries.map((diaryItem) {
                        return GestureDetector(
                          onTap: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(builder: (context) => DiaryDetailPage(diaryData: diaryItem)),
                            );
                          },
                          child: _buildDiaryItem(
                            diaryItem['date'] ?? '알 수 없는 날짜',
                            '최신 일기', // 혹은 요일 계산 로직
                            90, // 임시 활동 점수
                            false, // 임시 warning 로직
                            diaryItem['content'] ?? '내용 없음',
                          ),
                        );
                      }).toList(),
                    ),
 main
          // ------------------------------------------

          SizedBox(height: 12),
          _buildTrendSection(),
          SizedBox(height: 16),
          Center(
            child: Column(
              children: [
                Text('AI가 24시간 콩이를 모니터링하고 있어요', style: TextStyle(color: Colors.grey, fontSize: 12)),
                SizedBox(height: 2),
                Text('8가지 데이터셋 기반 건강 분석 시스템', style: TextStyle(color: Colors.grey, fontSize: 11)),
              ],
            ),
          ),
          SizedBox(height: 20),
        ],
      ),
    );
  }

  // --- 헬퍼 함수들 ---

  Widget _buildHeaderCard() {
    String petName = petData?['pet_name'] ?? '콩이';
    String petType = petData?['pet_type'] ?? '반려동물';
    return Container(
      padding: EdgeInsets.all(14),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16.r),
        gradient: const LinearGradient(colors: [Colors.purple, Colors.orangeAccent]),
      ),
      child: Column(
        children: [
          Row(
            children: [
              CircleAvatar(radius: 20.r, backgroundColor: Colors.white),
              SizedBox(width: 8),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('$petName의 건강일기', style: TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.bold)),
                  Text('AI 기반 반려동물 케어', style: TextStyle(color: Colors.white70, fontSize: 9)),
                ],
              )
            ],
          ),
          SizedBox(height: 9),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _buildStatItem('12', '총 일기'),
              _buildStatItem('85', '평균 활동'),
              _buildStatItem('98%', '건강도'),
            ],
          )
        ],
      ),
    );
  }

  Widget _buildStatItem(String value, String label) {
    return Column(
      children: [
        Text(value, style: TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold)),
        Text(label, style: TextStyle(color: Colors.white70, fontSize: 9)),
      ],
    );
  }

  Widget _buildActionButton(IconData icon, String title, String subTitle, Color color, VoidCallback onTap) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12.r),
        child: Container(
          padding: EdgeInsets.symmetric(vertical: 9),
          decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(9.r)),
          child: Column(
            children: [
              Icon(icon, color: Colors.white, size: 20),
              SizedBox(height: 2.5),
              Text(title, style: TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold)),
              Text(subTitle, style: TextStyle(color: Colors.white70, fontSize: 9)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDiaryItem(String date, String day, int activity, bool hasWarning, String contentSummary) {
    // 일기 텍스트 미리보기용 축약 (첫 30글자만)
    String preview = contentSummary.length > 30 ? '${contentSummary.substring(0, 30)}...' : contentSummary;
    
    return Container(
      margin: EdgeInsets.only(bottom: 8),
      padding: EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12.r),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Row(
        children: [

          Container(width: 50, height: 50, decoration: BoxDecoration(color: Colors.purple[100], borderRadius: BorderRadius.circular(8)), child: const Icon(Icons.menu_book, color: Colors.purple)),
          const SizedBox(width: 12),
 main
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [

                Text(date, style: const TextStyle(fontWeight: FontWeight.bold)),
                Text(preview, style: const TextStyle(color: Colors.grey, fontSize: 11), maxLines: 1, overflow: TextOverflow.ellipsis),
                Row(
                  children: [
                    const Icon(Icons.pets, size: 14, color: Colors.green),
                    const Text(' AI 일기 생성됨', style: TextStyle(fontSize: 10)),
 main
                    if (hasWarning) ...[
                      SizedBox(width: 6),
                      Container(
                        padding: EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                        decoration: BoxDecoration(color: Colors.orange[100], borderRadius: BorderRadius.circular(4.r)),
                        child: Text('주의사항', style: TextStyle(color: Colors.orange, fontSize: 10)),
                      )
                    ]
                  ],
                )
              ],
            ),
          ),

          const Icon(Icons.arrow_forward_ios, size: 14, color: Colors.grey),
 main
        ],
      ),
    );
  }

  Widget _buildTrendSection() {
    return Container(
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12.r),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.trending_up, color: Colors.green, size: 20),
              SizedBox(width: 6),
              Text('이번 주 건강 트렌드', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            ],
          ),
          SizedBox(height: 10),
          _buildTrendRow('평균 활동량', 0.82, Colors.green, '82%'),
          _buildTrendRow('체중 관리', 0.95, Colors.blue, '95%'),
          _buildTrendRow('스트레스 관리', 0.88, Colors.purple, '88%'),
          SizedBox(height: 10),
          Container(
            width: double.infinity,
            padding: EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: Colors.green[50],
              borderRadius: BorderRadius.circular(8.r),
              border: Border.all(color: Colors.green[100]!),
            ),
            child: Text(
              '🎉 콩이는 이번 주 매우 건강하게 지냈어요! 활동량과 식사 패턴이 안정적입니다.',
              style: TextStyle(color: Colors.green, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildTrendRow(String label, double value, Color color, String percent) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(flex: 3, child: Text(label, style: TextStyle(fontSize: 13))),
          Expanded(
            flex: 7,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10.r),
              child: LinearProgressIndicator(
                value: value,
                backgroundColor: Colors.grey[200],
                valueColor: AlwaysStoppedAnimation<Color>(color),
                minHeight: 8,
              ),
            ),
          ),
          SizedBox(width: 8),
          Text(percent, style: TextStyle(fontSize: 12, color: color, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
