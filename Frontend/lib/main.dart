import 'package:flutter/material.dart';
import 'package:pet_diary/mainPage/total_diary.dart';
import 'package:pet_diary/mainPage/odd_pet.dart';
import 'package:pet_diary/mainPage/daily_pet.dart';
import 'package:pet_diary/discription/onboarding_page.dart';
import 'package:pet_diary/mainPage/examination_page.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:pet_diary/mainPage/mypage.dart';

void main() {
  runApp(const MaterialApp(
    home: OnboardingPage(),
    debugShowCheckedModeBanner: false,
  ));
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

  @override
  void initState() {
    super.initState();
    _fetchPetInfo(); // 함수 이름도 의미에 맞게 변경
  }
  Future<void> _fetchPetInfo() async {
    final url = Uri.parse('http://localhost:8000/user-pet-info/${widget.userId}');

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
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeaderCard(),
          const SizedBox(height: 16),
          Row(
            children: [
              _buildActionButton(Icons.book, '일상 일기', '기분 & 활동량', Colors.blue,
                      () => Navigator.push(context, MaterialPageRoute(builder: (context) => daily_pet()))),
              const SizedBox(width: 12),
              _buildActionButton(Icons.error_outline, '이상 행동', '건강 체크', Colors.orange,
                      () => Navigator.push(context, MaterialPageRoute(builder: (context) => PageB()))),
            ],
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '최근 일기',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              GestureDetector(
                onTap: () {
                  // DiaryListPage로 이동
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => DiaryListPage()),
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
          isLoading
              ? const Center(child: CircularProgressIndicator()) // 로딩 중이면 뱅글뱅글
              : petData == null
              ? const Center(child: Text('서버에 저장된 기록이 없습니다.')) // 데이터가 없으면 메시지
              : Column(
            children: [
              if (petData != null)
                _buildDiaryItem(
                  petData!['pet_birthday'] ?? '날짜 정보 없음', // 생일 데이터 활용
                  petData!['pet_name'] ?? '이름 없음',       // 이름 데이터 활용
                  85, // 활동량 (현재 예시 값)
                  false, // 주의사항 배지 여부
                )
              else
                const Center(child: Text('등록된 반려동물 정보가 없습니다.')),
            ],
          ),
          // ------------------------------------------

          const SizedBox(height: 24),
          _buildTrendSection(),
          const SizedBox(height: 32),
          const Center(
            child: Column(
              children: [
                Text('AI가 24시간 콩이를 모니터링하고 있어요', style: TextStyle(color: Colors.grey, fontSize: 12)),
                SizedBox(height: 4),
                Text('8가지 데이터셋 기반 건강 분석 시스템', style: TextStyle(color: Colors.grey, fontSize: 11)),
              ],
            ),
          ),
          const SizedBox(height: 40),
        ],
      ),
    );
  }

  // --- 헬퍼 함수들 ---

  Widget _buildHeaderCard() {
    String petName = petData?['pet_name'] ?? '콩이';
    String petType = petData?['pet_type'] ?? '반려동물';
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: const LinearGradient(colors: [Colors.purple, Colors.orangeAccent]),
      ),
      child: Column(
        children: [
          Row(
            children: [
              const CircleAvatar(radius: 25, backgroundColor: Colors.white),
              const SizedBox(width: 12),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('$petName의 건강일기', style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
                  const Text('AI 기반 반려동물 케어', style: TextStyle(color: Colors.white70, fontSize: 12)),
                ],
              )
            ],
          ),
          const SizedBox(height: 20),
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
        Text(value, style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
        Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
      ],
    );
  }

  Widget _buildActionButton(IconData icon, String title, String subTitle, Color color, VoidCallback onTap) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(15),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 20),
          decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(15)),
          child: Column(
            children: [
              Icon(icon, color: Colors.white, size: 30),
              const SizedBox(height: 8),
              Text(title, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
              Text(subTitle, style: const TextStyle(color: Colors.white70, fontSize: 10)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildDiaryItem(String date, String day, int activity, bool hasWarning) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Row(
        children: [
          Container(width: 50, height: 50, decoration: BoxDecoration(color: Colors.grey[300], borderRadius: BorderRadius.circular(8))),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(date, style: const TextStyle(fontWeight: FontWeight.bold)),
                Text(day, style: const TextStyle(color: Colors.grey, fontSize: 12)),
                Row(
                  children: [
                    const Icon(Icons.trending_up, size: 14, color: Colors.green),
                    Text(' 활동 $activity', style: const TextStyle(fontSize: 12)),
                    if (hasWarning) ...[
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(color: Colors.orange[100], borderRadius: BorderRadius.circular(4)),
                        child: const Text('주의사항', style: TextStyle(color: Colors.orange, fontSize: 10)),
                      )
                    ]
                  ],
                )
              ],
            ),
          ),
          const Icon(Icons.sentiment_satisfied_alt, color: Colors.lightGreen),
        ],
      ),
    );
  }

  Widget _buildTrendSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
        boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.trending_up, color: Colors.green, size: 20),
              SizedBox(width: 8),
              Text('이번 주 건강 트렌드', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
            ],
          ),
          const SizedBox(height: 16),
          _buildTrendRow('평균 활동량', 0.82, Colors.green, '82%'),
          _buildTrendRow('체중 관리', 0.95, Colors.blue, '95%'),
          _buildTrendRow('스트레스 관리', 0.88, Colors.purple, '88%'),
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.green[50],
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.green[100]!),
            ),
            child: const Text(
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
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        children: [
          Expanded(flex: 3, child: Text(label, style: const TextStyle(fontSize: 13))),
          Expanded(
            flex: 7,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: LinearProgressIndicator(
                value: value,
                backgroundColor: Colors.grey[200],
                valueColor: AlwaysStoppedAnimation<Color>(color),
                minHeight: 8,
              ),
            ),
          ),
          const SizedBox(width: 10),
          Text(percent, style: TextStyle(fontSize: 12, color: color, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
