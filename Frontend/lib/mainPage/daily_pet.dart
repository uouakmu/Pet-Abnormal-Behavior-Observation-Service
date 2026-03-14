import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_screenutil/flutter_screenutil.dart';
import 'package:http/http.dart' as http;

class daily_pet extends StatefulWidget {
  final Map<String, dynamic>? petData;
  final String? userId; // 로그인한 유저 ID

  const daily_pet({super.key, required this.petData, this.userId});

  @override
  State<daily_pet> createState() => _DailyPetState();
}

class _DailyPetState extends State<daily_pet> {
  bool _isSaving = false;
  String? _diaryContent;
  String? _errorMessage;

  // 테스트용 더미 데이터 (실제는 분석 결과에서 받아오면 됨)
  final Map<String, dynamic> _dummyAnalysisResult = {
    "status": "success",
    "behavior_analysis": {
      "detected_behavior": "walking",
      "emotion": "happy"
    },
    "audio_analysis": {
      "detected_sound": "barking"
    },
    "patella_analysis": {
      "status": "정상"
    }
  };

  Future<void> _saveDiaryAndGenerate() async {
    final String userId = widget.userId ?? 'test_user';
    final String petType = widget.petData?['pet_type'] ?? 'dog';
    final String today = DateTime.now().toIso8601String().substring(0, 10); // YYYY-MM-DD
    final String baseUrl = 'http://localhost:8080';

    setState(() {
      _isSaving = true;
      _errorMessage = null;
      _diaryContent = null;
    });

    try {
      // 1. 전체 시뮬레이션 요청 (24개 데이터 저장 + 일기 생성)
      //    이 엔드포인트는 백엔드에서 sample1.mp4를 24개로 나눠 분석하고 Firebase에 저장한 뒤 일기를 생성합니다.
      final response = await http.post(
        Uri.parse('$baseUrl/api/simulate-full-day'),
        body: {
          "user_id": userId,
          "pet_type": petType == '강아지' ? 'dog' : 'cat', // 엔진 호환을 위해 변환
        },
      );
      
      if (response.statusCode != 200) {
        throw Exception('시뮬레이션 요청 실패: ${response.body}');
      }

      final resultJson = jsonDecode(utf8.decode(response.bodyBytes));
      if (resultJson['status'] == 'error') {
        throw Exception(resultJson['message']);
      }

      setState(() {
        _diaryContent = resultJson['diary'] ?? '일기 내용을 불러올 수 없습니다.';
      });

      if (context.mounted) {
        _showDiaryDialog(_diaryContent!);
      }
    } catch (e) {
      setState(() {
        _errorMessage = e.toString();
      });
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('오류: $_errorMessage'), backgroundColor: Colors.red),
        );
      }
    } finally {
      setState(() {
        _isSaving = false;
      });
    }
  }

  void _showDiaryDialog(String content) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('📖 오늘의 일기가 생성됐어요!'),
        content: SingleChildScrollView(
          child: Text(content, style: const TextStyle(height: 1.6)),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('확인'),
          )
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final String petName = widget.petData?['pet_name'] ?? '콩이';
    return Scaffold(
      backgroundColor: const Color(0xFFF9F9F9),
      appBar: AppBar(
        backgroundColor: Colors.blue,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: () => Navigator.pop(context),
        ),
        title: Column(
          children: [
            const Text('일상 행동 일기', style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
            const Text('2026년 3월 11일 화요일', style: TextStyle(color: Colors.white70, fontSize: 12)),
          ],
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildSectionTitle('🐾 오늘 하루 $petName 기분!'),
                  _buildMoodCard(),
                  const SizedBox(height: 24),

                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _buildSectionTitle('📸 오늘의 순간들'),
                      const Text('대표 사진', style: TextStyle(color: Colors.orange, fontSize: 10)),
                    ],
                  ),
                  _buildPhotoGallery(),
                  const SizedBox(height: 24),

                  _buildSectionTitle('💬 $petName야 오늘은 어땠어?'),
                  _buildAISummaryCard(petName),
                  const SizedBox(height: 24),

                  _buildTeacherAdviceCard(petName),
                  const SizedBox(height: 24),

                  _buildSectionTitle('보호자 메모'),
                  _buildMemoField(),
                  const SizedBox(height: 30),

                  _buildBottomButton(context),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.black87)),
    );
  }

  Widget _buildMoodCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF8F0),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('평균 기분', style: TextStyle(color: Colors.grey, fontSize: 12)),
              Text('즐거움', style: TextStyle(color: Colors.green, fontSize: 20, fontWeight: FontWeight.bold)),
            ],
          ),
          Icon(Icons.sentiment_satisfied_alt, color: Colors.green[300], size: 30),
        ],
      ),
    );
  }

  Widget _buildPhotoGallery() {
    return Column(
      children: [
        Container(
          height: 175.w,
          width: double.infinity,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(15),
            color: Colors.grey[300],
            image: const DecorationImage(
              image: NetworkImage('https://via.placeholder.com/600x400'),
              fit: BoxFit.cover,
            ),
          ),
        ),
        const SizedBox(height: 12),
        GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: 3,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 3,
            crossAxisSpacing: 8,
            mainAxisSpacing: 8,
            childAspectRatio: 1,
          ),
          itemBuilder: (context, index) {
            return Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                color: Colors.grey[300],
                image: const DecorationImage(
                  image: NetworkImage('https://via.placeholder.com/150'),
                  fit: BoxFit.cover,
                ),
              ),
            );
          },
        ),
      ],
    );
  }

  Widget _buildAISummaryCard(String petName) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF0F5FF),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        _diaryContent ?? '오늘 $petName는 아침 8시 30분에 맛있게 밥을 먹었어요! 그리고 10시에는 공원에서 신나게 뛰어놀았네요. 오후에는 편안하게 낮잠을 자고...',
        style: const TextStyle(color: Colors.blueGrey, fontSize: 13, height: 1.5),
      ),
    );
  }

  Widget _buildTeacherAdviceCard(String petName) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: const LinearGradient(colors: [Colors.purpleAccent, Colors.pinkAccent]),
        borderRadius: BorderRadius.circular(15),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const CircleAvatar(radius: 15, backgroundColor: Colors.white, child: Icon(Icons.person, size: 20)),
              const SizedBox(width: 10),
              Text('$petName의 하루!', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(
              '$petName가 오늘 활발하게 활동했네요! 규칙적인 식사와 충분한 운동, 휴식이 잘 이루어지고 있습니다.',
              style: const TextStyle(color: Colors.white, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMemoField() {
    return TextField(
      maxLines: 4,
      decoration: InputDecoration(
        hintText: '오늘 반려동물과 함께한 특별한 순간을 기록해주세요...',
        hintStyle: const TextStyle(fontSize: 13, color: Colors.grey),
        filled: true,
        fillColor: Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Colors.black12),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Colors.black12),
        ),
      ),
    );
  }

  Widget _buildBottomButton(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: _isSaving ? null : _saveDiaryAndGenerate,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.orange,
          disabledBackgroundColor: Colors.orange.shade200,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
        child: _isSaving
            ? const Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)),
                  SizedBox(width: 10),
                  Text('일기 생성 중...', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                ],
              )
            : const Text('일기 저장하기 →', style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
      ),
    );
  }
}