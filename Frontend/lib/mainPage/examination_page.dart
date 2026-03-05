import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:convert';

class ExaminationPage extends StatefulWidget {
  final Map<String, dynamic>? petData;
  const ExaminationPage({super.key, this.petData});

  @override
  State<ExaminationPage> createState() => _ExaminationPageState();
}

class _ExaminationPageState extends State<ExaminationPage> {
  final ImagePicker _picker = ImagePicker();
  
  // For selecting images
  XFile? _eyeImage;
  XFile? _skinImage;
  
  bool _isUploading = false;

  Future<void> _pickAndUploadImage(String diseaseType) async {
    final XFile? pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile == null) return;

    setState(() {
      if (diseaseType == 'eye') _eyeImage = pickedFile;
      if (diseaseType == 'skin') _skinImage = pickedFile;
      _isUploading = true;
    });

    // 펫 타입 가져오기 (기본값 설정)
    String petType = widget.petData?['pet_type'] ?? 'unknown';

    // 백엔드 API 주소 (Docker 환경에서 로컬 접속)
    var uri = Uri.parse('http://localhost:8000/api/analyze-disease');
    var request = http.MultipartRequest('POST', uri);

    // AI 모델에게 함께 보낼 메타데이터: 펫 종류와 질환 종류 (안구/피부)
    request.fields['pet_type'] = petType;
    request.fields['disease_type'] = diseaseType; // 'eye' 또는 'skin'

    // 웹 환경과 모바일(앱/데스크탑) 환경 호환을 위한 분기 처리
    if (kIsWeb) {
      request.files.add(http.MultipartFile.fromBytes(
        'file', 
        await pickedFile.readAsBytes(),
        filename: pickedFile.name,
      ));
    } else {
      request.files.add(await http.MultipartFile.fromPath('file', pickedFile.path));
    }

    try {
      // 서버 전송 실행
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);
      
      if (!mounted) return;
      
      if (response.statusCode == 200) {
        // UTF-8로 디코딩
        final decodedBody = utf8.decode(response.bodyBytes);
        final Map<String, dynamic> result = json.decode(decodedBody);
        
        if (result['status'] == 'success') {
          // 다이얼로그로 결과 표시
          showDialog(
            context: context,
            builder: (BuildContext context) {
              return AlertDialog(
                title: Text('😸 AI 분석 결과'),
                content: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('질환 분류: ${result['disease_category'].toString().contains('eye') ? '안구' : '피부'}'),
                    Text('발견된 증상: ${result['diagnosis']}'),
                    Text('분석 정확도: ${result['probability'].toStringAsFixed(1)}%'),
                  ],
                ),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: const Text('확인'),
                  ),
                ],
              );
            },
          );
        } else {
          // 서버에서 에러 응답
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('분석 실패: ${result['message']}')),
          );
        }
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('서버 에러: ${response.statusCode}')),
        );
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('서버 연결 오류: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _isUploading = false);
      }
    }
  }

  Widget _buildDiseaseSection(String title, String diseaseType, XFile? selectedImage, IconData placeholderIcon) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      margin: const EdgeInsets.symmetric(vertical: 12),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title, 
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87)
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  flex: 4,
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.grey[100],
                        borderRadius: BorderRadius.circular(15),
                        border: Border.all(color: Colors.grey[300]!),
                      ),
                      clipBehavior: Clip.hardEdge,
                      child: selectedImage != null 
                        ? (kIsWeb 
                            ? Image.network(selectedImage.path, fit: BoxFit.cover) 
                            : Image.file(File(selectedImage.path), fit: BoxFit.cover)) // 앱 환경일 때
                        : Center(child: Icon(placeholderIcon, size: 40, color: Colors.grey[400])),
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  flex: 6,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      ElevatedButton.icon(
                        onPressed: _isUploading ? null : () => _pickAndUploadImage(diseaseType),
                        icon: _isUploading 
                          ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                          : const Icon(Icons.upload_file),
                        label: Text(_isUploading ? '분석 중...' : '사진 올리고 분석하기'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.purple[400],
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                      ),
                      const SizedBox(height: 12),
                      const Text(
                        '환부의 사진을 밝고 선명하게 찍어 업로드해주세요.',
                        style: TextStyle(fontSize: 11, color: Colors.grey),
                        textAlign: TextAlign.center,
                      )
                    ],
                  ),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    String petName = widget.petData?['pet_name'] ?? '반려동물';
    String petType = widget.petData?['pet_type'] ?? '정보없음';
    
    return Scaffold(
      backgroundColor: Colors.white,
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
                gradient: LinearGradient(colors: [Colors.green[400]!, Colors.teal[400]!]),
              ),
              child: Row(
                children: [
                  const Icon(Icons.health_and_safety, color: Colors.white, size: 40),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('AI 질환 검진소', style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.bold)),
                        const SizedBox(height: 4),
                        Text('현재 종: $petType', style: const TextStyle(color: Colors.white70, fontSize: 13)),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            Text(
              '$petName의 걱정되는 부위가 있으신가요?',
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87),
            ),
            const SizedBox(height: 8),
            const Text(
              '사진을 업로드하여 안구 및 피부 질환을 AI로 검사받으세요.',
              style: TextStyle(fontSize: 14, color: Colors.grey),
            ),
            const SizedBox(height: 24),
            _buildDiseaseSection('👁️ 안구 질환 (Eye Disease)', 'eye', _eyeImage, Icons.visibility),
            _buildDiseaseSection('🩹 피부 질환 (Skin Disease)', 'skin', _skinImage, Icons.healing),
          ],
        ),
      ),
    );
  }
}
