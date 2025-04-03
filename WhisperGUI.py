import sys
import os
import whisper
import torch
import re
import threading
import nltk
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from googletrans import Translator
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                            QFileDialog, QComboBox, QProgressBar, QTextEdit,
                            QVBoxLayout, QHBoxLayout, QWidget, QGroupBox,
                            QRadioButton, QButtonGroup, QMessageBox, QCheckBox,
                            QSpinBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Download NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class WhisperThread(QThread):
    progress_update = pyqtSignal(str)
    process_complete = pyqtSignal(tuple)
    
    def __init__(self, file_path, language, model_size, output_formats, 
                 summarize=False, summary_ratio=0.3, 
                 translate=False, target_language=None):
        super().__init__()
        self.file_path = file_path
        self.language = language if language else None
        self.model_size = model_size
        self.output_formats = output_formats
        self.summarize = summarize
        self.summary_ratio = summary_ratio
        self.translate = translate
        self.target_language = target_language
        
    def run(self):
        try:
            # GPU 확인
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_model = torch.cuda.get_device_name(0)
                self.progress_update.emit(f"CUDA 사용 가능: {cuda_available}\nGPU 모델: {gpu_model}")
            else:
                self.progress_update.emit("GPU를 사용할 수 없습니다. CPU로 처리합니다.")
            
            # 모델 로드
            self.progress_update.emit(f"{self.model_size} 모델 로딩 중...")
            model = whisper.load_model(self.model_size)
            self.progress_update.emit("모델 로딩 완료!")
            
            # 파일 처리
            self.progress_update.emit(f"파일 처리 중: {self.file_path}")
            result = model.transcribe(self.file_path, language=self.language)
            
            # 출력 파일 경로 설정
            output_dir = os.path.dirname(self.file_path)
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            
            # 전체 텍스트 추출
            transcript_text = result["text"]
            
            # 요약 생성 (요청된 경우)
            summary_text = None
            if self.summarize:
                self.progress_update.emit("텍스트 요약 생성 중...")
                summary_text = self.generate_summary(transcript_text, self.summary_ratio)
                self.progress_update.emit("요약 생성 완료!")
            
            # 번역 (요청된 경우)
            translated_text = None
            translated_segments = None
            if self.translate and self.target_language:
                self.progress_update.emit(f"{self.target_language}로 번역 중...")
                translator = Translator()
                
                # 전체 텍스트 번역
                translated_text = translator.translate(
                    transcript_text, 
                    dest=self.target_language
                ).text
                
                # 요약 번역 (요약이 있는 경우)
                translated_summary = None
                if summary_text:
                    translated_summary = translator.translate(
                        summary_text, 
                        dest=self.target_language
                    ).text
                
                # 세그먼트 번역
                translated_segments = []
                for segment in result["segments"]:
                    translated_segment = segment.copy()
                    translated_segment["text"] = translator.translate(
                        segment["text"].strip(), 
                        dest=self.target_language
                    ).text
                    translated_segments.append(translated_segment)
                
                self.progress_update.emit("번역 완료!")
            
            # 출력 파일 생성
            created_files = []
            
            # SRT 파일 생성 (요청된 경우)
            if "srt" in self.output_formats:
                srt_file = os.path.join(output_dir, base_name + ".srt")
                self.create_srt(result["segments"], srt_file)
                created_files.append(srt_file)
                self.progress_update.emit(f"SRT 자막 파일이 생성되었습니다: {srt_file}")
                
                # 번역된 SRT 생성 (요청된 경우)
                if self.translate and translated_segments:
                    translated_srt_file = os.path.join(output_dir, base_name + f"_{self.target_language}.srt")
                    self.create_srt(translated_segments, translated_srt_file)
                    created_files.append(translated_srt_file)
                    self.progress_update.emit(f"번역된 SRT 자막 파일이 생성되었습니다: {translated_srt_file}")
            
            # TXT 파일 생성 (요청된 경우)
            if "txt" in self.output_formats:
                txt_file = os.path.join(output_dir, base_name + ".txt")
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                created_files.append(txt_file)
                self.progress_update.emit(f"텍스트 파일이 생성되었습니다: {txt_file}")
                
                # 요약 TXT 파일 생성 (요청된 경우)
                if self.summarize and summary_text:
                    summary_txt_file = os.path.join(output_dir, base_name + "_summary.txt")
                    with open(summary_txt_file, "w", encoding="utf-8") as f:
                        f.write(summary_text)
                    created_files.append(summary_txt_file)
                    self.progress_update.emit(f"요약 텍스트 파일이 생성되었습니다: {summary_txt_file}")
                
                # 번역된 TXT 파일 생성 (요청된 경우)
                if self.translate and translated_text:
                    translated_txt_file = os.path.join(output_dir, base_name + f"_{self.target_language}.txt")
                    with open(translated_txt_file, "w", encoding="utf-8") as f:
                        f.write(translated_text)
                    created_files.append(translated_txt_file)
                    self.progress_update.emit(f"번역된 텍스트 파일이 생성되었습니다: {translated_txt_file}")
                    
                    # 번역된 요약 TXT 파일 생성 (요약과 번역 모두 요청된 경우)
                    if self.summarize and translated_summary:
                        translated_summary_txt_file = os.path.join(output_dir, base_name + f"_summary_{self.target_language}.txt")
                        with open(translated_summary_txt_file, "w", encoding="utf-8") as f:
                            f.write(translated_summary)
                        created_files.append(translated_summary_txt_file)
                        self.progress_update.emit(f"번역된 요약 텍스트 파일이 생성되었습니다: {translated_summary_txt_file}")
            
            # PDF 파일 생성 (요청된 경우)
            if "pdf" in self.output_formats:
                if "srt" in self.output_formats:
                    # SRT 파일이 이미 생성된 경우 사용
                    srt_file = os.path.join(output_dir, base_name + ".srt")
                else:
                    # SRT 파일이 생성되지 않은 경우 임시로 생성
                    srt_file = os.path.join(output_dir, "_temp_" + base_name + ".srt")
                    self.create_srt(result["segments"], srt_file)
                
                pdf_file = os.path.join(output_dir, base_name + "_transcript.pdf")
                self.srt_to_pdf(srt_file, pdf_file, transcript_text, summary_text if self.summarize else None)
                created_files.append(pdf_file)
                
                # 임시 SRT 파일 삭제
                if "srt" not in self.output_formats and os.path.exists(srt_file) and "_temp_" in srt_file:
                    os.remove(srt_file)
                
                # 번역된 PDF 생성 (요청된 경우)
                if self.translate and translated_segments:
                    if "srt" in self.output_formats:
                        # 번역된 SRT 파일이 이미 생성된 경우 사용
                        translated_srt_file = os.path.join(output_dir, base_name + f"_{self.target_language}.srt")
                    else:
                        # 번역된 SRT 파일이 생성되지 않은 경우 임시로 생성
                        translated_srt_file = os.path.join(output_dir, "_temp_" + base_name + f"_{self.target_language}.srt")
                        self.create_srt(translated_segments, translated_srt_file)
                    
                    translated_pdf_file = os.path.join(output_dir, base_name + f"_transcript_{self.target_language}.pdf")
                    self.srt_to_pdf(translated_srt_file, translated_pdf_file, translated_text, 
                                   translated_summary if (self.summarize and 'translated_summary' in locals()) else None)
                    created_files.append(translated_pdf_file)
                    
                    # 임시 번역 SRT 파일 삭제
                    if "srt" not in self.output_formats and os.path.exists(translated_srt_file) and "_temp_" in translated_srt_file:
                        os.remove(translated_srt_file)
            
            # 완료 시그널 발생
            self.process_complete.emit((created_files, transcript_text))
            
        except Exception as e:
            self.progress_update.emit(f"오류 발생: {str(e)}")
            self.process_complete.emit(([], None))
    
    def format_time(self, seconds):
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = seconds % 60
        millisecs = int((secs - int(secs)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millisecs:03d}"
    
    def create_srt(self, segments, output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                f.write(f"{i+1}\n")
                start_time = self.format_time(segment["start"])
                end_time = self.format_time(segment["end"])
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text'].strip()}\n\n")
        return output_file
    
    def generate_summary(self, text, ratio=0.3):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        
        # 문장 수 계산
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        summary_sentence_count = max(1, int(sentence_count * ratio))
        
        # 요약 생성
        summary = summarizer(parser.document, summary_sentence_count)
        summary_text = ' '.join([str(sentence) for sentence in summary])
        
        return summary_text
    
    def srt_to_pdf(self, srt_file, pdf_file, full_text=None, summary_text=None):
        self.progress_update.emit(f"SRT 파일을 PDF로 변환 중: {pdf_file}")
        
        # SRT 파일 읽기
        with open(srt_file, 'r', encoding='utf-8') as file:
            srt_content = file.read()
        
        # SRT 내용 파싱
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
        matches = re.findall(pattern, srt_content, re.DOTALL)
        
        # PDF 생성 시작
        c = canvas.Canvas(pdf_file, pagesize=letter)
        width, height = letter
        
        # PDF 제목 추가
        c.setFont("Helvetica-Bold", 16)
        title = os.path.basename(srt_file).replace('.srt', '') + ' - 자막 텍스트'
        c.drawString(72, height - 50, title)
        
        # 요약 및 전체 텍스트 추가 (제공된 경우)
        y_position = height - 80
        line_height = 15
        
        if summary_text:
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, y_position, "요약:")
            y_position -= line_height * 1.5
            
            c.setFont("Helvetica", 12)
            # 요약 텍스트를 여러 줄로 나눔
            summary_lines = self._wrap_text(summary_text, 60)
            for line in summary_lines:
                c.drawString(80, y_position, line)
                y_position -= line_height
            
            y_position -= line_height * 2  # 요약과 타임스탬프 사이 간격
        
        # 타임스탬프가 있는 세부 텍스트 추가
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y_position, "세부 내용:")
        y_position -= line_height * 1.5
        
        # 내용 추가
        for match in matches:
            number, start_time, end_time, text = match
            
            # 페이지 넘기기
            if y_position < 100:
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = height - 50
            
            time_text = f"[{start_time} - {end_time}]"
            c.setFont("Helvetica-Bold", 10)
            c.drawString(72, y_position, time_text)
            y_position -= line_height
            
            # 텍스트 추가 (긴 텍스트는 여러 줄로 분할)
            c.setFont("Helvetica", 12)
            text_lines = self._wrap_text(text, 60)
            
            for line in text_lines:
                # 페이지 넘기기
                if y_position < 72:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = height - 50
                
                c.drawString(80, y_position, line)
                y_position -= line_height
            
            y_position -= line_height  # 문단 간격 추가
        
        c.save()
        self.progress_update.emit(f"PDF 파일이 생성되었습니다: {pdf_file}")
        return pdf_file
    
    def _wrap_text(self, text, max_width):
        """텍스트를 지정된 너비로 래핑합니다."""
        lines = []
        current_line = ""
        words = text.split()
        
        for word in words:
            if len(current_line + " " + word) < max_width:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines


class WhisperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper 자막 변환 도구")
        self.setGeometry(100, 100, 900, 700)  # 창 크기 증가
        self.initUI()
        
    def initUI(self):
        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # 파일 선택 섹션
        file_group = QGroupBox("파일 선택")
        file_layout = QHBoxLayout()
        
        self.file_path_label = QLabel("선택된 파일 없음")
        self.browse_button = QPushButton("파일 찾기")
        self.browse_button.clicked.connect(self.browse_file)
        
        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(self.browse_button)
        file_group.setLayout(file_layout)
        
        # 옵션 섹션
        options_group = QGroupBox("옵션 설정")
        options_layout = QGridLayout()
        
        # 왼쪽 패널: 언어 및 모델 선택
        # 언어 선택 드롭다운
        language_label = QLabel("원본 언어 선택:")
        self.language_combo = QComboBox()
        self.language_combo.addItem("자동 감지", None)
        self.language_combo.addItem("한국어", "ko")
        self.language_combo.addItem("영어", "en")
        self.language_combo.addItem("일본어", "ja")
        self.language_combo.addItem("중국어", "zh")
        
        options_layout.addWidget(language_label, 0, 0)
        options_layout.addWidget(self.language_combo, 0, 1)
        
        # 모델 크기 선택 라디오 버튼
        model_label = QLabel("모델 크기:")
        self.model_group = QButtonGroup()
        
        self.model_tiny = QRadioButton("tiny (가장 빠름, 낮은 정확도)")
        self.model_base = QRadioButton("base")
        self.model_small = QRadioButton("small")
        self.model_medium = QRadioButton("medium (권장)")
        self.model_large = QRadioButton("large (가장 느림, 높은 정확도)")
        
        self.model_group.addButton(self.model_tiny, 1)
        self.model_group.addButton(self.model_base, 2)
        self.model_group.addButton(self.model_small, 3)
        self.model_group.addButton(self.model_medium, 4)
        self.model_group.addButton(self.model_large, 5)
        
        self.model_medium.setChecked(True)  # 기본값
        
        options_layout.addWidget(model_label, 1, 0)
        options_layout.addWidget(self.model_tiny, 1, 1)
        options_layout.addWidget(self.model_base, 2, 1)
        options_layout.addWidget(self.model_small, 3, 1)
        options_layout.addWidget(self.model_medium, 4, 1)
        options_layout.addWidget(self.model_large, 5, 1)
        
        # 오른쪽 패널: 출력 옵션
        # 출력 형식 선택
        output_label = QLabel("출력 형식 (여러 개 선택 가능):")
        
        self.txt_check = QCheckBox("텍스트 파일 (.txt)")
        self.srt_check = QCheckBox("자막 파일 (.srt)")
        self.pdf_check = QCheckBox("PDF 문서 (.pdf)")
        
        self.txt_check.setChecked(True)
        self.srt_check.setChecked(True)
        self.pdf_check.setChecked(True)
        
        options_layout.addWidget(output_label, 0, 2)
        options_layout.addWidget(self.txt_check, 1, 2)
        options_layout.addWidget(self.srt_check, 2, 2)
        options_layout.addWidget(self.pdf_check, 3, 2)
        
        # 요약 옵션
        self.summarize_check = QCheckBox("텍스트 요약 생성")
        
        summary_ratio_layout = QHBoxLayout()
        summary_ratio_layout.addWidget(QLabel("요약 비율:"))
        self.summary_ratio_spin = QSpinBox()
        self.summary_ratio_spin.setRange(10, 90)
        self.summary_ratio_spin.setValue(30)
        self.summary_ratio_spin.setSuffix("%")
        summary_ratio_layout.addWidget(self.summary_ratio_spin)
        summary_ratio_layout.addStretch()
        
        self.summarize_check.toggled.connect(lambda checked: self.summary_ratio_spin.setEnabled(checked))
        self.summary_ratio_spin.setEnabled(False)
        
        options_layout.addWidget(self.summarize_check, 4, 2)
        options_layout.addLayout(summary_ratio_layout, 5, 2)
        
        # 번역 옵션
        self.translate_check = QCheckBox("번역 생성")
        
        translate_lang_layout = QHBoxLayout()
        translate_lang_layout.addWidget(QLabel("번역 언어:"))
        self.translate_lang_combo = QComboBox()
        self.translate_lang_combo.addItem("한국어", "ko")
        self.translate_lang_combo.addItem("영어", "en")
        self.translate_lang_combo.addItem("일본어", "ja")
        self.translate_lang_combo.addItem("중국어", "zh")
        translate_lang_layout.addWidget(self.translate_lang_combo)
        translate_lang_layout.addStretch()
        
        self.translate_check.toggled.connect(lambda checked: self.translate_lang_combo.setEnabled(checked))
        self.translate_lang_combo.setEnabled(False)
        
        options_layout.addWidget(self.translate_check, 6, 2)
        options_layout.addLayout(translate_lang_layout, 7, 2)
        
        options_group.setLayout(options_layout)
        
        # 실행 버튼
        self.process_button = QPushButton("변환 시작")
        self.process_button.setStyleSheet("font-size: 14pt; height: 40px;")
        self.process_button.clicked.connect(self.process_file)
        
        # 진행 상황 표시
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 무한 로딩 표시
        self.progress_bar.setVisible(False)
        
        # 로그 출력
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        # 모든 위젯을 메인 레이아웃에 추가
        main_layout.addWidget(file_group)
        main_layout.addWidget(options_group)
        main_layout.addWidget(self.process_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(log_group, 1)  # 로그 영역에 더 많은 공간 할당
        
        # 상태 표시줄
        self.statusBar().showMessage("준비")
        
    def browse_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "변환할 파일 선택", "", 
            "모든 파일 (*);;비디오 파일 (*.mp4 *.avi *.mkv *.mov);;오디오 파일 (*.mp3 *.wav *.ogg *.flac)", 
            options=options
        )
        
        if file_path:
            self.file_path_label.setText(file_path)
            self.log_text.append(f"파일 선택됨: {file_path}")
    
    def process_file(self):
        file_path = self.file_path_label.text()
        
        if file_path == "선택된 파일 없음":
            QMessageBox.warning(self, "경고", "파일을 선택해주세요!")
            return
        
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "경고", "파일을 찾을 수 없습니다!")
            return
        
        # 출력 형식 확인
        output_formats = []
        if self.txt_check.isChecked():
            output_formats.append("txt")
        if self.srt_check.isChecked():
            output_formats.append("srt")
        if self.pdf_check.isChecked():
            output_formats.append("pdf")
            
        if not output_formats:
            QMessageBox.warning(self, "경고", "하나 이상의 출력 형식을 선택해주세요!")
            return
        
        # 언어 선택
        language = self.language_combo.currentData()
        
        # 모델 크기 선택
        model_id = self.model_group.checkedId()
        if model_id == 1:
            model_size = "tiny"
        elif model_id == 2:
            model_size = "base"
        elif model_id == 3:
            model_size = "small"
        elif model_id == 4:
            model_size = "medium"
        elif model_id == 5:
            model_size = "large"
        else:
            model_size = "medium"  # 기본값
        
        # 요약 옵션
        summarize = self.summarize_check.isChecked()
        summary_ratio = self.summary_ratio_spin.value() / 100.0
        
        # 번역 옵션
        translate = self.translate_check.isChecked()
        target_language = self.translate_lang_combo.currentData() if translate else None
        
        # UI 상태 업데이트
        self.process_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("처리 중...")
        
        # 로그 초기화 및 시작 메시지
        self.log_text.clear()
        self.log_text.append("==== 처리 시작 ====")
        self.log_text.append(f"파일: {file_path}")
        self.log_text.append(f"언어: {language if language else '자동 감지'}")
        self.log_text.append(f"모델 크기: {model_size}")
        self.log_text.append(f"출력 형식: {', '.join(output_formats)}")
        self.log_text.append(f"요약 생성: {'예' if summarize else '아니오'}{' (비율: ' + str(summary_ratio * 100) + '%)' if summarize else ''}")
        self.log_text.append(f"번역 생성: {'예' if translate else '아니오'}{' (언어: ' + target_language + ')' if translate else ''}")
        self.log_text.append("-------------------")
        
        # 처리 스레드 시작
        self.worker_thread = WhisperThread(
            file_path, language, model_size, output_formats, 
            summarize, summary_ratio, translate, target_language
        )
        self.worker_thread.progress_update.connect(self.update_log)
        self.worker_thread.process_complete.connect(self.process_completed)
        self.worker_thread.start()
    
    def update_log(self, message):
        self.log_text.append(message)
        # 스크롤을 항상 아래로 유지
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def process_completed(self, result):
        created_files, transcript_text = result
        
        # UI 상태 복원
        self.process_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if created_files:
            self.log_text.append("==== 처리 완료 ====")
            for file in created_files:
                self.log_text.append(f"생성된 파일: {file}")
            self.statusBar().showMessage("완료")
            
            # 완료 메시지 표시
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("처리 완료")
            msg_box.setText("자막 변환이 완료되었습니다!")
            
            # 파일 목록 생성
            files_text = ""
            for file in created_files:
                files_text += f"- {file}\n"
            
            msg_box.setInformativeText(f"생성된 파일:\n{files_text}")
            
            open_folder_button = msg_box.addButton("폴더 열기", QMessageBox.ActionRole)
            msg_box.addButton("확인", QMessageBox.AcceptRole)
            
            msg_box.exec_()
            
            # 폴더 열기 버튼이 클릭된 경우
            if msg_box.clickedButton() == open_folder_button:
                os.startfile(os.path.dirname(created_files[0]))
        else:
            self.log_text.append("==== 처리 실패 ====")
            self.statusBar().showMessage("실패")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WhisperGUI()
    window.show()
    sys.exit(app.exec_())