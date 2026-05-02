from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import streamlit as st
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

client = OpenAI()

# -----------------------------
# GPT 리포트 생성
# -----------------------------
def generate_report(student, content, understanding, attitude, strengths):
    prompt = f"""
다음 정보를 기반으로 학부모용 수업 리포트를 작성하라.

[학생명] {student}
[수업 내용] {content}
[이해도] {understanding}
[태도] {attitude}
[강점] {strengths}

요구사항:
- 반드시 아래 형식을 유지할 것

[수업 내용]
- ...

[학습 평가]
- ...

[강점]
- ...

[보완 필요]
- ...

[다음 수업 계획]
- ...

- 1:1 수업 기준으로 작성
- "동료 학생", "다른 학생" 표현 금지
- 자연스럽고 전문적인 한국어
- to 부정사의 개념을 명확히 이해하고 있으며, 실전문제 적용능력 우수합니다.

🔥 감성 문체 규칙:
- 전체적으로 부드럽고 긍정적인 톤
- 학부모가 안심할 수 있는 표현 사용
- “~보여주었습니다”, “~확인할 수 있었습니다” 형태 사용
- 보완 필요 항목은 지적이 아니라 “성장 방향”처럼 표현
- 절대 부정적 표현 금지 (“부족”, “문제” 등)
- 잘된 점은 자연스럽게 강조


- 각 문장은 줄바꿈으로 구분
"""
    
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content


# -----------------------------
# PDF 생성
# -----------------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import tempfile
from datetime import datetime

def create_pdf(text, student):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    # 🔥 폰트
    pdfmetrics.registerFont(TTFont("Malgun", "malgun.ttf"))
    
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()

    # 🔥 스타일 정의
    title_style = ParagraphStyle(
        name="title",
        fontName="Malgun",
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20
    )

    normal_style = ParagraphStyle(
        name="normal",
        fontName="Malgun",
        fontSize=11,
        leading=16
    )

    story = []

    # 🔥 제목
    story.append(Paragraph("수업 리포트", title_style))
    story.append(Spacer(1, 10))

    # 🔥 날짜
    today = datetime.today().strftime("%Y-%m-%d")

    # 🔥 학생정보 라인
    info_line = f"학생명: {student}    |    수업일: {today}"
    story.append(Paragraph(info_line, normal_style))
    story.append(Spacer(1, 10))

    # 🔥 구분선 느낌 (텍스트로)
    story.append(Paragraph("────────────────────────────", normal_style))
    story.append(Spacer(1, 15))

    # 🔥 본문 (줄 단위 처리)
    for line in text.split("\n"):
        if line.strip() == "":
            story.append(Spacer(1, 10))
        else:
            story.append(Paragraph(line, normal_style))
            story.append(Spacer(1, 8))

    doc.build(story)

    return tmp.name




# -----------------------------
# UI
# -----------------------------
def report_ui():
    st.title("📘 수업 리포트 생성기")

    student = st.text_input("학생명")

    # Reading 결과 자동 반영
    if "reading_result" in st.session_state:
        default_content = st.session_state["reading_result"]
        st.info("Reading 결과 자동 입력됨")
    else:
        default_content = ""

    content = st.text_area("수업 내용", value=default_content)

    col1, col2 = st.columns(2)

    with col1:
        understanding = st.radio("이해도", ["높음", "보통", "부족"])

    with col2:
        attitude = st.radio("태도", ["적극적", "보통", "소극적"])

    strengths = st.selectbox("강점", ["집중력 좋음", "이해 빠름", "참여도 높음"])

    # 리포트 생성
    if st.button("리포트 생성"):
        report = generate_report(
            student,
            content,
            understanding,
            attitude,
            strengths
        )
        st.session_state["report_text"] = report

    # 결과 출력 + PDF
    if "report_text" in st.session_state:
        st.write("학부모님께,")
        st.write(st.session_state["report_text"])

        pdf_path = create_pdf(st.session_state["report_text"], student)
        
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 PDF 다운로드",
                f,
                file_name="report.pdf"
            )
