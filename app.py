import re
import streamlit as st
import spacy
import re

def preprocess_legal_text(text: str) -> str:
    import re

    # 사건명 v. 보호
    text = re.sub(r'\bv\.\s', 'v__ ', text)

    # U.S. reporter citation 전체 제거
    text = re.sub(r'\b\d+\s+U\.\s*S\.\s+\d+(?:,\s*\d+)*(?:–\d+)?\s*\(\d{4}\)', ' ', text)
    text = re.sub(r'\(\s*[“"].*?[”"]\s*\)', ' ', text)

    # see, e.g., 제거
    text = re.sub(r'\bsee,\s*e\.g\.,', ' ', text, flags=re.IGNORECASE)

    # 여러 citation 연결 세미콜론 제거
    text = re.sub(r';', '. ', text)

    # 약어 보호
    replacements = {
        r"\bU\s*\.\s*S\s*\.\s*C\s*\.": "USC",
        r"\bU\s*\.\s*S\s*\.": "US",
        r"\bB\s*\.\s*C\s*\.": "BC",
        r"\bA\s*\.\s*D\s*\.": "AD",
        r"\bNo\s*\.": "No",
        r"\bInc\s*\.": "Inc",
        r"\bCo\s*\.": "Co",
        r"\bLtd\s*\.": "Ltd",
        r"\be\s*\.\s*g\s*\.": "eg",
        r"\bi\s*\.\s*e\s*\.": "ie",
        r"\bet al\s*\.": "et al",
        r"\bcf\s*\.": "cf",
        r"\bvs\s*\.": "vs",
    }

    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def render_load_badge(label, score=None):
    color_map = {
        "Light": "#2e7d32",      # green
        "Moderate": "#ef6c00",   # orange
        "Heavy": "#c62828",      # red
    }

    color = color_map.get(label, "#ef6c00")

    text = label if score is None else f"{label} ({score})"

    return f"""
    <span style="
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        background:{color};
        color:white;
        font-weight:700;
        font-size:0.9rem;
    ">
        {text}
    </span>
    """

from mini_os_v3 import (
    annotate_doc_with_clauses,
    compute_sentence_load,
    compute_document_load,
    compute_core_score
)

nlp = spacy.load("en_core_web_sm")


def split_sentences(text: str):
    """
    아주 단순한 v1 문장 분리:
    마침표/물음표/느낌표 기준
    """
    text = text.strip()
    if not text:
        return []

    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def hawk_render(text: str):
    # slash styling
    text = text.replace(" / ", ' <span class="slash">/</span> ')

    # main subject
    text = text.replace("«", '<span class="main-subj">')
    text = text.replace("»", "</span>")

    # main predicate
    text = text.replace("[[", '<span class="main-pred">')
    text = text.replace("]]", "</span>")

    # subordinate subject
    text = text.replace("‹", '<span class="sub-subj">')
    text = text.replace("›", "</span>")

    # subordinate predicate
    text = text.replace("{{", '<span class="sub-pred">')
    text = text.replace("}}", "</span>")

    # relative clause
    text = text.replace("[", '<span class="rel-bracket">[</span><span class="rel-text">')
    text = text.replace("]", '</span><span class="rel-bracket">]</span>')

    return text


st.set_page_config(page_title="Hawk Eyes – Reading OS", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]  {
  font-family: "Inter", "Segoe UI", sans-serif;
}

.main-subj {
  color: #00c853;
  font-weight: 700;
}

.sub-subj {
  color: #00ff88;
}

.main-pred {
  color: #ff4d4d;
  font-weight: 700;
}

.sub-pred {
  color: #ff8a8a;
}

.rel-bracket {
  color: #8a97a8;
  font-weight: 600;
}

.rel-text {
  color: #6f7c8f;
}

.slash {
  color: #8a97a8;
  font-weight: 700;
}

.hawk-line {
  font-size: 1.15rem;
  line-height: 1.9;
  padding: 0.45rem 0;
  max-width: 680px;
  white-space: normal;
  word-break: keep-all;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.8;
}

.block-label {
  color: #9aa4b2;
  font-size: 1rem;
  margin-top: 1rem;
  margin-bottom: 0.35rem;
  font-weight: 600;
}

.limit-note {
  color: #9aa4b2;
  font-size: 0.9rem;
}

.load-light {
    border-left: 4px solid #2e7d32;
    padding-left: 10px;
}

.load-moderate {
    border-left: 4px solid #ef6c00;
    padding-left: 10px;
}

.load-heavy {
    border-left: 4px solid #c62828;
    padding-left: 10px;         

.connector {
    color: #3b82f6;
    font-weight: 600;
}            

.hawk-line {
    margin: 8px 0 18px 0;
    padding: 8px 12px;
    border-radius: 6px;
    background: #fafafa;
    line-height: 1.6;
}

.block-text {
    max-width: 720px;
    line-height: 1.6;
    font-size: 17px;
}
    
</style>
""", unsafe_allow_html=True)

st.title("Hawk Eyes – Reading OS")
st.caption("Lite mode · structure-first reading")

st.markdown("""
<div style="
font-size:14px;
margin-bottom:10px;
padding:8px 12px;
border-radius:8px;
background:#f5f5f5;
">

<b>Legend</b><br>
<span style="color:#2ecc71;">●</span> Subject  
<span style="color:#e74c3c;">●</span> Predicate  
<span style="color:#3b82f6;">●</span> Connector  
[ ] Relative Clause  
/ Clause Boundary  

</div>
""", unsafe_allow_html=True)

user_text = st.text_area(
    "Paste sentences",
    height=180
)

user_text = user_text.replace("“", "").replace("”", "")
user_text = user_text.replace("‘", "").replace("’", "")

clean_text = preprocess_legal_text(user_text)

col1, col2, col3 = st.columns([2,1,2])
with col2:
    scan = st.button("Scan Text")

sentences = split_sentences(clean_text)

if sentences:
    sentence_scores = []
    core_scores = []

    for sent in sentences:
        doc = nlp(sent)
        load_info = compute_sentence_load(doc)
        sentence_scores.append(load_info["score"])

        core_score = compute_core_score(doc, load_info)

        tokens = [t.text.lower() for t in doc]

        FIRST_POSITION_SIGNALS = {
            "however", "therefore", "thus", "hence",
            "consequently", "accordingly",
            "nevertheless", "nonetheless"
        }

        if tokens and tokens[0] in FIRST_POSITION_SIGNALS:
            core_score += 2
        elif any(sig in tokens for sig in FIRST_POSITION_SIGNALS):
            core_score += 1

        core_scores.append(core_score)

    doc_load = compute_document_load(sentence_scores)
    num_sent = len(core_scores)

    # secondary 개수 결정
    if num_sent <= 5:
        secondary_n = 0
    elif num_sent <= 12:
        secondary_n = 1
    else:
        secondary_n = 2

    # core ranking
    ranked = sorted(
        enumerate(core_scores),
        key=lambda x: x[1],
        reverse=True
    )

    primary_idx = ranked[0][0] if ranked else None
    secondary_idx = [r[0] for r in ranked[1:1+secondary_n]]

    doc_badge = render_load_badge(doc_load["label"])

    st.markdown(
        f"""
    Reading Load: {doc_badge}

    Avg Sentence Load: {doc_load["avg"]}  
    Heavy Sentences: {doc_load["heavy_ratio"]}%  
    Critical Sentences: {doc_load["critical_ratio"]}%
    """,
        unsafe_allow_html=True
    )

    
    st.markdown(f'<div class="limit-note">Detected sentences: {len(sentences)}</div>', unsafe_allow_html=True)

    for i, sent in enumerate(sentences, 1):
        doc = nlp(sent)
        load_info = compute_sentence_load(doc)
        annotated = annotate_doc_with_clauses(doc)
        html = hawk_render(annotated)
        html = html.replace("v__", "v.")

        badge = render_load_badge(load_info["label"], load_info["score"])

        if primary_idx is not None and i - 1 == primary_idx:
            core_mark = "★ "
            core_class = "core-primary"
        elif (i - 1) in secondary_idx:
            core_mark = "☆ "
            core_class = "core-secondary"
        else:
            core_mark = ""
            core_class = ""

        if primary_idx is not None and i - 1 == primary_idx:
            core_mark = "★ "
            core_style = 'background:#fff7cc; box-shadow: inset 0 0 0 2px #f4c542;'
        elif (i - 1) in secondary_idx:
            core_mark = "☆ "
            core_style = 'background:#f3f4f6; box-shadow: inset 0 0 0 1px #9ca3af;'
        else:
            core_mark = ""
            core_style = ""


        st.markdown(
            f'<div class="block-label">{core_mark}Sentence {i} · Load: {badge}</div>',
            unsafe_allow_html=True
        )

        load_class = load_info["label"].lower()
        if load_class == "light":
            load_class = "light"
        elif load_class == "moderate":
            load_class = "moderate"
        else:
            load_class = "heavy"

        st.markdown(
            f'<div class="hawk-line load-{load_class}" style="{core_style}">{html}</div>',
            unsafe_allow_html=True
        )

    # ← 여기부터 수정
    st.markdown("---")
    st.markdown("### Was this helpful?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful"):
            st.success("Thanks! 🙏")
            st.balloons()    

    with col2:
        if st.button("👎 Not helpful"):
            st.warning("Got it — thanks!")

    st.markdown("")

    st.markdown(
        "Want to share more detailed feedback? (optional)"
    )

    st.link_button(
        "Leave detailed feedback",
        "https://docs.google.com/forms/d/1R3RRjz9972fiL1A0LvLcO-d8Ch8weHH3-84v8Mfgmg4/edit"


