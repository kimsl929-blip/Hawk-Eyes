import streamlit as st
st.set_page_config(page_title="Hawk Eyes – Reading OS", layout="wide")

from report import report_ui
import re
from openai import OpenAI

client = OpenAI()

menu = st.sidebar.selectbox("메뉴", ["Report", "Reading"])

# -----------------------------
# 스타일
# -----------------------------
st.markdown("""
<style>
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
}

.card-title {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 6px;
}

.card-summary {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

def reading_ui():

    import spacy
    import requests
    import datetime


    from helper import classify_risk, refactor_prompt

    def explain_risk(load):
        if load > 3.5:
            return "Complex structure: multiple clauses or conditions"
        elif load > 2:
            return "Moderate complexity: may confuse AI"
        else:
            return "Simple structure"

    def get_improved_prompt(text):
        prompt = upgrade_prompt(text)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content


    def summarize_text(text, lang):
        prompt = f"""
    Rewrite the sentence into a VERY short headline.

    STRICT RULES:
    - Use ONLY the content of the sentence
    - DO NOT add new meaning
    - DO NOT interpret or classify
    - Max 6 words
    - Use simple everyday words
    - Avoid repeating similar words
    - Use ":" or "," to separate ideas

    For Korean:
    - Do NOT translate literally
    - Rewrite naturally in Korean

    Output ONLY in {lang}

    Sentence:
    {text}
    """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,  # 🔥 핵심
            messages=[
                {
                    "role": "system",
                    "content": "You rewrite sentences into short headlines. No interpretation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()


    import spacy
    
    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0


    import requests
    import datetime

    def log_to_sheet():
        url = "https://script.google.com/macros/s/AKfycbwXexmdxvp02bDClYknvWOYAlw0pPK5Pj7o6AaGkG3arX-pBjcvFenSiJF0IW6AG6NN/exec"

        data = {
            "timestamp": str(datetime.datetime.utcnow())
        }

        try:
            requests.post(url, json=data)
        except:
            pass

    # 👉 여기 추가
    is_admin = st.query_params.get("admin") == "1"

    if not is_admin:
        log_to_sheet()

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

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

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

    def extract_actions(doc):
        actions = []

        root = None
        for t in doc:
            if t.dep_ == "ROOT" and t.pos_ == "VERB":
                root = t
                break

        if not root:
            return actions

        verbs = [root] + [t for t in root.conjuncts if t.pos_ == "VERB"]

        modal = ""
        neg_flag = False

        for c in root.children:
            if c.dep_ == "aux":
                modal = c.text
            if c.dep_ == "neg":
                neg_flag = True

        for v in verbs:
            local_neg = neg_flag or any(c.dep_ == "neg" for c in v.children)

            obj = ""
            for c in v.children:
                if c.dep_ in ("dobj", "attr"):
                    obj = " ".join([t.text for t in c.subtree])
                    break
                if c.dep_ == "prep":
                    pobj = [x for x in c.children if x.dep_ == "pobj"]
                    if pobj:
                        obj = f"{c.text} " + " ".join([t.text for t in pobj[0].subtree])
                        break

            neg_text = "not " if local_neg else ""
            action = f"{modal} {neg_text}{v.lemma_} {obj}".strip()

            actions.append(action)

        return actions

    def upgrade_prompt(text):
        base = text.strip()

        improved = f"""Improve the following request to make it clear, specific, and effective:

    User request:
    "{base}"

    Rewrite it so that:
    - The goal is explicit
    - The output format is clearly defined
    - Necessary details are added
    - Ambiguity is removed

    Return ONLY the improved prompt.
    """
        return improved



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

    st.markdown("💡 Try editing the example or paste your own text")
    user_text = st.text_area(
        "Paste sentences",
        value="""The court held that the statute, although frequently criticized by commentators, remained binding precedent, and concluded that Congress had exceeded its authority when it enacted the provision.""",
        height=180
    )

    user_text = user_text.replace("“", "").replace("”", "")
    user_text = user_text.replace("‘", "").replace("’", "")

    clean_text = preprocess_legal_text(user_text)

    col1, col2, col3 = st.columns([2,1,2])
    with col2: 
        scan = st.button("Scan Text") 

    # 🔥 여기 추가 (정답 위치)
    lang = st.selectbox(
        "Summary Language",
        ["EN", "KR", "JP"]
    )

    # 🔥 여기 추가
    show_analysis = st.toggle("Show structure analysis")


    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0

    if scan: 
        if st.session_state.usage_count >= 999:
                st.warning("Free limit reached. Unlock full access for $3 →")
                st.stop()

        st.session_state.usage_count += 1

        st.markdown( 
            "💡 See sentence structure instantly and understand complex sentences with less effort." 
        ) 

        sentences = split_sentences(clean_text)

        sentences = sentences[:10]
        if sentences:
            sentence_scores = []
            core_scores = []

            docs = list(nlp.pipe(sentences, batch_size=20))
            
            analysis = []

            for doc in docs:
                for sent in doc.sents:
                    load_data = compute_sentence_load(sent)
                    load = load_data["score"]
                                    
                    analysis.append({
                        "text": sent.text,
                        "load": f"{round(load,1)}/5",
                        "risk": classify_risk(load)
                    })

            # 원문 기준 refactor (sentences를 하나로 합침)
            original_text = " ".join(sentences)
            
            refactored_sentences = []

            for a in analysis:
                if a["risk"] == "HIGH":
                    refactored_sentences.append(refactor_prompt(a["text"]))
                else:
                    refactored_sentences.append(a["text"])

            refactored = ""

            for i, a in enumerate(analysis, 1):
                reason = explain_risk(float(a["load"].split("/")[0]))

                if a["risk"] == "HIGH":
                    refactored += f"\n[FIXED {i}] ({reason})\n{refactor_prompt(a['text'])}\n"
                else:
                    refactored += f"\n[OK {i}] ({reason})\n{a['text']}\n"
            

            for doc in docs:
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
            doc = docs[i-1]
            
            load_info = compute_sentence_load(doc)
            
            # Sentence 정보 먼저
            st.markdown(
                f'<div class="card-title">★ Sentence {i} · Load: {load_info["score"]}</div>',
                unsafe_allow_html=True
            )
            

            # 🔥 그 다음 summary
            summary = summarize_text(sent, lang)

            # 🔥 여기 추가
            def simple_tag(summary):
                if any(x in summary for x in ["증가", "필요", "감소"]):
                    return "Main Claim"
                elif any(x in summary for x in ["예상", "수치", "BC", "년"]):
                    return "Support"
                elif any(x in summary for x in ["하지만", "적음", "제한"]):
                    return "Counterargument"
                else:
                    return "Support"

            tag = simple_tag(summary)

            st.markdown(f"[{tag}] {summary}")

            if show_analysis:
                annotated = annotate_doc_with_clauses(doc)
                html = hawk_render(annotated)
                        
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

            st.markdown("---")

        st.subheader("Improved Prompt")

        improved = get_improved_prompt(original_text)

        st.code(improved)

        if st.button("리포트로 보내기"):
            st.session_state["reading_result"] = improved
            st.session_state["auto_mode"] = True
            

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
        )


# -----------------------------
# 분기 (🔥 핵심)
# -----------------------------
if menu == "Reading":
    reading_ui()

elif menu == "Report":
    report_ui()





