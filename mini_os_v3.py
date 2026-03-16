# mini_os_v1.py

import spacy

# ----------------------------
# 0. Load
# ----------------------------
nlp = spacy.load("en_core_web_sm")


# ----------------------------
# 1. Test sentences
# ----------------------------
TEST_SENTENCES = [
    "Along with Egypt and Sumer, the third major early Bronze Age civilization was the Harappan civilization of the Indus Valley.",
    "Although the evidence is incomplete, the theory remains influential among legal scholars.",
    "Many of the students who had prepared carefully were still confused by the professor's final question.",
    "In recent years, courts have become more skeptical of arguments based solely on legislative history.",
    "Because the statute was poorly drafted, its meaning has remained disputed."
    # 6 fronted PP
    "After the committee reviewed the evidence, the proposal was rejected.",

    # 7 fronted PP
    "Under the new regulations, companies must disclose additional information.",

    # 8 fronted PP
    "In many countries, the policy remains controversial.",

    # 9 fronted adverb clause
    "Although the data appear reliable, the conclusion remains controversial.",

    # 10 fronted adverb clause
    "While the argument seems persuasive, its assumptions are questionable.",

    # 11 quantifier subject
    "Many of the researchers involved in the study were surprised by the result.",

    # 12 quantifier subject
    "Most of the participants in the experiment were unaware of the change.",

    # 13 be + adjective
    "The explanation is difficult to accept.",

    # 14 become + adjective
    "The issue has become increasingly controversial.",

    # 15 remain + adjective
    "The theory has remained influential among historians.",

    # 16 passive predicate
    "The decision was strongly criticized by legal scholars.",

    # 17 passive predicate
    "The proposal was rejected by the committee.",

    # 18 longer predicate
    "The results have become widely accepted among economists.",

    # 19 simple clause
    "The court rejected the argument.",

    # 20 simple clause
    "The policy has changed significantly in recent years.",

    # 21 advcl
    "Although the data appear reliable, the conclusion remains controversial.",

    # 22 advcl
    "Because the statute was poorly drafted, its meaning has remained disputed.",

    # 23 ccomp
    "The committee concluded that the policy was flawed.",

    # 24 ccomp
    "Many scholars argue that the theory remains influential.",

    # 25 explicit relative
    "The students who had prepared carefully were confused.",

    # 26 reduced relative (VBG)
    "The students preparing for the exam were nervous.",

    # 27 reduced relative (VBN)
    "The issues discussed at the meeting remain unresolved.",

    # 28 reduced relative (ADJ)
    "The people responsible for the decision were criticized.",

    # 29 
    "The book that the professor recommended was difficult to read.",
]


# ----------------------------
# 2. Core detectors
# ----------------------------

def find_root_verb(doc):
    """
    기본은 spaCy ROOT를 반환.
    다만 ROOT가 dash(—, –, --) 뒤에 있으면,
    dash 앞 구간에서 '주절 후보'를 다시 찾는다.
    """

    # 1) spaCy ROOT 찾기
    root = None
    for tok in doc:
        if tok.dep_ == "ROOT":
            root = tok
            break

    if root is None:
        return None

    # 2) dash 위치 찾기
    dash_idx = None
    for tok in doc:
        if tok.text in {"—", "–", "--"}:
            dash_idx = tok.i
            break

    # dash 없으면 원래 ROOT 사용
    if dash_idx is None:
        return root

    # ROOT가 dash 앞이면 원래 ROOT 사용
    if root.i < dash_idx:
        return root

    # 3) ROOT가 dash 뒤에 있으면, dash 앞에서 주절 후보 재탐색
    left_tokens = [tok for tok in doc if tok.i < dash_idx]

    # 우선순위 1:
    # be/cop/aux가 붙은 보어(acomp/attr/oprd) 또는 동사
    candidates = []
    for tok in left_tokens:
        has_aux_or_cop = any(c.dep_ in {"aux", "auxpass", "cop"} for c in tok.children)

        if tok.dep_ in {"acomp", "attr", "oprd"} and has_aux_or_cop:
            candidates.append(tok)
        elif tok.pos_ in {"VERB", "AUX"} and has_aux_or_cop:
            candidates.append(tok)

    if candidates:
        return candidates[-1]  # dash 앞에서 가장 오른쪽 후보

    # 우선순위 2:
    # dash 앞 finite verb / aux
    finite_candidates = []
    for tok in left_tokens:
        if tok.pos_ in {"VERB", "AUX"}:
            verb_form = tok.morph.get("VerbForm")
            if "Fin" in verb_form or tok.tag_ in {"VBD", "VBP", "VBZ", "MD"}:
                finite_candidates.append(tok)

    if finite_candidates:
        return finite_candidates[-1]

    # 4) comma 이후 주절 finite verb 재탐색
    comma_positions = [tok.i for tok in doc if tok.text == ","]

    if len(comma_positions) >= 1:
        last_comma = comma_positions[-1]

        right_finite = []
        for tok in doc:
            if tok.i <= last_comma:
                continue
            if tok.pos_ not in {"VERB", "AUX"}:
                continue

            verb_form = tok.morph.get("VerbForm")
            if "Fin" in verb_form or tok.tag_ in {"VBD", "VBP", "VBZ", "MD"}:
                right_finite.append(tok)

        if right_finite:
            return right_finite[0]

    # fallback
    return root


def find_subject_head(doc, root):
    """
    주어 head:
    - root의 children 중 nsubj / nsubjpass / csubj
    - 없으면 None

    mini v1에서는 Many of the students 같은 경우
    일단 spaCy 문법 head를 그대로 유지한다.
    """
    if root is None:
        return None

    for child in root.children:
        if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
            return child
    return None


def find_predicate_span(doc, root):
    """
    mini v1.1 predicate span 규칙

    1) root가 VERB/AUX이면
       - 왼쪽 auxiliary/neg + root
       예: have become, has remained, did not go

    2) root가 ADJ/NOUN 계열이고 cop이 있으면
       - cop lemma가 be 이면: cop + root
         예: were confused, is clear
       - cop lemma가 remain/become/seem/appear 등 이면: aux + cop
         예: has remained disputed -> has remained
             have become skeptical -> have become
    """
    if root is None:
        return []
    
    # case 1: root 자체가 동사/조동사
    if root.pos_ in {"VERB", "AUX"}:
        aux_like = [
            c for c in root.children
            if c.dep_ in {"aux", "auxpass", "neg"}
        ]

        # perfect tense 처리: have/has/had + ... + VBN
        perfect_aux = [
            c for c in root.children
            if c.dep_ in {"aux"} and c.lemma_ == "have"
        ]

        if perfect_aux and root.tag_ == "VBN":
            span_tokens = perfect_aux

            for c in root.children:
                if c.lemma_ == "be" and c.dep_ in {"aux", "auxpass"}:
                    span_tokens.append(c)

            span_tokens.append(root)

            return sorted(span_tokens, key=lambda x: x.i)

        verb_form = root.morph.get("VerbForm")

        # Lite: finite verb가 아니더라도 aux/modal이 붙은 bare infinitive는 허용
        if root.pos_ == "VERB" and "Fin" not in verb_form:
            if not aux_like:
                return []

        span_tokens = aux_like + [root]
        return sorted(span_tokens, key=lambda x: x.i)
        
    # case 2: root가 보어(ADJ/NOUN 등)이고 cop이 붙는 경우
    cop_children = [
        c for c in root.children
        if c.dep_ == "cop" and c.pos_ in {"AUX", "VERB"}
    ]

    if cop_children:
        cop = sorted(cop_children, key=lambda x: x.i)[-1]

        aux_like = []

        # aux가 root에 직접 붙는 경우
        aux_like.extend(
            c for c in root.children
            if c.dep_ in {"aux", "auxpass", "neg"}
        )

        # aux가 cop에 붙는 경우
        aux_like.extend(
            c for c in cop.children
            if c.dep_ in {"aux", "auxpass", "neg"}
        )

        # 중복 제거
        aux_like = list({t.i: t for t in aux_like}.values())
        aux_like = sorted(aux_like, key=lambda x: x.i)

        # be + 보어  => were confused / is clear
        if cop.lemma_.lower() == "be":
            span_tokens = aux_like + [cop, root]
            span_tokens = sorted({t.i: t for t in span_tokens}.values(), key=lambda x: x.i)
            return span_tokens

        # remain/become/seem/appear + 보어 => has remained / have become
        else:
            span_tokens = aux_like + [cop]
            span_tokens = sorted({t.i: t for t in span_tokens}.values(), key=lambda x: x.i)
            return span_tokens

    # fallback
    return [root]

def compute_core_score(doc, load_info=None):
    score = 0

    # 1) sentence load 반영
    if load_info is not None:
        score += load_info.get("score", 0)

    # 2) reasoning signal (강하게)
    reasoning_signals = {"therefore", "thus", "because", "since", "accordingly"}
    if any(tok.text.lower() in reasoning_signals for tok in doc):
        score += 3

    # 3) legal verb (강하게)
    legal_verbs = {"hold", "conclude", "determine", "find", "recognize"}
    if any(tok.lemma_.lower() in legal_verbs for tok in doc):
        score += 3

    # 4) that-clause (약하게)
    if any(tok.text.lower() == "that" for tok in doc):
        score += 1

    return score



def find_initial_boundary(doc, root):
    """
    mini v1 boundary:
    - root 앞의 첫 쉼표 뒤에 / 삽입
    """
    if root is None:
        return None

    for tok in doc:
        if tok.i < root.i and tok.text == ",":
            return tok.i
    return None


def get_clause_depth(doc):
    """
    Lite v1용 단순 clause depth
    - main clause만 있으면 0
    - subordinate / relative 중 하나 있으면 1
    - 둘 다 있거나 subordinate가 2개 이상이면 2
    """
    main_root = find_root_verb(doc)
    sub_roots = find_clause_roots(doc, main_root)
    rel_spans = find_relative_spans(doc)

    if not sub_roots and not rel_spans:
        return 0

    if (len(sub_roots) >= 2) or (sub_roots and rel_spans):
        return 2

    return 1


def has_subordinate_clause(doc):
    main_root = find_root_verb(doc)
    sub_roots = find_clause_roots(doc, main_root)
    return len(sub_roots) > 0


def has_relative_clause(doc):
    rel_spans = find_relative_spans(doc)
    return len(rel_spans) > 0


def has_predicate_chain(doc):
    """
    주절 외 predicate가 1개 이상 있으면 True
    subordinate clause 또는 additional predicate 흔적이 있으면 +1용
    """
    main_root = find_root_verb(doc)
    sub_roots = find_clause_roots(doc, main_root)

    if sub_roots:
        return True

    # main root와 연결된 conj verb도 predicate chain으로 볼 수 있음
    for tok in doc:
        if tok.dep_ == "conj" and tok.pos_ in {"VERB", "AUX"}:
            return True

    return False


def has_initial_delay(doc):
    """
    문두 modifier / clause 때문에 slash가 필요한 경우
    """
    main_root = find_root_verb(doc)
    boundary_idx = find_initial_boundary(doc, main_root)
    return boundary_idx is not None


def compute_sentence_load(doc):
    """
    Sentence Load v1
    score:
      depth_score (0~2)
      + subordinate clause
      + relative clause
      + predicate chain
      + initial delay

    final score capped at 5
    """
    depth = get_clause_depth(doc)

    score = 0

    # clause depth
    if depth == 0:
        score += 0
    elif depth == 1:
        score += 1
    else:
        score += 2

    # bonuses
    if has_subordinate_clause(doc):
        score += 1

    if has_relative_clause(doc):
        score += 1

    if has_predicate_chain(doc):
        score += 1

    if has_initial_delay(doc):
        score += 1

    score = min(score, 5)

    token_len = len(doc)

    if token_len <= 8:
        score = max(0, score - 4)
    elif token_len <= 12:
        score = max(0, score - 3)

    if score <= 1:
        label = "Light"
    elif score <= 3:
        label = "Moderate"
    else:
        label = "Heavy"

    return {
        "score": score,
        "label": label,
        "depth": depth,
        "has_subordinate": has_subordinate_clause(doc),
        "has_relative": has_relative_clause(doc),
        "has_predicate_chain": has_predicate_chain(doc),
        "has_initial_delay": has_initial_delay(doc),
    }

def compute_document_load(scores):
    if not scores:
        return None

    avg = sum(scores) / len(scores)

    heavy = sum(1 for s in scores if s >= 3)
    critical = sum(1 for s in scores if s >= 4)

    heavy_ratio = heavy / len(scores)
    critical_ratio = critical / len(scores)

    if avg < 2.0 and heavy_ratio < 0.25:
        label = "Light"
    elif avg < 3.0 and critical_ratio < 0.15:
        label = "Moderate"
    else:
        label = "Heavy"

    return {
        "label": label,
        "avg": round(avg, 2),
        "heavy_ratio": round(heavy_ratio * 100),
        "critical_ratio": round(critical_ratio * 100),
    }


def annotate_doc(doc):
    """
    표시 규칙
    - 주어 head → «word»
    - predicate span → [[...]]
    - 문장 경계 → /
    """
    
    root = find_root_verb(doc)
    subj = find_subject_head(doc, root)
    
    import streamlit as st
    st.write("ROOT:", root.text if root else None)

    # main predicates (root + promoted conj verbs)
    main_preds = []
    if root is not None:
        main_preds.append(root)

        for tok in doc:
            if tok.dep_ != "conj" or tok.pos_ not in {"VERB", "AUX"}:
                continue

            # case 1: direct main-level conj
            if tok.head == root:
                main_preds.append(tok)
                continue

            # case 2: conj verb with its own ccomp
            has_own_ccomp = any(child.dep_ == "ccomp" for child in tok.children)
            if has_own_ccomp:
                main_preds.append(tok)

   
    pred_tokens = []
    for p in main_preds:
        pred_tokens.extend(find_predicate_span(doc, p))

    boundary_idx = find_initial_boundary(doc, root)

    pred_id_set = {t.i for t in pred_tokens}

    out = []

    for tok in doc:
        text = tok.text

        # subject head
        if subj is not None and tok.i == subj.i:
            text = f"«{text}»"

        prev_in_pred = (tok.i - 1) in pred_id_set
        curr_in_pred = tok.i in pred_id_set
        next_in_pred = (tok.i + 1) in pred_id_set

        # predicate span 시작
        if curr_in_pred and not prev_in_pred:
            text = "[[" + text

        # predicate span 끝
        if curr_in_pred and not next_in_pred:
            text = text + "]]"

        out.append(text)

        # boundary
        if boundary_idx is not None and tok.i == boundary_idx:
            out.append("/")    

    res = " ".join(out)

    # punctuation spacing cleanup
    res = res.replace(" ,", ",")
    res = res.replace(" .", ".")
    res = res.replace(" ?", "?")
    res = res.replace(" !", "!")
    res = res.replace(" ;", ";")
    res = res.replace(" :", ":")

    # slash spacing
    res = res.replace(", /", ", /")
    res = res.replace(" /", " / ")

    # collapse double spaces
    while "  " in res:
        res = res.replace("  ", " ")

    return res.strip()

def find_clause_roots(doc, main_root):
    """
    subordinate clause root 후보
    - advcl
    - ccomp attached to any main predicate
    - conj only when it is inside a subordinate clause
    """
    clause_roots = []

    main_preds = set()
    if main_root is not None:
        main_preds.add(main_root)
        for tok in doc:
            if tok.dep_ != "conj" or tok.pos_ not in {"VERB", "AUX"}:
                continue

            if tok.head == main_root:
                main_preds.add(tok)
                continue

            has_own_ccomp = any(child.dep_ == "ccomp" for child in tok.children)
            if has_own_ccomp:
                main_preds.add(tok)
    
    
    for tok in doc:
        if tok == main_root:
            continue

        if tok.dep_ == "advcl":
            clause_roots.append(tok)

        elif tok.dep_ == "ccomp" and tok.head in main_preds:
            clause_roots.append(tok)

        elif tok.dep_ == "conj":
            verb_form = tok.morph.get("VerbForm")
            if (
                tok.pos_ in {"VERB", "AUX"}
                and ("Fin" in verb_form or tok.pos_ == "AUX")
                and tok.head.dep_ in {"ccomp", "advcl"}
            ):
                clause_roots.append(tok)

    return sorted(clause_roots, key=lambda x: x.i)


def find_subject_head_from_root(root):
    """
    주어진 절 root의 subject head 찾기
    """
    if root is None:
        return None

    for child in root.children:
        if child.dep_ in {"nsubj", "nsubjpass", "csubj"}:
            return child

    return None


def find_relative_spans(doc):
    """
    Relative clause spans

    포함
    1) explicit relative clause (relcl)
    2) reduced relative
       - acl + VBG/VBN/ADJ
       - noun-following ADJ phrase
    """

    spans = []

    for tok in doc:

        # 1 explicit relative clause
        if tok.dep_ == "relcl":
            left = min(t.i for t in tok.subtree)
            right = max(t.i for t in tok.subtree)
            spans.append((left, right))
            continue

        # 2 reduced relative: acl 기반
        if tok.dep_ == "acl" and (tok.pos_ in {"VERB", "ADJ"} or tok.tag_ in {"VBG", "VBN", "JJ"}):
            # 명사 뒤에서 명사를 수식하는 acl만 허용
            if tok.head.pos_ not in {"NOUN", "PROPN", "PRON"}:
                continue
            if tok.i <= tok.head.i:
                continue

            verb_form = tok.morph.get("VerbForm")

            # to-infinitive acl 제외
            if "Inf" in verb_form:
                continue

            left = min(t.i for t in tok.subtree)
            right = max(t.i for t in tok.subtree)

            # 부사절류는 제외
            if doc[left].text.lower() in {"when", "while", "if", "because", "although", "though"}:
                continue
            
            # comma insertion 제외
            if left > 0 and right < len(doc) - 1:
                if doc[left - 1].text == "," and doc[right + 1].text == ",":
                    continue    

            spans.append((left, right))
            continue

        # 3 noun-following adjective phrase
        # 예: people responsible for the decision
        if tok.pos_ == "ADJ" and tok.dep_ in {"amod", "acl"}:
            if tok.head.pos_ in {"NOUN", "PROPN", "PRON"} and tok.i == tok.head.i + 1:
                left = min(t.i for t in tok.subtree)
                right = max(t.i for t in tok.subtree)

                # comma insertion 제외
                if left > 0 and right < len(doc) - 1:
                    if doc[left - 1].text == "," and doc[right + 1].text == ",":
                        continue

                spans.append((left, right))
                continue




    # 중복 제거
    spans = sorted(set(spans))
    return spans


def find_that_clause_starts(doc):
    """
    ccomp that-clause 시작점 찾기
    that이 marker로 붙어 있으면 that 앞에 / 삽입
    """
    starts = set()

    for tok in doc:
        if tok.dep_ == "ccomp":
            for child in tok.children:
                if child.dep_ == "mark" and child.text.lower() == "that":
                    starts.add(child.i)

    return starts


def annotate_doc_with_clauses(doc):
    """
    표시 규칙
    - main subject      → «word»
    - main predicate    → [[...]]
    - subordinate subj  → ‹word›
    - subordinate pred  → {{...}}
    - boundary          → /
    """

    main_root = find_root_verb(doc)
    main_subj = find_subject_head(doc, main_root)

    main_preds = []
    if main_root is not None:
        main_preds.append(main_root)
        for tok in doc:
            if tok.dep_ == "conj" and tok.pos_ in {"VERB", "AUX"} and tok.head == main_root:
                main_preds.append(tok)

    main_pred_root_ids = {p.i for p in main_preds}

    main_pred_tokens = []
    for p in main_preds:
        main_pred_tokens.extend(find_predicate_span(doc, p))

    boundary_idx = find_initial_boundary(doc, main_root)

    clause_roots = find_clause_roots(doc, main_root)

    # main IDs
    main_pred_id_set = {t.i for t in main_pred_tokens} if main_pred_tokens else set()
    main_subj_id = main_subj.i if main_subj is not None else None

    # subordinate maps
    sub_subj_ids = set()
    sub_pred_start_ids = set()
    sub_pred_end_ids = set()
    sub_pred_id_set = set()

    for root in clause_roots:
        if root.i in main_pred_root_ids:
            continue

        sub_subj = find_subject_head_from_root(root)
        sub_pred_tokens = find_predicate_span(doc, root)

        if sub_subj is not None:
            sub_subj_ids.add(sub_subj.i)

        if sub_pred_tokens:
            ids = [t.i for t in sub_pred_tokens]
            sub_pred_start_ids.add(min(ids))
            sub_pred_end_ids.add(max(ids))
            for t in sub_pred_tokens:
                sub_pred_id_set.add(t.i)

    rel_spans = find_relative_spans(doc)
    that_clause_starts = find_that_clause_starts(doc)

    out = []

    rel_start = {s for s, e in rel_spans}
    rel_end = {e for s, e in rel_spans}

    for tok in doc:
        text = tok.text

    in_quote = False

    for tok in doc:
        text = tok.text

        APPOSITIVE_WHETHER_NOUNS = {"question","issue","problem","doubt","uncertainty"}

        prev_tok = doc[tok.i - 1] if tok.i > 0 else None

        if tok.text.lower() == "whether" and prev_tok is not None:
            if prev_tok.lemma_.lower() in APPOSITIVE_WHETHER_NOUNS:
                text = "/ " + text
    
        if tok.text in {'"', '“', '”'}:
            in_quote = not in_quote

        # 인용부 안의 verb는 predicate/subject 처리하지 않음
        if in_quote:
            out.append(text)
            continue    

        if tok.dep_ == "mark":
            text = f'<span class="connector">{text}</span>'

        # relative clause start
        if tok.i in rel_start:    
            text = "[" + text

        # relative clause end
        if tok.i in rel_end:
            text = text + "]"

        # subordinate subject
        if tok.i in sub_subj_ids:
            text = f"‹{text}›"

        # main subject
        if main_subj_id is not None and tok.i == main_subj_id:
            text = f"«{text}»"

        # subordinate predicate start
        if tok.i in sub_pred_start_ids:
            text = "{{" + text

        prev_in_main_pred = ((tok.i - 1) in main_pred_id_set) and ((tok.i - 1) not in sub_pred_id_set)
        curr_in_main_pred = (tok.i in main_pred_id_set) and (tok.i not in sub_pred_id_set)
        next_in_main_pred = ((tok.i + 1) in main_pred_id_set) and ((tok.i + 1) not in sub_pred_id_set)

        # main predicate start
        if curr_in_main_pred and not prev_in_main_pred:
            text = "[[" + text

        # subordinate predicate end
        if tok.i in sub_pred_end_ids:
            text = text + "}}"

        # main predicate end
        if curr_in_main_pred and not next_in_main_pred:
            text = text + "]]"    

        APPOSITIVE_THAT_NOUNS = {
            "argument",
            "fact",
            "claim",
            "evidence",
            "idea",
            "belief",
            "assumption",
            "conclusion",
            "finding",
            "notion",
            "view",
            "position",
            "contention",
            "premise",
            "principle",
            "doctrine",
            "theory",
            "possibility",
            "probability",
            "hypothesis",
            "observation",
            "understanding"
        }

        prev_tok = doc[tok.i - 1] if tok.i > 0 else None
        if tok.text.lower() == "that" and prev_tok is not None:
            if prev_tok.pos_ == "NOUN" and prev_tok.lemma_.lower() in APPOSITIVE_THAT_NOUNS:
                if not out or out[-1] != "/":
                    out.append("/")
              
        # that-clause boundary
        if tok.i in that_clause_starts:
            out.append("/")

        out.append(text)

        # boundary
        if boundary_idx is not None and tok.i == boundary_idx:
            suppress_boundary = False

            # 쉼표 뒤에 삽입형 부사절이 오는 경우: / 금지
            if tok.text == "," and tok.i + 1 < len(doc):
                nxt = doc[tok.i + 1].text.lower()
                nxt2 = doc[tok.i + 2].text.lower() if tok.i + 2 < len(doc) else ""

                if nxt in {"although", "while", "if", "because", "though", "when"}:
                    suppress_boundary = True

                # even though / even when / even if
                if nxt == "even" and nxt2 in {"though", "when", "if"}:
                    suppress_boundary = True

            if not suppress_boundary:
                out.append("/")
        
    res = " ".join(out)

    # punctuation spacing cleanup
    res = res.replace(" ,", ",")
    res = res.replace(" .", ".")
    res = res.replace(" ?", "?")
    res = res.replace(" !", "!")
    res = res.replace(" ;", ";")
    res = res.replace(" :", ":")

    # slash spacing
    res = res.replace(" /", " / ")

    # collapse double spaces
    while "  " in res:
        res = res.replace("  ", " ")

    return res.strip()

def quick_check(doc):
    root = find_root_verb(doc)
    subj = find_subject_head(doc, root)
    pred_tokens = find_predicate_span(doc, root)
    boundary_idx = find_initial_boundary(doc, root)

    pred_text = " ".join(tok.text for tok in pred_tokens) if pred_tokens else None
    boundary_tok = doc[boundary_idx].text if boundary_idx is not None else None

    print("ROOT     :", root.text if root else None)
    print("SUBJ HEAD:", subj.text if subj else None)
    print("PRED SPAN:", pred_text)
    print("BOUNDARY :", boundary_idx, boundary_tok)

def quick_check_clauses(doc):
    main_root = find_root_verb(doc)
    clause_roots = find_clause_roots(doc, main_root)

    print("MAIN ROOT :", main_root.text if main_root else None)

    if not clause_roots:
        print("SUB CLAUSE ROOTS: None")
        return

    print("SUB CLAUSE ROOTS:")
    for root in clause_roots:
        subj = find_subject_head_from_root(root)
        pred = find_predicate_span(doc, root)
        pred_text = " ".join(tok.text for tok in pred) if pred else None

        print(
            f"  - dep={root.dep_:<6} root={root.text:<12} "
            f"subj={subj.text if subj else None:<12} pred={pred_text}"
        )


def show_tokens(doc):
    print("=" * 80)
    print(doc.text)
    print("-" * 80)
    print(f"{'i':<3} {'text':<15} {'dep':<12} {'pos':<8} {'head'}")
    for tok in doc:
        print(f"{tok.i:<3} {tok.text:<15} {tok.dep_:<12} {tok.pos_:<8} {tok.head.text}")


def main():
    for i, sent in enumerate(TEST_SENTENCES, 1):
        doc = nlp(sent)
        for tok in doc:
            if tok.text.lower() == "make":
                print(
                    "TEXT:", tok.text,
                    "| DEP:", tok.dep_,
                    "| POS:", tok.pos_,
                    "| MORPH:", tok.morph,
                    "| HEAD:", tok.head.text,
                    "| HEAD_DEP:", tok.head.dep_
                )

        print(f"\n[Sentence {i}]")
        show_tokens(doc)
        quick_check(doc)
        quick_check_clauses(doc)
        print("ANNOTATED:")
        print(annotate_doc_with_clauses(doc))

if __name__ == "__main__":
    main()

