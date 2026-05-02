import re

def classify_risk(load):
    if load < 1.5:
        return "LOW"
    elif load < 3:
        return "MEDIUM"
    else:
        return "HIGH"


def split_long_sentence(text):
    if len(text) > 120:
        return text.replace(" and ", ". ")
    return text


def simplify_conditions(text):
    text = re.sub(r"\bif\b", "STEP 1:", text)
    text = re.sub(r"\bthen\b", "STEP 2:", text)
    return text


def remove_extra_clauses(text):
    return re.sub(r"\(.*?\)", "", text)


def refactor_prompt(text: str):
    text = text.strip()

    # 1. 조건문 분리 (완전한 구조로)
    if "if" in text.lower():
        parts = text.split(",", 1)
        if len(parts) == 2:
            step1 = parts[0].strip()
            step2 = parts[1].strip()
            return f"STEP 1: {step1}.\nSTEP 2: {step2}"

    # 2. and 분리 (라인 단위)
    if " and " in text:
        parts = text.split(" and ")
        return "\n".join([f"- {p.strip()}" for p in parts])

    return text