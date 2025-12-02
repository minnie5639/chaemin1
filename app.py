import streamlit as st
import pandas as pd
import nltk
from spellchecker import SpellChecker
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import zipfile
import io

# ----------------------------------------
# NLTK Setup
# ----------------------------------------
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

ensure_nltk()

spell = SpellChecker()
detok = TreebankWordDetokenizer()

# ----------------------------------------
# Spell-check logic
# ----------------------------------------
def process_text(text):
    tokens = word_tokenize(text)
    corrected_tokens = []
    error_count = 0

    for token in tokens:
        if token.isalpha():
            corrected = spell.correction(token)
            if corrected.lower() != token.lower():
                error_count += 1
            corrected_tokens.append(corrected)
        else:
            corrected_tokens.append(token)

    corrected_text = detok.detokenize(corrected_tokens)
    total_words = len([t for t in tokens if t.isalpha()])

    return corrected_text, error_count, total_words

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.title("🪄 Spelling Checker (Streamlit Version)")

uploaded_files = st.file_uploader(
    "텍스트 파일(.txt)을 여러 개 업로드하세요",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"총 업로드 파일: **{len(uploaded_files)}개**")

    results = []
    corrected_files = {}

    for file in uploaded_files:
        raw_text = file.read().decode("utf-8", errors="ignore")
        corrected_text, error_cnt, total_words = process_text(raw_text)
        error_rate = (error_cnt / total_words * 100) if total_words else 0

        results.append({
            "filename": file.name,
            "total_words": total_words,
            "error_count": error_cnt,
            "error_rate(%)": round(error_rate, 2)
        })

        corrected_files[file.name] = corrected_text

    df = pd.DataFrame(results)
    st.subheader("📊 Summary")
    st.dataframe(df)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Summary CSV 다운로드",
        data=csv_data,
        file_name="summary.csv",
        mime="text/csv"
    )

    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, "w") as zf:
        for fname, content in corrected_files.items():
            zf.writestr(f"corrected_{fname}", content)

    st.download_button(
        label="📥 수정된 텍스트 ZIP 다운로드",
        data=zip_stream.getvalue(),
        file_name="corrected_texts.zip",
        mime="application/zip"
    )

else:
    st.info("여러 개의 .txt 파일을 업로드하면 자동으로 맞춤법 교정이 이루어집니다.")
