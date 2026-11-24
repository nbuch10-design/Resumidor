import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Text Summarization", page_icon="📝")
st.title("📝 Text Summarization")


@st.cache_resource  # carrega el model una sola vegada per a tots els usuaris
def load_model():
    return pipeline("summarization", model="t5-small", device=-1)  # -1 = CPU


summarizer = load_model()

text = st.text_area("Enganxa el text que vols resumir:", height=250)
col1, col2 = st.columns(2)
min_len = col1.slider("Longitud mínima", 10, 100, 30)
max_len = col2.slider("Longitud màxima", 50, 300, 130)

if st.button("Resumir"):
    if not text.strip():
        st.warning("Escriu o enganxa algun text primer.")
    else:
        with st.spinner("Generant resum..."):
            result = summarizer(text, min_length=min_len, max_length=max_len,
                                do_sample=False, truncation=True)
        st.subheader("Resum")
        st.write(result[0]["summary_text"])