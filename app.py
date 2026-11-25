import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
import streamlit as st

# dispositiu: cuda o cpu
device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Generador de resumenes en inglés", page_icon="📝")
st.title("Resumidor")
st.subheader("📝 Programa generado con ZyroGod/flan-t5-base-summarization")

# Load tokenizer and full model
@st.cache_resource  # carrega el model una sola vegada per a tots els usuaris
def load_model():
    model_name = "ZyroGod/flan-t5-base-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    return model, tokenizer

def resumator(texto):
    input_text = "summarize: " + texto
    # creem el tokenitzador
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    # generem la sortida tokenitzada
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=150,
            min_length=60,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3
        )
    # decodifiquem la sortida per tal de què sigui llegible
    resumen = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return resumen


model, tokenizer = load_model()

text1 = st.text_area("Texto para resumir:", height=250)

boton1 = st.button("Resumir")
if boton1:
    if not text1.strip():
        st.warning("Escribe un texto para resumir")
    else:
        with st.spinner("Generando resumen..."):
            resultado = resumator(text1)

        st.subheader("Resumen")
        st.write(resultado)


