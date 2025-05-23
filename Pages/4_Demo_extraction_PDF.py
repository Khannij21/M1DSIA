import streamlit as st
import fitz  # PyMuPDF
import re

st.title("Démonstration : Extraction des données depuis un PDF médical")

# --- Fonction d'extraction du texte ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# --- Fonction d'extraction des variables ---
def extract_variables(text):
    vars = {}
    patterns = {
        'Age': r'\b(?:Âge|Age)\s*[:\-]?\s*(\d+)',
        'Glucose': r'\b(?:Glycémie.*?|Glucose)\s*[:\-]?\s*(\d+)',
        'BloodPressure': r'\b(?:Tension artérielle|BloodPressure)\s*[:\-]?\s*(\d+)',
        'Insulin': r'\bInsuline\s*[:\-]?\s*(\d+)',
        'SkinThickness': r'\b(?:Épaisseur du pli cutané|SkinThickness)\s*[:\-]?\s*(\d+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        vars[key] = int(match.group(1)) if match else None
    return vars

# --- Upload PDF ---
uploaded_file = st.file_uploader("Téléversez un fichier PDF pour voir l'extraction automatique", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    
    st.subheader("Texte extrait du PDF")
    st.text(text)

    extracted = extract_variables(text)
    
    st.subheader("Variables extraites ")
    st.json(extracted)
