import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --- Extraction depuis le PDF ---
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# --- Extraction des variables depuis le texte ---
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

# --- Entraînement du modèle SVM ---
@st.cache_data

def train_model():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = SVC(probability=True)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, acc

# --- Application Streamlit ---
st.title("Prédiction du diabète à partir d'une analyse médicale")

model, scaler, accuracy = train_model()
st.write(f"Précision du modèle SVM sur les données de test : **{accuracy:.2f}**")

uploaded_file = st.file_uploader("Uploader un document PDF médical contenant les valeurs...", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    

    extracted = extract_variables(text)
    

    st.subheader("Complétez les informations suivantes")
    pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, step=1)
    poids = st.number_input("Poids (kg)", min_value=30.0, max_value=200.0, step=0.1)
    taille = st.number_input("Taille (m)", min_value=1.0, max_value=2.5, step=0.01)
    antecedents = st.radio("Avez-vous des antécédents familiaux de diabète ?", ("Oui", "Non"))

    if all(v is not None for v in extracted.values()) and poids and taille:
        bmi = poids / (taille ** 2)
        pedigree = 0.7 if antecedents == "Oui" else 0.2

        input_data = np.array([[pregnancies, extracted['Glucose'], extracted['BloodPressure'], 
                                extracted['SkinThickness'], extracted['Insulin'], bmi, 
                                pedigree, extracted['Age']]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0][1]

        st.subheader("Résultat de la prédiction")
        st.write(f"🧬 Probabilité d'être diabétique : **{proba:.2%}**")
        if prediction == 1:
            st.error("Le modèle prédit que la personne est **diabétique**.")
        else:
            st.success("Le modèle prédit que la personne n'est **pas diabétique**.")
    else:
        st.warning("Assurez-vous que toutes les valeurs soient bien extraites et complétées.")
