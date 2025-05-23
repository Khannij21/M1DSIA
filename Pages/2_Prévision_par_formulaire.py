import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("📝 Prédiction du diabète via un formulaire")

# --- Chargement du modèle et normaliseur ---
@st.cache_data
def load_model():
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

model, scaler, accuracy = load_model()
st.write(f"Précision du modèle sur les données de test : **{accuracy:.2f}**")

# --- Formulaire de saisie ---
with st.form("formulaire_prediction"):
    st.subheader("Remplissez les informations suivantes :")
    pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glycémie (mg/dL)", min_value=0)
    blood_pressure = st.number_input("Tension artérielle (mmHg)", min_value=0)
    skin_thickness = st.number_input("Épaisseur du pli cutané (mm)", min_value=0)
    insulin = st.number_input("Insuline (µU/mL)", min_value=0)
    weight = st.number_input("Poids (kg)", min_value=20.0)
    height = st.number_input("Taille (m)", min_value=1.0)
    age = st.number_input("Âge", min_value=0, max_value=120, step=1)
    family_history = st.radio("Antécédents familiaux de diabète ?", ("Oui", "Non"))

    submitted = st.form_submit_button("Prédire")

# --- Prédiction ---
if submitted:
    bmi = weight / (height ** 2)
    pedigree = 0.7 if family_history == "Oui" else 0.2

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("Résultat de la prédiction")
    st.write(f"🧬 Probabilité d'être diabétique : **{proba:.2%}**")

    if prediction == 1:
        st.error("Le modèle prédit que la personne est **diabétique**.")
    else:
        st.success("Le modèle prédit que la personne n'est **pas diabétique**.")
