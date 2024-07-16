
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
def main():
    st.title("Application de prévision avec Prophet")

    # Téléchargement du fichier Excel
    uploaded_file = st.file_uploader("Choisissez votre fichier Excel", type="xlsx")
    
    if uploaded_file is not None:
        # Lecture du fichier Excel
        df = pd.read_excel(uploaded_file)
        st.write("Aperçu des données :")
        st.write(df.head())

        # Saisie des paramètres
        start_date = st.date_input("Date de début d'entraînement")
        forecast_days = st.number_input("Nombre de jours à prévoir", min_value=1, value=30)

        if st.button("Lancer la prévision"):
            # Préparation des données pour Prophet
            df_prophet = df.rename(columns={'Date': 'ds', 'CAHT': 'y'})
            df_prophet = df_prophet[df_prophet['ds'] >= pd.to_datetime(start_date)]

            # Création et entraînement du modèle
            model = Prophet()
            model.fit(df_prophet)

            # Création des dates futures
            future_dates = model.make_future_dataframe(periods=forecast_days)
            
            # Prévision
            forecast = model.predict(future_dates)

            # Affichage des résultats
            st.write("Résultats de la prévision :")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Graphique
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

if __name__ == "__main__":
    main()

