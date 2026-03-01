import streamlit as st
import pandas as pd
import pickle
import requests

# -----------------------------
# 1. URLs desde HuggingFace
# -----------------------------
DATA_URL = "https://huggingface.co/datasets/Jacke23/steam-recommender-assets/resolve/main/steam_games_preprocessed.csv"
VECTORIZER_URL = "https://huggingface.co/datasets/Jacke23/steam-recommender-assets/resolve/main/tfidf_vectorizer.pkl"
SIM_MATRIX_URL = "https://huggingface.co/datasets/Jacke23/steam-recommender-assets/resolve/main/similarity_hybrid.pkl"

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(
    page_title="Steam Game Recommender",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Steam Game Recommender System")
st.write("Recomienda juegos similares basados en descripciones, tags y géneros usando un modelo híbrido de NLP + Machine Learning.")


# -----------------------------
# 2. Descarga de archivos
# -----------------------------
def download_file(url, filename):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)


@st.cache_resource
def load_data():
    download_file(DATA_URL, "steam_games_preprocessed.csv")
    data = pd.read_csv("steam_games_preprocessed.csv")
    data["description_lemm"] = data["description_lemm"].fillna(
        "unknown").astype(str)
    return data


@st.cache_resource
def load_vectorizer():
    download_file(VECTORIZER_URL, "tfidf_vectorizer.pkl")
    return pickle.load(open("tfidf_vectorizer.pkl", "rb"))


@st.cache_resource
def load_similarity():
    download_file(SIM_MATRIX_URL, "similarity_hybrid.pkl")
    return pickle.load(open("similarity_hybrid.pkl", "rb"))


# -----------------------------
# 3. Cargar los recursos
# -----------------------------
data = load_data()
tfidf_vectorizer = load_vectorizer()
sim_hybrid_matrix = load_similarity()


# -----------------------------
# 4. Función de recomendación
# -----------------------------
def recommend_hybrid(game_name, data, sim_matrix, top_n=5):

    if game_name not in data['name'].values:
        raise ValueError(
            f"El juego '{game_name}' no existe en la base de datos.")

    game_index = data[data['name'] == game_name].index[0]

    sim_scores = list(enumerate(sim_matrix[game_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    return data.iloc[[index for index, _ in sim_scores]][['appid', 'name', 'genres', 'steamspy_tags']]


# -----------------------------
# 5. Sidebar
# -----------------------------
st.sidebar.header("Selecciona tus preferencias")

game_list = data['name'].sort_values().unique()
game_name = st.sidebar.selectbox(
    "Elige un juego para recibir recomendaciones", game_list)

num_results = st.sidebar.slider("Número de recomendaciones", 5, 20, 5)


# -----------------------------
# 6. Output
# -----------------------------
if st.sidebar.button("Generar recomendaciones"):

    try:
        recommendations = recommend_hybrid(
            game_name,
            data,
            sim_hybrid_matrix,
            num_results
        )

        st.subheader(f"🎯 Juegos recomendados similares a: **{game_name}**")

        for _, row in recommendations.iterrows():
            st.markdown(f"### 🎮 {row['name']} — ID: {row['appid']}")
            st.write(f"**Géneros:** {row['genres']}")
            st.write(f"**Tags:** {row['steamspy_tags']}")
            st.divider()

    except Exception as e:
        st.error(f"Error: {e}")
