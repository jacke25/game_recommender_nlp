# 🎮 Steam Game Recommender System  
**Sistema de recomendación híbrido con NLP + Machine Learning (TF–IDF Multicomponente)**  

Este proyecto implementa un recomendador de videojuegos basado en texto utilizando descripciones, géneros y etiquetas de más de **27,000 juegos de Steam**.  
Se aplican técnicas avanzadas de **Procesamiento de Lenguaje Natural (NLP)**, **vectorización TF-IDF**, **similaridad del coseno**, e integración de múltiples representaciones textuales para generar recomendaciones altamente precisas.

---

## 📌 Tabla de Contenido
- [🎯 Objetivo del Proyecto](#-objetivo-del-proyecto)
- [🧠 Arquitectura y Metodología](#-arquitectura-y-metodología)
- [📊 Dataset](#-dataset)
- [📓 Notebooks del Proyecto](#️notebooks-del-proyecto)
- [🖥️ Archivo Principal de la Aplicación](#️-archivo-principal-de-la-aplicación)
- [🛠️ Procesamiento y Feature Engineering](#️-procesamiento-y-feature-engineering)
- [🤖 Modelos Utilizados](#-modelos-utilizados)
- [🧩 Sistema de Recomendación](#-sistema-de-recomendación)
- [📈 Resultados](#-resultados)
- [🖥️ Aplicación (versión local)](#️-aplicación-versión-local)
- [🎬 Video de Demostración](#-video-de-demostración)
- [📦 Requerimientos](#-requerimientos)
- [🔍 Próximas Mejoras](#-próximas-mejoras)
- [👤 Autor](#-autor)

---

## 🎯 Objetivo del Proyecto

El objetivo principal es construir un **sistema de recomendación de videojuegos** capaz de sugerir títulos similares utilizando exclusivamente información textual.  
Este modelo se basa en:

- Descripción lematizada del juego  
- Lista de géneros  
- Tags de SteamSpy  
- Vectorización TF-IDF  
- Similaridad del coseno  
- Esquema híbrido con ponderaciones  

El propósito final del proyecto es demostrar la capacidad de:

✔ Trabajar con datasets grandes  
✔ Crear pipelines de NLP completos  
✔ Entrenar modelos reproducibles  
✔ Desarrollar una app interactiva tipo producto  
✔ Explicar un proyecto de machine learning a nivel profesional  

---

## 🧠 Arquitectura y Metodología

El pipeline completo incluye:

### **1️⃣ Ingesta y limpieza del dataset**
- Manejo de nulos
- Unificación de columnas
- Normalización y limpieza de texto
- Conversión de campos a formato estándar

### **2️⃣ Procesamiento NLP**
- Tokenización  
- Lematización  
- Removal de stopwords  
- Conversión a minúsculas  
- Limpieza de caracteres especiales  

### **3️⃣ Feature Engineering**
- TF-IDF de **descripciones**
- TF-IDF de **tags**
- TF-IDF de **géneros**

### **4️⃣ Construcción de matrices de similitud**
Cada característica genera su propia matriz de similitud mediante **cosine similarity**.

### **5️⃣ Ensamble híbrido**
Las matrices se combinan bajo esta fórmula:

Hybrid = 0.60 * Descriptions
+ 0.25 * Tags
+ 0.15 * Genres


Con esto logramos capturar:

- Semántica → descripción
- Intención del jugador → tags
- Tipo general de juego → géneros

---

## 📓 Notebooks del Proyecto

Este repositorio incluye **tres notebooks principales**, cada uno documentando una etapa crítica del pipeline del sistema de recomendación. En conjunto, ofrecen una trazabilidad completa del desarrollo: desde el análisis inicial hasta el modelo híbrido final y la app funcional.

---

### **📘 01_EDA__preprocessing_game_recomendation_project.ipynb**
Primer notebook del proyecto. Aquí se realiza:

- Análisis exploratorio del dataset (tamaño, estructura, duplicados, valores faltantes)
- Limpieza completa de datos
- Normalización y transformación de columnas clave
- Procesamiento de texto:
  - Limpieza profunda  
  - Tokenización  
  - Lematización  
- Conversión de géneros y etiquetas al formato de strings estandarizados utilizados por la app
- Exportación del **dataset preprocesado final**

Este notebook establece los cimientos para todos los modelos posteriores.

---

### **📗 02_tfidf_similarity.ipynb**
Notebook dedicado exclusivamente al **modelo basado únicamente en descripciones**.

Incluye:

- Construcción del `TfidfVectorizer()` para descripciones lematizadas
- Entrenamiento del vectorizador (fit)
- Transformación TF-IDF de todas las descripciones
- Cálculo de la **matriz de similitud basada solo en texto descriptivo**
- Evaluación preliminar de resultados.

Este notebook permite comparar qué tan bueno sería el sistema si solo usara texto, sin tags ni géneros.

---

### **📙 03_hybrid_model_genre_tags_decription.ipynb**
Notebook central del proyecto — aquí se construye el **modelo híbrido completo** utilizado en la app.

Incluye:

- Generación del TF-IDF para etiquetas
- Generación del TF-IDF para géneros
- Creación de matrices de similitud independientes por cada característica
- Combinación ponderada en un **modelo híbrido**:
  - **60% Descripción**
  - **25% Etiquetas**
  - **15% Géneros**
- Evaluación del rendimiento comparado vs. modelos simples
- Construcción de la función de recomendación híbrida final
- Exportación de función de recomendación y de las matriz hibrida

Este es el notebook que contiene la versión más completa y precisa del recomendador.

---

## 🖥️ Archivo Principal de la Aplicación

El archivo `app.py` contiene la implementación completa del sistema de recomendación interactivo desarrollado con **Streamlit**:

- Descarga de los recursos (dataset, vectorizador, matriz híbrida)
- Carga y preparación de los assets generados en los notebooks
- Implementación de la función `recommend_hybrid()` utilizada en producción
- Interfaz gráfica completamente funcional con:
  - Menú de selección de juego
  - Selector de número de recomendaciones
  - Despliegue de géneros, tags y appid.
- Arquitectura diseñada para reproducir fielmente los resultados obtenidos en los notebooks

Este archivo representa el front-end y el motor de ejecución del modelo híbrido final.

---

## 📊 Dataset

- **27,075 videojuegos**
- Columnas incluidas:
  - `name`
  - `appid`
  - `description_lemm`
  - `genres`
  - `steamspy_tags`
  - Otros atributos relevantes para limpiar y enriquecer los datos

📌 *Enlace al dataset procesado:*  
👉 https://huggingface.co/datasets/Jacke23/steam-recommender-assets/resolve/main/steam_games_preprocessed.csv

---

## 🛠️ Procesamiento y Feature Engineering

El dataset se preprocesó en varios notebooks, donde se realizaron:

### ✔ EDA y limpieza completa  
### ✔ Lematización con spaCy  
### ✔ Uniformización de campos textuales  
### ✔ Ingeniería de características  
### ✔ Creación de matrices TF-IDF separadas  
### ✔ Generación de la matriz híbrida final  

---

## 🤖 Modelos Utilizados

### **TF-IDF Vectorizers**
- TF-IDF para descripción 
- TF-IDF para tags  
- TF-IDF para géneros  

### **Métricas**
- Cosine Similarity

### **Modelo final**
- Recomendador híbrido basado en similitud vectorial

📌 *Enlaces vectorizador y matriz de similitud hibrida:*

👉 https://huggingface.co/datasets/Jacke23/steam-recommender-assets/resolve/main/tfidf_vectorizer.pkl

👉 https://huggingface.co/datasets/Jacke23/steam-recommender-assets/resolve/main/similarity_hybrid.pkl

---

## 🧩 Sistema de Recomendación
La app calcula recomendaciones así:

1. Toma un juego seleccionado por el usuario  
2. Obtiene su vector TF-IDF  
3. Calcula el índice en la matriz híbrida  
4. Ordena los juegos más similares  
5. Retorna el top *N* solicitado  

Este enfoque permite:

- Recomendaciones coherentes
- Alta velocidad de inferencia
- Bajo costo computacional para búsqueda
- Explicabilidad del modelo

---

## 📈 Resultados

Durante las pruebas se observaron:

- Recomendaciones altamente relevantes  
- Juegos similares en temática, estilo y juego real  
- Robustez al variar ponderaciones del modelo  
- Comportamiento estable incluso con 27k instancias  

Ejemplo:

> Si eliges **Counter-Strike**, recomienda títulos con tags similares como acción, FPS, y multijugador.

---

## 🖥️ Aplicación (versión local)

La aplicación está desarrollada con **Streamlit**, permitiendo:

- Seleccionar un juego  
- Elegir número de recomendaciones  
- Ver títulos similares con sus géneros y tags  
- Interfaz limpia y rápida  

Debido al tamaño de la matriz (~5GBs), la app se ejecuta **exclusivamente en local**.

---

## 🎬 Video de Demostración

👉 https://www.loom.com/share/e925286d05ff4d34b437f410c6249e4a

---

## 📦 Requerimientos

- streamlit
- pandas
- numpy
- scikit-learn==1.7.1
- huggingface-hub

---

## 🔍 Próximas Mejoras

- Implementar embeddings con BERT o Sentence Transformers
- Integrar FAISS para búsquedas vectoriales más rápidas
- Optimizar peso del modelo para deploy cloud
- Dashboard con estadísticas de usuarios
- Versión mobile de la app

---

## 👤 Autor

**Javier Andrés Díaz Charry**\
Data Science • NLP • Machine Learning\
📍 Huesca, España\
🎮 Apasionado por IA, videojuegos y ciencia de datos\
🔗 www.linkedin.com/in/javier-andres-diaz-data-scientist\
