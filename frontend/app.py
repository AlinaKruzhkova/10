import io
import os
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path


st.set_page_config(
    page_title="Классификация животных",
    layout="wide"
)

# Базовый URL API
API_BASE_URL = "https://image-classification-project-nl0r.onrender.com/predict"
ANIMAL_API_URL = f"{API_BASE_URL}/predict/animal"

# Пути к изображениям с метриками
BASE_DIR = Path(__file__).parent
CONFUSION_PATH = BASE_DIR / "assets" / "confusion_matrices.png"
METRICS_PATH = BASE_DIR / "assets" / "metrics_comparison.png"


def prepare_image_bytes(image: Image.Image) -> bytes:
    """
    Преобразование изображения в PNG-байты для отправки в API.
    """
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def send_to_api(api_url: str, image: Image.Image) -> dict:
    """
    Отправка изображения в FastAPI.
    """
    image_bytes = prepare_image_bytes(image)

    files = {
        "file": ("image.png", image_bytes, "image/png")
    }

    response = requests.post(api_url, files=files, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return response.json()


def show_prediction_result(result: dict):
    """
    Отображение результатов классификации.
    """
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    st.success("Классификация выполнена успешно")
    st.subheader("Результат классификации")
    st.write(f"**Предсказанный класс:** {predicted_class}")
    st.write(f"**Уверенность модели:** {confidence:.4f}")

    df = pd.DataFrame({
        "Класс": list(probabilities.keys()),
        "Вероятность": list(probabilities.values())
    }).sort_values(by="Вероятность", ascending=False)

    st.subheader("Вероятности по классам")
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Класс"))


def show_metrics_images():
    """
    Отображение матриц ошибок и сравнительных метрик.
    """
    st.subheader("Оценка качества моделей")

    col1, col2 = st.columns(2)

    with col1:
        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), caption="Матрицы ошибок моделей", use_container_width=True)
        else:
            st.warning(f"🖼 Файл не найден: {CONFUSION_PATH}")
            st.info(f"Текущая директория: {Path.cwd()}")
            st.info(f"Содержимое assets: {list((BASE_DIR / 'assets').glob('*')) if (BASE_DIR / 'assets').exists() else 'Папка assets не существует'}")

    with col2:
        if METRICS_PATH.exists():
            st.image(str(METRICS_PATH), caption="Сравнение метрик моделей", use_container_width=True)
        else:
            st.warning(f"🖼 Файл не найден: {METRICS_PATH}")


def animal_page():
    st.title("Классификация животных")

    tabs = st.tabs(["Классификация", "Метрики и матрицы ошибок"])

    with tabs[0]:
        uploaded_file = st.file_uploader(
            "Загрузите изображение животного: кот, кролик, змея",
            type=["png", "jpg", "jpeg"],
            key="animal_uploader"
        )

        image = None

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Загруженное изображение", width=350)

        if image is not None:
            if st.button("Классифицировать животное"):
                with st.spinner("Выполняется классификация..."):
                    try:
                        result = send_to_api(ANIMAL_API_URL, image)
                        show_prediction_result(result)
                    except Exception as e:
                        st.error("Ошибка при обращении к API")
                        st.code(str(e))

    with tabs[1]:
        show_metrics_images()


# Боковое меню с двумя страницами
st.sidebar.title("Меню")
page = st.sidebar.radio(
    "Выберите страницу",
    [
        "Классификация животных"
    ]
)

if page == "Классификация цифр (MNIST)":
    digit_page()
else:
    animal_page()