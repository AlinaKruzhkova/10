import io
import os
import traceback

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Animal Classification API",
    version="1.0.0",
    description="API для классификации изображений животных"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Путь к TFLite-модели.
# В Render у тебя Root Directory = backend, поэтому путь начинается с models/
ANIMAL_MODEL_PATH = "models/animal_model.tflite"

# Классы должны быть в том же порядке, что и при обучении модели
ANIMAL_CLASSES = ["cat", "rabbit", "snake"]

# Размер изображения должен совпадать с размером, на котором обучалась модель
ANIMAL_IMAGE_SIZE = (224, 224)


animal_interpreter = None
animal_input_details = None
animal_output_details = None


def get_animal_model():
    """
    Ленивая загрузка TFLite-модели.
    Модель загружается только при первом запросе /predict/animal,
    чтобы Render успевал быстро запустить сервер и открыть порт.
    """
    global animal_interpreter, animal_input_details, animal_output_details

    if animal_interpreter is None:
        import tensorflow as tf

        print("Loading TFLite animal model...")
        print("Current working directory:", os.getcwd())
        print("Model path:", ANIMAL_MODEL_PATH)
        print("Model exists:", os.path.exists(ANIMAL_MODEL_PATH))

        if not os.path.exists(ANIMAL_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {ANIMAL_MODEL_PATH}")

        animal_interpreter = tf.lite.Interpreter(model_path=ANIMAL_MODEL_PATH)
        animal_interpreter.allocate_tensors()

        animal_input_details = animal_interpreter.get_input_details()
        animal_output_details = animal_interpreter.get_output_details()

        print("TFLite animal model loaded")
        print("Input details:", animal_input_details)
        print("Output details:", animal_output_details)

    return animal_interpreter


def preprocess_animal_image(image: Image.Image) -> np.ndarray:
    """
    Предобработка изображения:
    - перевод в RGB;
    - изменение размера до 224x224;
    - нормализация пикселей в диапазон [0, 1];
    - добавление batch dimension: (1, 224, 224, 3).
    """
    image = image.convert("RGB")
    image = image.resize(ANIMAL_IMAGE_SIZE)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax через NumPy, чтобы не тянуть TensorFlow отдельно внутри предсказания.
    """
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def prepare_input_for_tflite(image_array: np.ndarray) -> np.ndarray:
    """
    Подготавливает input под тип данных TFLite-модели.
    Если модель float32 — подаём float32.
    Если модель quantized — учитываем scale и zero_point.
    """
    input_detail = animal_input_details[0]
    input_dtype = input_detail["dtype"]

    if input_dtype == np.float32:
        return image_array.astype(np.float32)

    scale, zero_point = input_detail["quantization"]

    if scale and scale > 0:
        input_data = image_array / scale + zero_point
        input_data = np.clip(input_data, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
        return input_data.astype(input_dtype)

    return image_array.astype(input_dtype)


def read_output_from_tflite(raw_output: np.ndarray) -> np.ndarray:
    """
    Получает output модели.
    Если output quantized — переводит обратно в float.
    """
    output_detail = animal_output_details[0]
    output_dtype = output_detail["dtype"]

    preds = raw_output[0]

    if output_dtype != np.float32:
        scale, zero_point = output_detail["quantization"]

        if scale and scale > 0:
            preds = scale * (preds.astype(np.float32) - zero_point)
        else:
            preds = preds.astype(np.float32)

    return preds


def make_prediction(interpreter, image_array: np.ndarray, class_names: list[str]) -> dict:
    """
    Выполняет предсказание через TFLite Interpreter.
    Возвращает:
    - predicted_class;
    - confidence;
    - probabilities.
    """
    input_data = prepare_input_for_tflite(image_array)

    input_index = animal_input_details[0]["index"]
    output_index = animal_output_details[0]["index"]

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_index)
    preds = read_output_from_tflite(raw_output)

    # Если модель вернула logits, а не вероятности, применяем softmax
    if not np.isclose(np.sum(preds), 1.0, atol=1e-2):
        preds = softmax(preds)

    predicted_idx = int(np.argmax(preds))
    predicted_class = class_names[predicted_idx]
    confidence = float(preds[predicted_idx])

    probabilities = {
        class_names[i]: float(preds[i]) for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }


@app.get("/")
def root():
    return {
        "message": "API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": animal_interpreter is not None,
        "model_path": ANIMAL_MODEL_PATH,
        "model_exists": os.path.exists(ANIMAL_MODEL_PATH)
    }


@app.post("/predict/animal")
async def predict_animal(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Файл должен быть изображением"
            )

        image_bytes = await file.read()

        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Не удалось прочитать изображение"
            )

        image_array = preprocess_animal_image(image)

        model = get_animal_model()
        result = make_prediction(model, image_array, ANIMAL_CLASSES)

        return {
            "task": "animal",
            "filename": file.filename,
            **result
        }

    except HTTPException:
        raise

    except Exception as e:
        print("PREDICTION ERROR:", repr(e))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при предсказании: {repr(e)}"
        )