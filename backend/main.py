import io
import numpy as np
from PIL import Image

import tensorflow as tf
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

ANIMAL_MODEL_PATH = "models/best_model_final.keras"

animal_model = tf.keras.models.load_model(
    ANIMAL_MODEL_PATH,
    compile=False
)

ANIMAL_CLASSES = ["cat", "rabbit", "snake"]

ANIMAL_IMAGE_SIZE = (224, 224)


def preprocess_animal_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(ANIMAL_IMAGE_SIZE)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def make_prediction(model, image_array: np.ndarray, class_names: list[str]) -> dict:
    preds = model.predict(image_array, verbose=0)[0]

    if not np.isclose(np.sum(preds), 1.0, atol=1e-2):
        preds = tf.nn.softmax(preds).numpy()

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
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/animal")
async def predict_animal(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение")

    image_array = preprocess_animal_image(image)
    result = make_prediction(animal_model, image_array, ANIMAL_CLASSES)

    return {
        "task": "animal",
        "filename": file.filename,
        **result
    }