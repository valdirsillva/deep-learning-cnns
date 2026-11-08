import io
from app.utils.image import process_image
from fastapi import APIRouter, File, Form, UploadFile, status
import tensorflow as tf

router = APIRouter(prefix="/api/v1", tags=["Endpoints"])

_model = tf.keras.models.load_model("app/models/classificador_gato_cachorro_v2.keras")

@router.post("/predict", status_code=status.HTTP_200_OK)
async def predict_image(file: UploadFile = File(...)):
    # Read the uploaded file
    content = await file.read()

    # prepare the image for prediction (this is a placeholder, implement your own logic)
    image_processed = process_image(content)

    # execute predict 
    prediction = _model.predict(image_processed)

    classe = "gato" if prediction[0][0] > 0.5 else "cachorro"

    trust = float(prediction[0][0]) if classe == "gato" else float(1 - prediction[0][0])

    return {"label": classe, "confidence": trust}

