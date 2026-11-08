import numpy as np
from PIL import Image
import io

def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((64, 64))
    
    # Converte para array e normaliza (se você normalizou no treinamento)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Adiciona a dimensão do batch
    return img_array