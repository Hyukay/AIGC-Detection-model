import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Configuration
model_path = 'saved_models/best_model.h5'
img_size = 512

# Load the model
model = load_model(model_path)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "AI-generated" if prediction[0] > 0.5 else "Photographic"

# Test prediction
img_path = 'path_to_test_image'
print(f"Prediction for {img_path}: {predict_image(img_path)}")
