from tensorflow.keras.models import load_model
from data_loader import create_data_generators

# Configuration
val_dir = 'data/val'
img_size = 512
batch_size = 32
model_path = 'saved_models/best_model.h5'

# Load the model
model = load_model(model_path)

# Create validation data generator
_, val_generator = create_data_generators('data/train', val_dir, img_size, batch_size)

# Evaluate the model
val_loss, val_acc = model.evaluate(val_generator, steps=val_generator.samples // batch_size)
print(f'Validation Accuracy: {val_acc * 100:.2f}%')

