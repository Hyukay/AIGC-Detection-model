import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from data_loader import create_data_generators

# Configuration
train_dir = 'data/train'
val_dir = 'data/val'
img_size = 512
batch_size = 32
epochs = 20
model_save_path = 'saved_models/best_model.h5'
num_classes = 10  # Change this to the actual number of classes in your dataset

# Create data generators
train_generator, val_generator = create_data_generators(train_dir, val_dir, img_size, batch_size)

# Create the model with ConvNeXt
def create_model(img_size, num_classes):
    base_model = ConvNeXtBase(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False  # Freeze the base model if you only want to fine-tune the top layers
    
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model(img_size, num_classes)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size,
                    epochs=epochs,
                    callbacks=[early_stop, checkpoint])

print("Training complete. Best model saved at:", model_save_path)
