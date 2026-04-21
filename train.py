import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras.optimizers as optimizers

# Set parameters
img_width, img_height = 224, 224
batch_size = 32
epochs = 20
num_classes = 4

train_data_dir = '../data/Training'
validation_data_dir = '../data/Testing'

# Data Augmentation Setup
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load Generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Build the CNN model architecture using Transfer Learning
def build_model():
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(img_width, img_height, 3)
    )
    
    # Freeze the base model to prevent destroying the pre-trained weights
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()

    # Define Callbacks
    # Ensure models directory exists
    os.makedirs('../models', exist_ok=True)
    
    checkpoint_filepath = '../models/brain_tumor_model.h5'
    checkpoint = ModelCheckpoint(
        checkpoint_filepath, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    print("Starting training process...")
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // batch_size),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // batch_size),
        callbacks=[checkpoint, early_stopping]
    )

    # Final save of the weights
    model.save('../models/brain_tumor_model_final.h5')
    print("Training completed. Final model saved to '../models/brain_tumor_model_final.h5'")
