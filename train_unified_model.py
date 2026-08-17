import tensorflow as tf
from tf_keras.applications import MobileNetV2
from tf_keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tf_keras.models import Model
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.optimizers import Adam
from tf_keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import json


DATASET_PATH = r"C:\Users\Dell\.cache\kagglehub\datasets\abdallahalidev\plantvillage-dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 
MODEL_NAME = "unified_model.h5"

def train():
    data_dir = None
    for root, dirs, files in os.walk(DATASET_PATH):
        if 'color' in dirs:
            data_dir = os.path.join(root, 'color')
            break
    
    if not data_dir:
        print("Error: 'color' directory not found in", DATASET_PATH)
        return
    
    print(f"Using data directory: {data_dir}")

    # 2. Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Save class indices
    class_indices = train_generator.class_indices
    with open('classes.json', 'w') as f:
        json.dump(class_indices, f)
    print("Saved classes.json")

    # 3. Model Architecture
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False # Freeze base for initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(len(class_indices), activation='softmax')(x)

    model =  Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

   
    checkpoint = ModelCheckpoint(MODEL_NAME, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    
    print("Starting training...")
    try:
        model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stop]
        )
        print(f"Training complete! Model saved as {MODEL_NAME}")
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    train()
