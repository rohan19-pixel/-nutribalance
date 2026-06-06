import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from datapreparation import get_data_generators  

DATA_DIR = "data/food11 dataset"
MODEL_PATH = "model/nutribalance_model.h5"


def build_model(num_classes):
    base_model = MobileNetV2(
        weights='imagenet',   
        include_top=False,
        input_shape=(224, 224, 3)
    )

    
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train():
    print(" Loading data...")
    train_gen, val_gen, test_gen = get_data_generators(DATA_DIR)

    print(" Classes:", train_gen.class_indices)

    print(" Building model...")
    model = build_model(train_gen.num_classes)

    print(" Starting training...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        verbose=1
    )

    print(" Evaluating on test data...")
    loss, acc = model.evaluate(test_gen)
    print(" Test Accuracy:", acc)

    print(" Saving model...")
    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)

    print(" Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    train()