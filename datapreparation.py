import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def find_folder(base_dir, possible_names):
    """Finds the correct folder automatically"""
    for name in possible_names:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            print(f"✅ Found folder: {name}")
            return path
    raise FileNotFoundError(f"❌ None of these folders found: {possible_names}")


def get_data_generators(data_dir):
    """
    Works with:
    train OR training
    evaluate OR evaluation OR test
    """

    
    train_dir = find_folder(data_dir, ["train", "training"])
    val_dir = find_folder(data_dir, ["validation", "valid"])
    test_dir = find_folder(data_dir, ["evaluate", "evaluation", "test"])

    
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)


    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    
    val_gen = test_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    #
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_gen, val_gen, test_gen