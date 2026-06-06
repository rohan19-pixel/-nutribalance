import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from nutrition_database import nutrition_data

print("Prediction system running...")

# Load model
model = load_model("model/nutribalance_model.h5")

# Class labels (must match training)
class_labels = {
    0: "Bread",
    1: "Dairy",
    2: "Dessert",
    3: "Egg",
    4: "Fried Food",
    5: "Meat"
}

def predict_food(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    food_name = class_labels[class_index]
    nutrition = nutrition_data.get(food_name, {})

    return food_name, nutrition


if __name__ == "__main__":
    # Input image
    img_path = input("Enter image path: ").strip().strip('"')

    #  Predict food
    food, nutrition = predict_food(img_path)

    print("\n Food Detected:", food)

    #  Ask for portion size
    grams = float(input("Enter quantity (grams): "))

    #  Calculate dynamic nutrition
    calories = nutrition["calories"] * (grams / 100)
    carbs = nutrition["carbs"] * (grams / 100)
    protein = nutrition["protein"] * (grams / 100)
    fat = nutrition["fat"] * (grams / 100)

    #  Output
    print("\n Nutrition for", grams, "g:")
    print("Calories:", round(calories, 2))
    print("Carbs:", round(carbs, 2))
    print("Protein:", round(protein, 2))
    print("Fat:", round(fat, 2))
    print("\n prediction   of calories , fat , carbs and protein have been completed  successfully ")