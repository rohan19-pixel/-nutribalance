import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from nutrition_database import nutrition_data
from recommendation_engine import recommend_next_meal   # ✅ NEW
import matplotlib.pyplot as plt

# Load model
model = load_model("model/nutribalance_model.h5")

# Class labels
class_labels = {
    0: "Bread",
    1: "Dairy",
    2: "Dessert",
    3: "Egg",
    4: "Fried Food",
    5: "Meat"
}

# Page config
st.set_page_config(page_title="NutriBalance AI", page_icon="🍔", layout="centered")

# Styling
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff914d);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🍽 NutriBalance AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload food → Get nutrition + AI suggestions</p>", unsafe_allow_html=True)

# -------- SESSION STATE (REAL-TIME TRACKING) --------
if "daily_intake" not in st.session_state:
    st.session_state.daily_intake = {
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "fat": 0
    }

# -------- USER OPTIONS --------
goal = st.selectbox("🎯 Select Your Goal", ["balanced", "weight_loss", "muscle_gain"])
meal_time = st.selectbox("🕒 Meal Time", ["morning", "afternoon", "night"])

# Upload
uploaded_file = st.file_uploader("📸 Upload Food Image", type=["jpg", "png", "jpeg"])

# Prediction function
def predict_food(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    food = class_labels[class_index]
    nutrition = nutrition_data.get(food, {})

    return food, nutrition


# -------- MAIN APP --------
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption=" Uploaded Image", use_container_width=True)

    grams = st.slider(" Select Portion Size (grams)", 50, 500, 100)

    if st.button(" Analyze Food"):
        with st.spinner("Analyzing your food..."):
            food, nutrition = predict_food(img)

        st.success(f" Detected Food: {food}")

        # Nutrition calculation
        calories = nutrition["calories"] * (grams / 100)
        carbs = nutrition["carbs"] * (grams / 100)
        protein = nutrition["protein"] * (grams / 100)
        fat = nutrition["fat"] * (grams / 100)

        # -------- UPDATE DAILY INTAKE --------
        st.session_state.daily_intake["calories"] += calories
        st.session_state.daily_intake["carbs"] += carbs
        st.session_state.daily_intake["protein"] += protein
        st.session_state.daily_intake["fat"] += fat

        # Metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric(" Calories", round(calories, 2))
            st.metric(" Carbs", round(carbs, 2))

        with col2:
            st.metric(" Protein", round(protein, 2))
            st.metric(" Fat", round(fat, 2))

        st.markdown("---")

        # PIE CHART
        st.subheader("Macronutrient Distribution")

        labels = ['Carbs', 'Protein', 'Fat']
        values = [carbs, protein, fat]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')

        st.pyplot(fig)

        st.markdown("---")

        # -------- REAL-TIME SUGGESTIONS --------
        advice = recommend_next_meal(
            st.session_state.daily_intake,
            goal=goal,
            meal_time=meal_time
        )

        st.subheader(" Real-Time Dietary Suggestions")

        for tip in advice:
            st.warning(tip)

        st.markdown("---")

        st.info(f" Nutrition calculated for {grams}g of {food}")

# -------- RESET BUTTON --------
if st.button("🔄    Reset Daily Intake"):
    st.session_state.daily_intake = {
        "calories": 0,
        "protein": 0,
        "carbs": 0,
        "fat": 0
    }
    st.success("Daily intake reset!")

# Footer
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>Built with  | NutriBalance AI</p>
""", unsafe_allow_html=True)