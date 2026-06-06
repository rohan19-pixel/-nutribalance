# recommendation_engine.py

DAILY_GOALS = {
    "calories": 2000,
    "protein": 50,
    "carbs": 250,
    "fat": 70
}


def recommend_next_meal(intake, goal="balanced", meal_time="general"):
    """
    Real-time rule-based dietary recommendation engine
    intake: dictionary with current daily intake
    goal: user goal (weight_loss / muscle_gain / balanced)
    meal_time: morning / afternoon / night
    """

    advice = []

    # ---------------- BASIC RULES ---------------- #

    if intake["carbs"] > DAILY_GOALS["carbs"] * 0.5:
        advice.append(" High carbs intake. Choose low-carb foods like salad or paneer.")

    if intake["fat"] > DAILY_GOALS["fat"] * 0.5:
        advice.append(" High fat intake. Prefer grilled or steamed food.")

    if intake["protein"] < DAILY_GOALS["protein"] * 0.3:
        advice.append(" Low protein intake. Add eggs, chicken, tofu, or dal.")

    if intake["calories"] > DAILY_GOALS["calories"] * 0.6:
        advice.append(" High calorie intake. Keep your next meal light.")

    # ---------------- GOAL BASED ---------------- #

    if goal == "weight_loss":
        advice.append(" For weight loss: Focus on low-calorie, high-fiber foods.")

    elif goal == "muscle_gain":
        advice.append(" For muscle gain: Increase protein and calorie intake.")

    elif goal == "balanced":
        advice.append(" Maintain a balanced diet with proper macros.")

    # ---------------- TIME BASED ---------------- #

    if meal_time == "morning":
        advice.append("Breakfast tip: Include protein + healthy carbs (eggs, oats).")

    elif meal_time == "afternoon":
        advice.append(" Lunch tip: Balanced meal with carbs + protein + veggies.")

    elif meal_time == "night":
        advice.append(" Dinner tip: Keep it light and low in carbs.")

    # ---------------- FINAL FALLBACK ---------------- #

    if not advice:
        advice.append(" Your diet looks balanced. Keep it up!")

    return advice