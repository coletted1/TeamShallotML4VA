import streamlit as st
from utils.data_loader import load_data
from utils.recommendation import train_model, recommend_recipes
import pandas as pd

# Streamlit UI
st.title("Recipe Recommendation System")
st.write("Get personalized recipe recommendations based on your preferences!")

# Step 1: Load Data
st.header("Step 1: Load Data")
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

if st.button("Load Data"):
    st.write("[INFO] Loading data...")
    try:
        (
            st.session_state["interactions_train"],
            st.session_state["interactions_validation"],
            st.session_state["interactions_test"],
            st.session_state["recipes"],
            st.session_state["ingr_map"],
        ) = load_data()
        st.session_state["data_loaded"] = True
        st.success("[INFO] Data loaded successfully.")
    except Exception as e:
        st.error(f"[ERROR] Failed to load data: {e}")
        raise

# Step 2: Train Model
st.header("Step 2: Train Model")
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False

if st.session_state["data_loaded"]:
    if st.button("Train Model"):
        st.write("[INFO] Starting model training...")
        try:
            st.session_state["model"] = train_model(
                st.session_state["interactions_train"],
                st.session_state["interactions_validation"],
                st.session_state["interactions_test"],
            )
            st.session_state["model_trained"] = True
            st.success("[INFO] Model training completed successfully.")
        except Exception as e:
            st.error(f"[ERROR] Model training failed: {e}")
            raise
else:
    st.warning("Please load the data before training the model.")

# Step 3: User Preferences
st.header("Step 3: Enter Your Preferences")
if st.session_state["model_trained"]:
    st.subheader("Calorie Level")
    calorie_min = st.slider("Minimum Calorie Level", 0, 2, 1)
    calorie_max = st.slider("Maximum Calorie Level", 0, 2, 2)
    calorie_range = (calorie_min, calorie_max)

    st.subheader("Ingredients")
    ingredients_include = st.text_input(
        "Ingredients to include (comma-separated)", "lettuce, tomato"
    ).split(", ")
    ingredients_exclude = st.text_input(
        "Ingredients to exclude (comma-separated)", "chicken"
    ).split(", ")

    # Step 4: Generate Recommendations
    st.header("Step 4: Generate Recommendations")
    if st.button("Get Recommendations"):
        st.write("[INFO] Generating recommendations...")
        try:
            recommended_recipes = recommend_recipes(
                calorie_range=calorie_range,
                ingredients_include=ingredients_include,
                ingredients_exclude=ingredients_exclude,
                recipes=st.session_state["recipes"],
                ingr_map=st.session_state["ingr_map"],
                model=st.session_state["model"],
            )
            if recommended_recipes.empty:
                st.warning("No recommendations found for the given inputs.")
            else:
                st.success("Recommendations generated successfully!")
                st.subheader("Recommended Recipes")
                st.write(
                    recommended_recipes[
                        ["id", "calorie_level", "ingredient_tokens"]
                    ]
                )
        except Exception as e:
            st.error(f"[ERROR] Failed to generate recommendations: {e}")
            raise
else:
    st.warning("Please train the model before entering preferences.")
