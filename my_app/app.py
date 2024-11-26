import streamlit as st
from utils.data_loader import load_data
from utils.recommendation import recommend_recipes, train_model
import pickle
import os

# Save the trained model to a file
def save_model(model, filename="models/svd_model.pkl"):
    """Save the trained model to a pickle file."""
    with open(filename, "wb") as file:
        pickle.dump(model, file)

@st.cache_resource
def load_model():
    """Load the pre-trained SVD model from a pickle file."""
    if os.path.exists("models/svd_model.pkl"):
        with open("models/svd_model.pkl", "rb") as file:
            return pickle.load(file)
    else:
        return None

def main():
    st.title("Recipe Recommendation System")

    # Load data
    st.write("[INFO] Loading data...")
    interactions_train, interactions_validation, interactions_test, raw_recipes = load_data()

    # Check for existing model or retrain
    st.sidebar.header("Model Options")
    retrain_model = st.sidebar.checkbox("Retrain Model", value=False)

    if retrain_model:
        st.write("[INFO] Retraining the model. This might take some time...")
        model = train_model(interactions_train, interactions_validation, interactions_test)
        save_model(model)
        st.write("[INFO] Model retrained and saved successfully.")
    else:
        model = load_model()
        if model is None:
            st.error("[ERROR] No pre-trained model found. Please enable retraining.")
            return

    # User inputs for filters
    with st.form("filters_form"):
        calorie_range = st.slider("Select Calorie Range (Calories)", 0.0, 5000.0, (500.0, 2000.0))
        total_fat_range = st.slider("Select Total Fat Range (grams)", 0.0, 200.0, (10.0, 50.0))
        protein_range = st.slider("Select Protein Range (grams)", 0.0, 200.0, (10.0, 50.0))
        carb_range = st.slider("Select Carbohydrate Range (grams)", 0.0, 200.0, (10.0, 50.0))
        time_range = st.slider("Select Cooking Time Range (minutes)", 0, 240, (10, 60))

        # Select tags
        tags = st.multiselect(
            "Select Tags",
            options=['breakfast', 'dinner', 'lunch', 'snacks', 'desserts'],
            default=[]
        )

        # Ingredients to include/exclude
        include_ingredients = st.text_input("Enter Ingredients to Include (comma-separated):")
        exclude_ingredients = st.text_input("Enter Ingredients to Exclude (comma-separated):")

        include_ingredients = [x.strip().lower() for x in include_ingredients.split(",") if x.strip()]
        exclude_ingredients = [x.strip().lower() for x in exclude_ingredients.split(",") if x.strip()]

        # Submit button for the form
        submitted = st.form_submit_button("Get Recommendations")

    # Generate recommendations only when the form is submitted
    if submitted:
        st.write("[INFO] Generating recommendations...")
        recommended_recipes = recommend_recipes(
            calorie_range, total_fat_range, protein_range, carb_range, time_range, tags,
            include_ingredients, exclude_ingredients, raw_recipes, model
        )

        if recommended_recipes.empty:
            st.warning("No recommendations found even after relaxing filters.")
        else:
            for _, row in recommended_recipes.iterrows():
                recipe_url = f"https://www.food.com/recipe/{row['name'].replace(' ', '-').lower()}-{row['id']}"
                st.markdown(f"### [{row['name']}]({recipe_url})")
                st.write(f"**Description**: {row['description']}")
                st.write(f"**Minutes to Cook**: {row['minutes']} minutes")
                st.write(f"**Calories**: {row['calories']} kcal")
                st.write(f"**Macros**: Total Fat: {row['total_fat_grams']} g, Protein: {row['protein_grams']} g, Carbs: {row['carbohydrates_grams']} g")
                st.write("---")

if __name__ == "__main__":
    main()
