import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV

def train_model(interactions_train, interactions_validation, interactions_test):
    print("[INFO] Preparing data for training...")
    # Prepare the data for collaborative filtering
    reader = Reader(rating_scale=(0, 5))
    train_data = Dataset.load_from_df(interactions_train[['user_id', 'recipe_id', 'rating']], reader)
    trainset = train_data.build_full_trainset()

    print("[INFO] Setting up hyperparameter grid for SVD...")
    # Expand hyperparameter ranges for better tuning
    param_grid = {
        'n_factors': [50],
        'n_epochs': [20],
        'lr_all': [0.005],
        'reg_all': [0.1]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

    print("[INFO] Starting grid search for hyperparameter tuning...")
    gs.fit(train_data)
    print("[INFO] Grid search completed.")

    # Check and print the best hyperparameters
    best_params = gs.best_params['rmse']
    print(f"[INFO] Best parameters found: {best_params}")

    # Retrieve validation RMSE
    validation_rmse = gs.best_score['rmse']
    print(f"[INFO] Validation RMSE for best model: {validation_rmse:.4f}")

    # Train the best model on the full training set
    print("[INFO] Training the best model with optimal hyperparameters...")
    model = gs.best_estimator['rmse']
    model.fit(trainset)
    print("[INFO] Model training completed.")

    return model

def recommend_recipes(calorie_range, ingredients_include, ingredients_exclude, recipes, ingr_map, model):
    import time

    print("[INFO] Starting recipe recommendation process...")

    # Step 1: Filter recipes by calorie level
    print(f"[INFO] Filtering recipes within calorie range: {calorie_range}")
    filtered_recipes = recipes[
        (recipes["calorie_level"] >= calorie_range[0]) &
        (recipes["calorie_level"] <= calorie_range[1])
    ]
    print(f"[INFO] Number of recipes after calorie filtering: {len(filtered_recipes)}")

    # Step 2: Filter recipes by included ingredients
    start_time = time.time()
    if ingredients_include:
        print(f"[INFO] Including recipes with ingredients: {ingredients_include}")
        include_ids = ingr_map[ingr_map['replaced'].isin(ingredients_include)]['id'].unique()
        filtered_recipes = filtered_recipes[filtered_recipes['ingredient_ids'].apply(
            lambda x: any(ing_id in include_ids for ing_id in eval(x)))]
        print(f"[INFO] Number of recipes after including ingredients: {len(filtered_recipes)}")
    else:
        print("[INFO] No specific ingredients to include.")
    print(f"[INFO] Ingredient inclusion filtering completed in {time.time() - start_time:.2f} seconds")

    # Step 3: Filter recipes by excluded ingredients
    start_time = time.time()
    if ingredients_exclude:
        print(f"[INFO] Excluding recipes with ingredients: {ingredients_exclude}")
        exclude_ids = ingr_map[ingr_map['replaced'].isin(ingredients_exclude)]['id'].unique()
        filtered_recipes = filtered_recipes[~filtered_recipes['ingredient_ids'].apply(
            lambda x: any(ing_id in exclude_ids for ing_id in eval(x)))]
        print(f"[INFO] Number of recipes after excluding ingredients: {len(filtered_recipes)}")
    else:
        print("[INFO] No specific ingredients to exclude.")
    print(f"[INFO] Ingredient exclusion filtering completed in {time.time() - start_time:.2f} seconds")

    # Step 4: Predict user preferences using collaborative filtering
    start_time = time.time()
    print("[INFO] Predicting recipe scores using collaborative filtering model...")
    filtered_recipe_ids = filtered_recipes['id'].values
    predicted_scores = [
        (recipe_id, model.predict(0, recipe_id).est)  # Using a default user_id = 0 for predictions
        for recipe_id in filtered_recipe_ids
    ]
    print(f"[INFO] Prediction completed in {time.time() - start_time:.2f} seconds")

    # Step 5: Sort recipes by predicted score
    print("[INFO] Sorting recipes by predicted scores...")
    sorted_recipes = sorted(predicted_scores, key=lambda x: x[1], reverse=True)
    top_recipe_ids = [recipe[0] for recipe in sorted_recipes[:10]]  # Top 10 recommendations

    # Step 6: Map recommended recipe IDs to their details
    print("[INFO] Mapping recommended recipe IDs to details...")
    recommended_recipes = recipes[recipes['id'].isin(top_recipe_ids)][
        ['id', 'name_tokens', 'calorie_level', 'ingredient_tokens']
    ]
    print("[INFO] Recommendation process completed.")
    
    return recommended_recipes
