import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import GridSearchCV, train_test_split
from surprise import accuracy

from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import GridSearchCV

def train_model(interactions_train, interactions_validation, interactions_test):
    print("[INFO] Preparing data for training...")

    # Prepare the data for collaborative filtering
    reader = Reader(rating_scale=(0, 5))

    # Load training data
    train_data = Dataset.load_from_df(interactions_train[['user_id', 'recipe_id', 'rating']], reader)
    trainset = train_data.build_full_trainset()

    # Prepare validation and test data
    validation_data = [
        (row['user_id'], row['recipe_id'], row['rating'])
        for _, row in interactions_validation.iterrows()
    ]
    test_data = [
        (row['user_id'], row['recipe_id'], row['rating'])
        for _, row in interactions_test.iterrows()
    ]

    print("[INFO] Setting up hyperparameter grid for SVD...")
    param_grid = {
        'n_factors': [50, 100],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)

    print("[INFO] Starting grid search for hyperparameter tuning...")
    gs.fit(train_data)
    print("[INFO] Grid search completed.")

    # Get the best model and hyperparameters
    best_params = gs.best_params['rmse']
    print(f"[INFO] Best parameters found: {best_params}")
    validation_rmse = gs.best_score['rmse']
    print(f"[INFO] Best RMSE on validation data: {validation_rmse:.4f}")

    print("[INFO] Training the best model with optimal hyperparameters...")
    model = gs.best_estimator['rmse']
    model.fit(trainset)
    print("[INFO] Model training completed.")

    # Evaluate on the test set
    print("[INFO] Evaluating model on the test set...")
    predictions = model.test(test_data)
    test_rmse = accuracy.rmse(predictions, verbose=True)
    print(f"[INFO] RMSE on test data: {test_rmse:.4f}")

    # Save the trained model
    with open("models/svd_model.pkl", "wb") as file:
        pickle.dump(model, file)
    print("[INFO] Model saved to 'models/svd_model.pkl'.")

    return model



def recommend_recipes(calorie_range, total_fat_range, protein_range, carb_range, time_range, tags,
                      include_ingredients, exclude_ingredients, raw_recipes, model):
    def filter_recipes(recipes, calorie_range, total_fat_range, protein_range, carb_range, time_range, tags):
        # Filter by calorie range
        recipes = recipes[
            (recipes['calories'] >= calorie_range[0]) & (recipes['calories'] <= calorie_range[1])
        ]

        # Filter by fat, protein, and carb ranges
        recipes = recipes[
            (recipes['total_fat_grams'] >= total_fat_range[0]) & (recipes['total_fat_grams'] <= total_fat_range[1]) &
            (recipes['protein_grams'] >= protein_range[0]) & (recipes['protein_grams'] <= protein_range[1]) &
            (recipes['carbohydrates_grams'] >= carb_range[0]) & (recipes['carbohydrates_grams'] <= carb_range[1])
        ]

        # Filter by cooking time
        recipes = recipes[
            (recipes['minutes'] >= time_range[0]) & (recipes['minutes'] <= time_range[1])
        ]

        # Filter by tags (if any tags are selected)
        if tags:
            recipes = recipes[recipes['tags'].apply(lambda x: any(tag in x for tag in tags))]

        return recipes

    def filter_by_ingredients(recipes, include_ingredients, exclude_ingredients):
        # Filter by included ingredients
        if include_ingredients:
            recipes = recipes[
                recipes['ingredients'].apply(lambda x: all(ing in x for ing in include_ingredients))
            ]
        
        # Filter by excluded ingredients
        if exclude_ingredients:
            recipes = recipes[
                ~recipes['ingredients'].apply(lambda x: any(ing in x for ing in exclude_ingredients))
            ]
        
        return recipes

    # Step 1: Initial filtering
    filtered_recipes = filter_recipes(raw_recipes, calorie_range, total_fat_range, protein_range, carb_range, time_range, tags)
    filtered_recipes = filter_by_ingredients(filtered_recipes, include_ingredients, exclude_ingredients)

    # Step 2: Try relaxing filters
    if filtered_recipes.empty:
        print("[INFO] No recipes found. Relaxing tag filters...")
        # Remove tag filters and try again
        filtered_recipes = filter_recipes(raw_recipes, calorie_range, total_fat_range, protein_range, carb_range, time_range, [])
        filtered_recipes = filter_by_ingredients(filtered_recipes, include_ingredients, exclude_ingredients)

    # Step 3: Relax ranges further if still empty
    if filtered_recipes.empty:
        print("[INFO] No recipes found. Expanding ranges...")
        relaxed_calorie_range = (max(0, calorie_range[0] - 500), calorie_range[1] + 500)
        relaxed_fat_range = (max(0, total_fat_range[0] - 10), total_fat_range[1] + 10)
        relaxed_protein_range = (max(0, protein_range[0] - 10), protein_range[1] + 10)
        relaxed_carb_range = (max(0, carb_range[0] - 10), carb_range[1] + 10)
        relaxed_time_range = (max(0, time_range[0] - 10), time_range[1] + 10)

        filtered_recipes = filter_recipes(raw_recipes, relaxed_calorie_range, relaxed_fat_range,
                                          relaxed_protein_range, relaxed_carb_range, relaxed_time_range, [])
        filtered_recipes = filter_by_ingredients(filtered_recipes, include_ingredients, exclude_ingredients)

    # Step 4: Return no recipes if still empty
    if filtered_recipes.empty:
        print("[INFO] No recipes found after relaxing all filters.")
        return pd.DataFrame()

    # Step 5: Predict user preferences using collaborative filtering
    print("[INFO] Predicting recipe scores...")
    recipe_ids = filtered_recipes['id'].values
    predicted_scores = [
        (recipe_id, model.predict(0, recipe_id).est)  # Default user ID for prediction
        for recipe_id in recipe_ids
    ]

    # Sort recipes by predicted score
    print("[INFO] Sorting recipes by predicted scores...")
    sorted_recipes = sorted(predicted_scores, key=lambda x: x[1], reverse=True)
    top_recipe_ids = [recipe[0] for recipe in sorted_recipes[:10]]  # Top 10 recommendations

    print("[INFO] Mapping recommended recipe IDs to details...")
    return raw_recipes[raw_recipes['id'].isin(top_recipe_ids)]
