import pandas as pd

def load_data():
    interactions_train = pd.read_csv("data/interactions_train.csv")
    interactions_validation = pd.read_csv("data/interactions_validation.csv")
    interactions_test = pd.read_csv("data/interactions_test.csv")
    raw_recipes = pd.read_csv("data/RAW_recipes.csv")
    return interactions_train, interactions_validation, interactions_test, raw_recipes
