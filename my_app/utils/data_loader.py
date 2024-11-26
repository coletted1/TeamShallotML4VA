import pandas as pd
import pickle

def load_data():
    interactions_train = pd.read_csv('data/interactions_train.csv')
    interactions_validation = pd.read_csv('data/interactions_validation.csv')
    interactions_test = pd.read_csv('data/interactions_test.csv')
    recipes = pd.read_csv('data/PP_recipes.csv')
    ingr_map = pd.read_pickle("data/ingr_map.pkl")
    
    return interactions_train, interactions_validation, interactions_test, recipes, ingr_map

