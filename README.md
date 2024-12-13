# TeamShallotML4VA

# Personalized Recipe Recommendation System

This repository contains the implementation of a personalized recipe recommendation system. The system uses collaborative filtering and content-based filtering techniques to suggest recipes tailored to user preferences, including calorie ranges, macronutrient goals, dietary tags, and ingredient preferences.

## Features
- Personalized recipe recommendations based on user input.
- Flexible filtering for calories, macronutrients, cooking time, and dietary preferences.
- Integration of USDA FoodData Central and Food.com datasets.
- A streamlined and user-friendly interface built with Streamlit.

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or later
- `pip` package manager

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/coletted1/TeamShallotML4VA.git
   ```
2. Navigate to the project directory:
   ```bash
   cd TeamShallotML4VA/my_app
   ```

## Running the Application
1. Navigate to the `my_app` directory:
   ```bash
   cd my_app
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the app in your browser. By default, Streamlit runs on `http://localhost:8501`.

## Usage
1. **Input User Preferences**:
   - Set calorie, macronutrient, and cooking time ranges.
   - Choose dietary tags (e.g., breakfast, lunch, dinner, snacks, desserts).
   - Specify ingredients to include or exclude.

2. **Get Recommendations**:
   - Click the "Get Recommendations" button to receive personalized recipe suggestions.

3. **View Results**:
   - Explore recipes with detailed nutritional information, cooking time, and a link to the full recipe on Food.com.

## Retraining the Model
1. Enable the "Retrain Model" checkbox in the Streamlit sidebar.
2. The system will retrain the recommendation model and save it as `svd_model.pkl`.

## Project Structure
```
TeamShallotML4VA/
├── data/                    # Contains datasets for recipes and user interactions
├── models/                  # Directory for saved models (e.g., svd_model.pkl)
├── my_app/                  # Main application directory
│   ├── app.py               # Streamlit application
│   ├── utils/               # Helper modules for data loading and recommendation
│   │   ├── data_loader.py   # Handles dataset loading
│   │   ├── recommendation.py # Recommendation logic and model training
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

## Dataset Sources
- **USDA FoodData Central**: Nutrition data for individual ingredients.
- **Food.com (Kaggle)**: Recipe and user interaction data.

## Contribution
Contributions are welcome! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or feedback, please reach out to:
- **Colette D'Costa** (Team Lead) - [Your Email/Contact Info]

---

Happy Cooking! 🍽️
```
