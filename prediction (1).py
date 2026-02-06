import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (download from Kaggle: House Prices - Advanced Regression Techniques)
data = pd.read_csv("train.csv")

# Select features based on your question
features = ["BedroomAbvGr", "TotRmsAbvGrd", "Neighborhood", "LotArea", "GrLivArea", "TotalBsmtSF"]
X = data[features]
y = data["SalePrice"]

# Separate categorical and numerical features
categorical = ["Neighborhood"]
numerical = ["BedroomAbvGr", "TotRmsAbvGrd", "LotArea", "GrLivArea", "TotalBsmtSF"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ])

# Build pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Example prediction
sample_house = pd.DataFrame([[3, 7, "CollgCr", 5000, 1800, 800]],
                            columns=features)
print("Predicted Price:", model.predict(sample_house)[0])
