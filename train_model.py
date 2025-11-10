import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


# Load dataset

df = pd.read_csv("Carbon Emission.csv")

print("🔹 Dataset Loaded Successfully")
print(df.head(), "\n")
print(df.info())


# Define features and target

X = df.drop(columns=['CarbonEmission'])
y = df['CarbonEmission']

# Identify categorical & numeric columns

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print("\nCategorical Columns:", categorical_cols)
print("Numeric Columns:", numeric_cols)


# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep numeric columns
)

# Build Pipeline

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        max_depth=15,
        min_samples_split=4,
        n_jobs=-1
    ))
])


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Model

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n✅ Model trained successfully!")


# Evaluation Metrics

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📊 Model Evaluation Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# Visualization

# --- 1. Actual vs Predicted Plot ---
plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test, y=y_pred, color='green', alpha=0.7)
plt.xlabel("Actual Emission")
plt.ylabel("Predicted Emission")
plt.title("Actual vs Predicted Carbon Emission")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# --- 2. Residual Plot ---
residuals = y_test - y_pred
plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True, color='purple')
plt.title("Distribution of Residuals (Error)")
plt.xlabel("Residuals")
plt.show()

# --- 3. Feature Importance ---
rf_model = model.named_steps['regressor']
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
importance = pd.Series(rf_model.feature_importances_, index=feature_names)
top_features = importance.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features, y=top_features.index, palette="viridis")
plt.title("Top 15 Important Features Affecting Carbon Emission")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

#Save Trained Model
joblib.dump(model, "carbon_model.pkl")
print("\n💾 Model saved as carbon_model.pkl")
