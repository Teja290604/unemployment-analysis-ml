# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

# ===============================
# 2. Load Dataset
# ===============================
df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# ===============================
# 3. Data Cleaning
# ===============================
# Fix column names (remove spaces)
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Merge datasets
df = pd.concat([df1, df2], ignore_index=True)
print("Initial Shape:", df.shape)
print(df.head())
# Rename columns for consistency
df.rename(columns={
    "Region": "State",
    "Date": "Date",
    "Frequency": "Frequency",
    "Estimated Unemployment Rate (%)": "Unemployment_Rate",
    "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "Labour_Participation_Rate",
    "Area": "Area"
}, inplace=True)

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')

# Drop missing values
df = df.dropna(subset=['Unemployment_Rate'])

# Remove duplicates
df.drop_duplicates(inplace=True)

# ===============================
# 4. Feature Engineering
# ===============================
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Encode categorical columns
le = LabelEncoder()

# Encode ALL object (string) columns automatically
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
# Encode categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# ✅ ADD HERE
df = df.drop(columns=['longitude', 'latitude', 'Region.1'], errors='ignore')

# Drop unnecessary columns
df = df.drop(columns=['Date', 'Frequency'], errors='ignore')
df.fillna(df.mean(numeric_only=True), inplace=True)
# Drop unnecessary columns
df = df.drop(columns=['Date', 'Frequency'], errors='ignore')
# ===============================
# 5. Exploratory Data Analysis
# ===============================
plt.figure(figsize=(10,5))
sns.lineplot(x='Month', y='Unemployment_Rate', data=df)
plt.title("Unemployment Trend by Month")
plt.show()

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ===============================
# 6. Define Features & Target
# ===============================
X = df.drop('Unemployment_Rate', axis=1)
y = df['Unemployment_Rate']

# Train-test split (time-based better, but simple split here)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 7. Model 1: Linear Regression
# ===============================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ===============================
# 8. Model 2: Random Forest + Tuning
# ===============================
rf = RandomForestRegressor()

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='r2')
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# ===============================
# 9. Model 3: XGBoost + Tuning
# ===============================
xgb = XGBRegressor()

param_grid_xgb = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6],
    'n_estimators': [100, 200]
}

grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3, scoring='r2')
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)

# ===============================
# 10. Evaluation Function
# ===============================
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Performance:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score:", r2_score(y_test, y_pred))

# ===============================
# 11. Evaluate All Models
# ===============================
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)

# ===============================
# 12. Feature Importance (Best Model)
# ===============================
importances = best_xgb.feature_importances_
features = X.columns

plt.figure(figsize=(10,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (XGBoost)")
plt.show()

# ===============================
# 13. Save Model
# ===============================
import joblib
joblib.dump(best_xgb, "unemployment_model.pkl")

print("\nModel saved successfully!")