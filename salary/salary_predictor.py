import sys
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Load data ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "salary_data.csv"))

X = df[["YearsExperience"]]
y = df["Salary"]

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train Linear Regression model ─────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Model performance ──────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)

print("=" * 45)
print("       Salary Prediction Model Ready")
print("=" * 45)
print(f"  Model  : Linear Regression")
print(f"  R² Score : {r2:.4f}  (1.0 = perfect fit)")
print(f"  Mean Absolute Error : ${mae:,.2f}")
print(f"  Formula : Salary = {model.coef_[0]:,.2f} × Years + {model.intercept_:,.2f}")
print("=" * 45)

# ── Interactive prediction loop ────────────────────────────────────────────────
print("\nEnter years of experience to get predicted salary.")
print("Type 'quit' to exit.\n")

while True:
    try:
        user_input = input("Years of Experience: ").strip()
    except EOFError:
        print("\nGoodbye!")
        break

    if user_input.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    try:
        years = float(user_input)
        if years < 0:
            print("  ⚠  Please enter a non-negative number.\n")
            continue

        predicted = model.predict([[years]])[0]
        predicted = max(predicted, 0)          # salary can't be negative
        print(f"  >> Predicted Salary: ${predicted:,.2f}\n")

    except ValueError:
        print("  !! Invalid input. Please enter a numeric value (e.g. 3.5).\n")
