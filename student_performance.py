
import os

import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# Paths are built relative to this file, so the project works on any computer.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Maths.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# The column we want to predict. The script tries these names in order.
# If your dataset uses a different name, add it to this list.
TARGET_CANDIDATES = ["math score", "math_score", "G3", "Math", "maths", "final_score"]

RANDOM_STATE = 42  # makes results repeatable
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def find_target_column(df):
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    # Fallback: use the last numeric column and tell the user.
    numeric_cols = df.select_dtypes(include="number").columns
    print(f"Could not find a known target name, using '{numeric_cols[-1]}' instead.")
    return numeric_cols[-1]



print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

# sep=None lets pandas detect whether the file uses commas or semicolons.
df = pd.read_csv(DATA_PATH, sep=None, engine="python")
target = find_target_column(df)

print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Target column (what we predict): {target}\n")
print(df.head(), "\n")


print("=" * 60)
print("STEP 2: Exploring the data")
print("=" * 60)

print("Missing values per column:")
print(df.isnull().sum(), "\n")
print("Summary statistics:")
print(df.describe(), "\n")

# Remove duplicate rows and rows with missing values (simple cleaning).
df = df.drop_duplicates().dropna()

numeric_cols = df.select_dtypes(include="number").columns.tolist()
category_cols = df.select_dtypes(exclude="number").columns.tolist()

# Chart 1: how are the scores distributed?
plt.figure(figsize=(7, 4))
sns.histplot(df[target], bins=20, kde=True, color="steelblue")
plt.title(f"Distribution of {target}")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_score_distribution.png"), dpi=150)
plt.close()

#  average score for each group of the first two text columns
# (for example gender or parental education).
for i, col in enumerate(category_cols[:2], start=1):
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=col, y=target, color="lightcoral")
    plt.title(f"{target} by {col}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"2_{i}_score_by_{i}.png"), dpi=150)
    plt.close()

# Chart 3: correlation heatmap (which numeric columns move together?)
if len(numeric_cols) > 1:
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation between numeric columns")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_correlation_heatmap.png"), dpi=150)
    plt.close()

print(f"Charts saved in: {OUTPUT_DIR}\n")


print("=" * 60)
print("STEP 3: Preparing data")
print("=" * 60)

# X = inputs (features), y = the answer we want to predict.
y = df[target]
X = df.drop(columns=[target])

# Models only understand numbers, so convert text columns into 0/1 columns
# (this is called one-hot encoding).
X = pd.get_dummies(X, drop_first=True)

# Keep 80% of the data for training and 20% for testing the models.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}\n")


print("=" * 60)
print("STEP 4: Training models")
print("=" * 60)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
}

results = []
trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)          # learn from the training data
    predictions = model.predict(X_test)  # predict on unseen data
    r2 = r2_score(y_test, predictions)   # 1.0 = perfect, 0 = no better than average
    mae = mean_absolute_error(y_test, predictions)  # average error in score points
    results.append({"Model": name, "R2": round(r2, 3), "MAE": round(mae, 2)})
    trained[name] = model

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False), "\n")
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"), index=False)


print("=" * 60)
print("STEP 5: Feature importance (Random Forest)")
print("=" * 60)

importances = pd.Series(
    trained["Random Forest"].feature_importances_, index=X.columns
).sort_values(ascending=False)
print(importances.head(10).round(3), "\n")

plt.figure(figsize=(8, 5))
importances.head(10).sort_values().plot(kind="barh", color="seagreen")
plt.title("Top 10 features that influence the score")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "4_feature_importance.png"), dpi=150)
plt.close()

print("Done! Check the 'outputs' folder for charts and results.")
