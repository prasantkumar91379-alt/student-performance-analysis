# 🎓 Student Performance Analysis

A beginner-friendly data science and machine learning project that explores what influences students' maths scores and trains simple models to predict them.

## 📌 Project Overview
The goal is to answer two questions:
1. Which factors (gender, parental education, test preparation, etc.) are related to student performance?
2. Can a machine learning model predict a student's maths score from those factors?

## 📂 Project Structure
```
student-performance-analysis/
├── data/                      # dataset (Maths.csv)
├── outputs/                   # generated charts and results
├── student_performance.py     # main script (fully commented)
├── requirements.txt           # Python libraries needed
├── .gitignore
└── README.md
```

## 📊 Dataset
Kaggle: **Students Performance in Exams**. Each row is a student, with details such as gender, parental level of education, lunch type, test preparation course, and exam scores.

Download the CSV from Kaggle and save it as `data/Maths.csv`.

## ⚙️ Tools Used
Python, pandas, matplotlib, seaborn, scikit-learn

## 🔍 What the Project Does
1. **Loads and cleans** the data (removes duplicates and missing values)
2. **Explores** the data (EDA) with summary statistics and charts
3. **Prepares** the data (one-hot encoding, 80/20 train-test split)
4. **Trains** Linear Regression and Random Forest models
5. **Evaluates** them with R² and MAE
6. **Shows** which features matter most

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/student-performance-analysis.git
cd student-performance-analysis

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install libraries
pip install -r requirements.txt

# 4. Add the dataset as data/Maths.csv, then run
python student_performance.py
```

## 📈 Results
After running, the `outputs/` folder contains:
- `1_score_distribution.png`: how scores are spread
- `2_*_score_by_*.png`: scores compared across groups
- `3_correlation_heatmap.png`: relationships between numeric columns
- `4_feature_importance.png`: top features affecting the score
- `model_results.csv`: R² and MAE for each model

> After you run it, paste your own charts and model scores here. For example:
> `![Feature importance](outputs/4_feature_importance.png)`

| Model | R² | MAE |
|-------|----|-----|
| Linear Regression | _your result_ | _your result_ |
| Random Forest | _your result_ | _your result_ |

**R²** shows how much of the score variation the model explains (closer to 1 is better). **MAE** is the average prediction error in marks (lower is better).

## 💡 Ideas for Improvement
- Try more models (Decision Tree, Gradient Boosting)
- Tune hyperparameters with GridSearchCV
- Predict pass/fail (classification) instead of the exact score
- Build a small web app with Streamlit

## 📄 License
Free to use for learning purposes.
