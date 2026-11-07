# ⚽ BETALYTIX
> **Team Performance-Based Football Match Outcome Prediction System**

BETALYTIX is an end-to-end machine learning project designed to predict football match outcomes (Home Win / Draw / Away Win) strictly using historical team performance data. Instead of copying market betting odds, this data engine builds true predictive capability to identify "Value Bets" through advanced feature engineering and XGBoost optimization.

## 🛠 Tech Stack
* **Database & Data Exploration:** SQL
* **Data Processing & Engineering:** Python, Pandas
* **Machine Learning:** Scikit-learn, XGBoost
* **Productization & UI:** Streamlit

## 🧠 Methodology & Pipeline
The project follows a rigorous 7-phase machine learning pipeline to ensure real-world reliability and prevent data leakage:
1. **Data Understanding:** SQL-based exploration.
2. **Data Cleaning:** Chronological sorting and type formatting.
3. **Feature Engineering:** Creation of 20+ analytical variables.
4. **Baseline Modeling:** Logistic Regression & Random Forest.
5. **Advanced Modeling:** Hyperparameter tuning with XGBoost.
6. **Business Interpretation:** Model probabilities vs. Market odds.
7. **Productization:** Interactive Streamlit dashboard.

**Key Methodological Rules:**
* **Strict Chronological Split:** Train/Test splits are strictly time-based to prevent future data leakage.
* **Isolated Feature Engineering:** Built using `groupby` and `shift` operations to ensure models only see past data at any given match week.
* **Pure Performance Focus:** Zero market odds were used as input features, forcing the algorithm to learn purely from offensive/defensive metrics.

## 📊 Feature Selection
To eliminate noise, correlation analyses and feature importance tests were conducted to select the **Top-20 Features**. Key feature categories include:
* **Form:** 3, 5, and 10-match moving averages and goal trends.
* **Attack / Defense:** Shot accuracy, clean sheet ratios, and strength ratings.
* **Home Advantage:** Historical home/away win rates and goal differentials.
* **Advanced Metrics:** Temporal Decay (reducing the weight of older matches) and Dominance Score.

## 🚀 Model Performance
The target variable is a 3-class classification (Home, Draw, Away), meaning the random guessing baseline is **33.3%**.

| Model | Accuracy | F1 Score (Macro) |
| :--- | :--- | :--- |
| Logistic Regression | 0.536 | 0.44 |
| Random Forest | 0.533 | 0.42 |
| **XGBoost (Final)** | **0.562** | **0.46** |

* **Insight:** Logistic Regression provided a strong baseline. Random Forest struggled with overfitting on the test set. XGBoost emerged as the leader by successfully capturing complex, non-linear team interactions through gradient boosting.

## 💡 Real-World Business Value (Value Bet Detection)
The system is designed to compare its own calculated probabilities against market odds to find mathematical advantages. 
* *Example (Galatasaray vs. Fenerbahçe):* The model heavily weighted Galatasaray's Home Form and Dominance Score, assigning a **58%** win probability compared to the market's **49%**—successfully detecting a **+9 point "Value"** margin.

## 💻 Running the App Locally
The project includes a ready-to-use Streamlit dashboard that visualizes team forms, feature importance, and live match predictions.

```bash
# Clone the repository
git clone [https://github.com/yourusername/betalytix.git](https://github.com/yourusername/betalytix.git)

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/app.py
