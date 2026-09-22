# 📈 Stock Market Intelligence

An end-to-end stock market analysis and machine learning project combining **financial data analysis, risk & performance analytics, machine learning, live market data, and an interactive Streamlit dashboard**.

The project analyzes five major technology companies and includes a machine learning system for predicting the **5-trading-day direction of Apple (AAPL)** using historical price-based features.

---

## 🚀 Project Overview

This project was developed as a complete data analysis and machine learning workflow.

It covers:

- 📊 Historical stock market analysis
- 🧹 Data preparation and cleaning
- 🔎 Exploratory Data Analysis (EDA)
- 📉 Risk analysis
- 📈 Performance analysis
- 🔗 Correlation analysis
- 📊 Interactive visualizations
- 🤖 Machine Learning classification
- ⏳ Time-series validation
- ⚙️ Hyperparameter tuning
- 🍎 AAPL 5-day direction prediction
- 🌐 Live market data using `yfinance`
- 🖥️ Unified Streamlit GUI

### Stocks Analyzed

- **AAPL** — Apple
- **MSFT** — Microsoft
- **NVDA** — NVIDIA
- **AMZN** — Amazon
- **TSLA** — Tesla

---

# 🎯 Project Objectives

The main objectives of the project are:

1. Analyze historical stock prices.
2. Calculate daily returns and statistical measures.
3. Compare stock performance.
4. Analyze historical risk and volatility.
5. Study correlations between stocks.
6. Build interactive financial visualizations.
7. Develop a machine learning model for AAPL direction prediction.
8. Evaluate multiple classification algorithms.
9. Integrate live market data.
10. Build a unified interactive Streamlit application.

---

# 🗂️ Dataset

The project uses historical stock price data for:

- Apple (`AAPL`)
- Microsoft (`MSFT`)
- NVIDIA (`NVDA`)
- Amazon (`AMZN`)
- Tesla (`TSLA`)

The main historical analysis dataset contains the closing prices of the five stocks.

The machine learning component focuses specifically on **AAPL**.

---

# Part 1 — Data Processing

The data preparation workflow included:

- Loading the individual stock datasets.
- Converting the `Date` column to datetime format.
- Sorting observations chronologically.
- Checking for missing values.
- Combining stock closing prices.
- Preparing the dataset for financial analysis.
- Preparing the dataset for machine learning.

The main combined dataset contains:

- `Date`
- `AAPL`
- `MSFT`
- `NVDA`
- `AMZN`
- `TSLA`

---

# Part 2 — Exploratory Data Analysis

The exploratory analysis focused on understanding the historical movement and behavior of stock prices.

The analysis included:

- Historical closing price analysis.
- Daily returns calculation.
- Distribution analysis of daily returns.
- Mean and standard deviation analysis.
- Kurtosis analysis.
- Comparison of normalized stock prices.

## Daily Returns

Daily returns were calculated using percentage change:

```python
Daily_Returns = df.pct_change()
```

Daily returns measure the percentage change in a stock's price from one trading day to the next.

## Normalization

Stock prices were normalized using:

```python
norm = df / df.iloc[0, :]
```

Normalization makes it easier to compare the relative performance of different stocks because all stocks start from the same base value.

---

# Part 3 — Risk Analysis

Several metrics were used to evaluate the historical risk characteristics of the stocks.

## Volatility

Annualized volatility measures the variability of stock returns.

It was calculated using:

```python
annual_volatility = daily_returns.std() * np.sqrt(252)
```

Higher volatility indicates larger fluctuations in historical returns.

## Kurtosis

Kurtosis was used to analyze the distribution of daily returns and identify the presence of extreme return observations.

Higher kurtosis can indicate heavier tails in the return distribution.

## Maximum Drawdown

Maximum Drawdown measures the largest decline from a previous historical peak.

It was calculated using the running maximum of stock prices.

A more negative Maximum Drawdown represents a larger historical decline from a previous peak.

---

# Part 4 — Performance Analysis

## Cumulative Return

Cumulative Return measures the total historical price return over the selected analysis period.

It was calculated using:

```python
cumulative_return = (
    (df.iloc[-1] / df.iloc[0]) - 1
) * 100
```

## Annualized Return

Annualized Return estimates the yearly return based on the average daily return.

```python
annual_return = daily_returns.mean() * 252
```

## Sharpe Ratio

The Sharpe Ratio measures historical return relative to return variability.

It was calculated using:

```python
sharpe_ratio = annual_return / (
    daily_returns.std() * np.sqrt(252)
)
```

A higher Sharpe Ratio indicates a higher historical return relative to the measured volatility under this calculation.

## Risk vs Return

A Risk vs Return analysis was performed using:

- Annualized Return
- Annualized Volatility

This visualization allows users to examine the relationship between historical return and risk.

---

# Part 5 — Correlation Analysis

A correlation matrix was calculated using daily stock returns.

Correlation analysis helps identify how the returns of different stocks moved relative to each other during the historical period.

The correlation matrix was visualized using a heatmap.

---

# Part 6 — Data Visualizations

The project includes several visualizations:

- Daily Return Distribution
- Normalized Stock Performance
- Cumulative Return Comparison
- Sharpe Ratio Comparison
- Risk vs Return Analysis
- Correlation Heatmap
- Maximum Drawdown Comparison
- Performance Summary Table

---

# Part 7 — Machine Learning

## Machine Learning Objective

The machine learning component focuses on predicting the future direction of Apple's stock price.

The selected target stock is:

```text
AAPL
```

The prediction task is a binary classification problem:

```text
0 → DOWN
1 → UP
```

The final ML experiment predicts whether the AAPL closing price will be higher or lower after the next **5 trading days**.

---

# 🎯 Target Definition

The 5-day target was created by comparing the future AAPL price with the current AAPL price:

```python
df["AAPL_Target_5D"] = (
    df["AAPL"].shift(-5) > df["AAPL"]
).astype(int)
```

Therefore:

- `1` → AAPL price is higher after 5 trading days.
- `0` → AAPL price is lower or not higher after 5 trading days.

The resulting 5-day target contained:

- **405 UP observations**
- **331 DOWN observations**

---

# 🛠️ Feature Engineering

Several features were created from AAPL historical price data.

The final model uses 8 features:

```text
AAPL
AAPL_Return
AAPL_MA_5
AAPL_MA_10
AAPL_Lag_1
AAPL_Lag_2
AAPL_Lag_3
AAPL_Volatility_5
```

## Feature Description

### AAPL

The current AAPL closing price.

### AAPL_Return

The daily percentage return:

```python
AAPL_Return = AAPL.pct_change()
```

### AAPL_MA_5

5-day moving average.

### AAPL_MA_10

10-day moving average.

Moving averages provide information about recent price trends.

### AAPL_Lag_1

Previous trading day's closing price.

### AAPL_Lag_2

Closing price from two trading days earlier.

### AAPL_Lag_3

Closing price from three trading days earlier.

### AAPL_Volatility_5

5-day rolling standard deviation of daily returns.

It represents short-term historical volatility.

---

# 🤖 Machine Learning Models

Three classification algorithms were evaluated:

## 1. Logistic Regression

Logistic Regression was used as a baseline linear classification model.

The implementation used a pipeline containing:

```text
StandardScaler
+
Logistic Regression
```

## 2. Random Forest

Random Forest was used as a tree-based ensemble classification model.

The model combines multiple decision trees to produce a classification result.

## 3. XGBoost

XGBoost was used as a gradient boosting classification algorithm.

It builds an ensemble of decision trees sequentially to improve predictive performance.

---

# ⏳ Time Series Validation

Because stock market data is time-dependent, a random train/test split was avoided.

Instead, the project used:

```text
TimeSeriesSplit
```

This preserves the chronological order of the observations.

The models were trained using historical observations and validated on later observations.

---

# ⚙️ Hyperparameter Tuning

Grid Search was used to test different model configurations.

The optimization metric was:

```text
Balanced Accuracy
```

This was selected because the target classes were not perfectly balanced.

Using balanced accuracy also prevents the evaluation from relying only on overall accuracy when the model behaves differently across the two classes.

---

# 📊 Model Evaluation

Several classification metrics were used.

## Accuracy

Accuracy measures the percentage of all predictions that were correct.

```text
Accuracy =
(TP + TN) / (TP + TN + FP + FN)
```

## Precision

Precision measures how many of the observations predicted as a specific class were actually members of that class.

For the UP class:

```text
Precision =
TP / (TP + FP)
```

## Recall

Recall measures how many of the actual observations belonging to a class were successfully identified by the model.

For the UP class:

```text
Recall =
TP / (TP + FN)
```

## F1 Score

F1 Score combines Precision and Recall into a single metric.

```text
F1 =
2 × (Precision × Recall)
/
(Precision + Recall)
```

## Balanced Accuracy

Balanced Accuracy calculates the average recall across the two classes.

```text
Balanced Accuracy =
(Recall of Class 0 + Recall of Class 1) / 2
```

This gives both classes equal importance.

## Macro F1

Macro F1 calculates the F1 score independently for each class and then takes the average.

This gives both classes equal weight.

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two classes across different classification thresholds.

A value around:

```text
0.50
```

represents performance close to random classification.

A value closer to:

```text
1.00
```

indicates stronger class separation.

---

# 📈 Machine Learning Results

The models were evaluated on a held-out test period after training and hyperparameter tuning.

| Model | CV Balanced Accuracy | Test Accuracy | Test Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 57.63% | 56.76% | 62.62% | 0.556 |
| Random Forest | 53.11% | 54.05% | 56.66% | 0.540 |
| XGBoost | 55.73% | 54.73% | 53.22% | 0.532 |

The Logistic Regression model achieved the highest **test Balanced Accuracy** among the tested models in the 5-day experiment.

Its ROC-AUC on the held-out test period was:

```text
0.622
```

These results indicate that the model captured some signal in this dataset, but the predictive separation remained limited.

The results should therefore be interpreted as experimental results for this dataset and test period rather than as a guarantee of future stock market performance.

---

# 📅 1-Day vs 5-Day Prediction

Two prediction horizons were experimentally evaluated.

| Prediction Horizon | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| 1 Trading Day | 53.69% | 53.40% | 0.534 | 0.503 |
| 5 Trading Days | 56.76% | 62.62% | 0.556 | 0.622 |

The 5-day experiment produced stronger held-out metrics in this particular dataset and experimental setup.

However, this should not be interpreted as evidence that a 5-day strategy will consistently outperform a 1-day strategy in future market conditions.

---

# 🏆 Final Machine Learning Model

The final prediction system uses:

```text
Logistic Regression
```

for the 5-day AAPL direction prediction.

The trained model is stored as:

```text
aapl_5day_model.pkl
```

The model was selected based on the experimental evaluation performed on the historical dataset.

---

# 🌐 Part 8 — Live Machine Learning Prediction

The machine learning component was integrated into the Streamlit application.

Instead of using only the historical static dataset, the ML prediction interface retrieves the latest available AAPL market data using:

```text
yfinance
```

The application automatically:

1. Retrieves the latest AAPL market data.
2. Detects the latest available trading date.
3. Gets the latest closing price.
4. Calculates the required ML features.
5. Creates the latest feature row.
6. Sends the features to the trained model.
7. Generates the 5-day prediction.
8. Displays UP/DOWN probabilities.

---

# 📌 Live Prediction Features

The application dynamically calculates:

```text
AAPL
AAPL Return
5-Day Moving Average
10-Day Moving Average
Lag 1
Lag 2
Lag 3
5-Day Volatility
```

This allows the prediction to use the latest available market information rather than relying on a fixed feature row.

---

# 🖥️ Part 9 — Unified Streamlit Application

The project was integrated into a single Streamlit application.

The application contains two main sections:

```text
📊 Market Dashboard
🤖 ML Prediction
```

The application uses a custom dark-themed interface with interactive navigation.

---

## 📊 Market Dashboard

The Market Dashboard provides:

- Stock selection
- Date range filtering
- Historical performance analysis
- Normalized stock performance
- Cumulative returns
- Sharpe Ratio
- Risk vs Return
- Correlation Heatmap
- Maximum Drawdown
- Performance Summary Table
- Key Performance Indicators

---

## 🤖 ML Prediction

The ML Prediction section provides:

- Latest AAPL price
- Latest available market date
- 5-day direction prediction
- UP probability
- DOWN probability
- Prediction horizon
- Machine learning model information

Example interface output:

```text
AAPL 5-Day Prediction

Prediction: DOWN

DOWN Probability: 56.62%
UP Probability: 43.38%

Forecast Horizon:
Next 5 Trading Days
```

The displayed prediction is generated dynamically from the latest available market data.

---

# 🎨 Interactive GUI

The final application provides a single user interface combining financial analysis and machine learning.

Users can navigate between:

```text
📊 Market Dashboard
🤖 ML Prediction
```

The interface was developed using:

```text
Streamlit
```

The application also uses live market data through:

```text
yfinance
```

---

## 🚀 Live Demo

The application is deployed using **Streamlit Community Cloud**.

## 🌐 Live Application

[Open Stock Market Intelligence Dashboard](https://abdulrahman-stock-market-intelligence.streamlit.app/)

The application provides:

- 📊 Interactive stock market analysis
- 📈 Performance and risk analytics
- 🔗 Correlation analysis
- 🤖 AAPL 5-Day ML Prediction
- 🌐 Live market data using `yfinance`
- 🖥️ Unified interactive Streamlit interface
---

# 📁 Project Structure

```text
stock-market-intelligence/
│
├── AAPL.csv
├── AMZN.csv
├── MSFT.csv
├── NVDA.csv
├── TSLA.csv
│
├── combined_close_prices.csv
│
├── project.ipynb
├── pred.ipynb
│
├── app.py
│
├── aapl_5day_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📄 Files Description

## Data

- `AAPL.csv` — Apple historical stock data.
- `AMZN.csv` — Amazon historical stock data.
- `MSFT.csv` — Microsoft historical stock data.
- `NVDA.csv` — NVIDIA historical stock data.
- `TSLA.csv` — Tesla historical stock data.
- `combined_close_prices.csv` — Combined closing prices used for financial analysis.

## Analysis

- `project.ipynb` — Main Jupyter Notebook containing data processing, exploratory analysis, risk analysis, performance analysis, and visualizations.
- `pred.ipynb` — Jupyter Notebook containing the machine learning workflow, feature engineering, model training, validation, tuning, and evaluation.

## Application

- `app.py` — Unified Streamlit application containing both the Market Dashboard and ML Prediction interface.

## Machine Learning

- `aapl_5day_model.pkl` — Trained Logistic Regression model used for AAPL 5-day direction prediction.

## Configuration

- `requirements.txt` — Required Python libraries.
- `README.md` — Project documentation.
- `.gitignore` — Git ignore configuration.

---

# 🛠️ Technologies Used

## Data Analysis

- Python
- Pandas
- NumPy

## Data Visualization

- Matplotlib
- Seaborn
- Plotly

## Machine Learning

- Scikit-learn
- XGBoost

## Application

- Streamlit
- yfinance

## Model Persistence

- Joblib

## Development Environment

- Jupyter Notebook
- Visual Studio Code
- Python

## Version Control

- Git
- GitHub

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/abdulrahman2005-prog/stock-market-intelligence.git
```

## 2. Navigate to the Project Directory

```bash
cd stock-market-intelligence
```

## 3. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

## 4. Install the Required Libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the unified Streamlit application:

```bash
streamlit run app.py
```

The application will open automatically in your web browser.

If it does not open automatically, Streamlit will provide a local URL in the terminal.

---

# 🤖 Machine Learning Workflow

The complete machine learning workflow can be summarized as:

```text
Historical Stock Data
        ↓
Data Preparation
        ↓
Feature Engineering
        ↓
Target Creation
        ↓
Time-Based Train/Test Split
        ↓
TimeSeriesSplit Cross Validation
        ↓
Hyperparameter Tuning
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Final Logistic Regression Model
        ↓
Live Market Data
        ↓
Feature Generation
        ↓
5-Day AAPL Prediction
        ↓
Streamlit GUI
```

---

# 🔄 Complete Project Workflow

```text
Raw Stock Data
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Risk & Performance Analysis
       ↓
Interactive Market Dashboard
       ↓
Machine Learning Feature Engineering
       ↓
Model Training & Evaluation
       ↓
AAPL 5-Day Prediction
       ↓
Live Market Data Integration
       ↓
Unified Streamlit Application
```

---

# 🔬 Machine Learning Notes

The machine learning experiment uses chronological data splitting to reduce the risk of training on future observations.

The model evaluation is based on a held-out historical test period.

The project uses multiple metrics because accuracy alone may not fully describe classification performance.

In particular, **Balanced Accuracy and Macro F1** are included to evaluate both classes more equally.

The live prediction system should be considered an experimental analytical component rather than a guaranteed forecasting system.

---

# ⚠️ Important Limitations

This project has several limitations:

- The model uses a relatively small set of historical price-based features.
- Stock prices are affected by many external factors that are not included in the model.
- Historical performance does not guarantee future performance.
- The model was evaluated on one historical held-out period.
- Market conditions can change over time.
- Live predictions may change as new market data becomes available.
- Repeated experimentation on the same historical dataset can introduce selection bias.

Therefore, the reported ML metrics should be interpreted as results for this specific dataset and experimental setup.

---

# 📌 Disclaimer

This project was developed for **educational and analytical purposes only**.

The financial analysis, machine learning predictions, probabilities, visualizations, and other outputs should **not** be considered financial or investment advice.

Machine learning predictions are based on historical data and the features used by the model. Actual future market behavior can differ significantly from model predictions.

The project should not be used as the sole basis for financial or investment decisions.

---

# 👨‍💻 Author

**Abdulrahman**

GitHub Repository:

https://github.com/abdulrahman2005-prog/stock-market-intelligence

---

# ⭐ Project Summary

This project combines:

```text
Data Analysis
      +
Financial Analytics
      +
Risk Analysis
      +
Data Visualization
      +
Machine Learning
      +
Live Market Data
      +
Streamlit
```

into one complete **Stock Market Intelligence** project.

The final application provides an interactive environment for exploring historical stock performance and experimenting with machine learning-based AAPL 5-day direction predictions.