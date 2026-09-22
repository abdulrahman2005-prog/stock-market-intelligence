# Stock Market Analysis & ML Prediction

## Project Overview

This project provides a complete analysis and machine learning workflow for historical stock market data of five major technology companies:

- Apple (AAPL)
- Microsoft (MSFT)
- NVIDIA (NVDA)
- Amazon (AMZN)
- Tesla (TSLA)

The project combines:

- Data Cleaning & Preparation
- Exploratory Data Analysis
- Financial Performance Analysis
- Risk Analysis
- Interactive Data Visualization
- Machine Learning
- Stock Direction Prediction
- Interactive Streamlit Dashboard

The final application provides a unified interface where users can explore historical market performance and access a machine learning prediction for Apple's (AAPL) stock direction over the next 5 trading days.

---

## Project Objectives

The main objectives of this project are:

- Clean and prepare historical stock price data.
- Analyze stock closing prices.
- Compare the historical performance of multiple companies.
- Calculate daily returns.
- Measure investment risk using volatility and return distributions.
- Analyze relationships between stock returns.
- Calculate cumulative returns.
- Evaluate risk-adjusted performance using the Sharpe Ratio.
- Measure downside risk using Maximum Drawdown.
- Build an interactive financial analysis dashboard.
- Develop a machine learning model for stock direction prediction.
- Compare different machine learning algorithms.
- Evaluate the models using appropriate classification metrics.
- Integrate the machine learning model into an interactive Streamlit application.

---

# Part 1 — Data Processing

## Data Sources

Historical stock price data was collected for:

- AAPL
- MSFT
- NVDA
- AMZN
- TSLA

The datasets were processed and cleaned before performing the analysis.

## Data Cleaning

The data processing steps included:

- Removing unnecessary rows.
- Renaming columns.
- Converting the Date column to datetime format.
- Converting numerical columns to numeric data types.
- Checking for missing values.
- Removing duplicate records.
- Sorting the data chronologically.
- Combining the closing prices of all companies into a single dataset.

The final combined dataset is:

combined_close_prices.csv

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

Daily_Returns = df.pct_change()

Daily returns measure the percentage change in a stock's price from one trading day to the next.

## Normalization

Stock prices were normalized using:

norm = df / df.iloc[0, :]

Normalization makes it easier to compare the relative performance of different stocks because all stocks start from the same base value.

---

# Part 3 — Risk Analysis

Several metrics were used to evaluate the historical risk characteristics of the stocks.

## Volatility

Annualized volatility measures the variability of stock returns.

It was calculated using:

annual_volatility = daily_returns.std() * np.sqrt(252)

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

cumulative_return = ((df.iloc[-1] / df.iloc[0]) - 1) * 100

## Annualized Return

Annualized Return estimates the yearly return based on the average daily return.

annual_return = daily_returns.mean() * 252

## Sharpe Ratio

The Sharpe Ratio measures historical return relative to return variability.

It was calculated using:

sharpe_ratio = annual_return / (daily_returns.std() * np.sqrt(252))

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

AAPL

The prediction task is a binary classification problem:

0 → DOWN
1 → UP

The model predicts whether the AAPL closing price will be higher or lower after the next 5 trading days.

## Target Definition

The 5-day target was created by comparing the future AAPL price with the current AAPL price:

df["AAPL_Target_5D"] = (
    df["AAPL"].shift(-5) > df["AAPL"]
).astype(int)

Therefore:

- 1 → AAPL price is higher after 5 trading days.
- 0 → AAPL price is lower or not higher after 5 trading days.

---

# Feature Engineering

Several features were created from AAPL historical price data.

The final model uses 8 features:

AAPL
AAPL_Return
AAPL_MA_5
AAPL_MA_10
AAPL_Lag_1
AAPL_Lag_2
AAPL_Lag_3
AAPL_Volatility_5

## Feature Description

### AAPL

The current AAPL closing price.

### AAPL_Return

The daily percentage return:

AAPL_Return = AAPL.pct_change()

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

# Machine Learning Models

Three classification algorithms were evaluated:

## 1. Logistic Regression

Logistic Regression was used as a baseline linear classification model.

It was implemented using a pipeline containing:

StandardScaler
+
Logistic Regression

## 2. Random Forest

Random Forest was used as a tree-based ensemble classification model.

The model combines multiple decision trees to produce a classification result.

## 3. XGBoost

XGBoost was used as a gradient boosting classification algorithm.

It builds an ensemble of decision trees sequentially to improve predictive performance.

---

# Time Series Validation

Because stock market data is time-dependent, a random train/test split was avoided.

Instead, the project used:

TimeSeriesSplit

This preserves the chronological order of the observations.

The models were trained using historical observations and validated on later observations.

---

# Hyperparameter Tuning

Grid Search was used to test different model configurations.

The optimization metric was:

Balanced Accuracy

This was selected because the target classes were not perfectly balanced.

---

# Model Evaluation

Several classification metrics were used.

## Accuracy

Accuracy measures the percentage of all predictions that were correct.

Accuracy = (TP + TN) / (TP + TN + FP + FN)

## Precision

Precision measures how many of the observations predicted as a specific class were actually members of that class.

For the UP class:

Precision = TP / (TP + FP)

## Recall

Recall measures how many of the actual observations belonging to a class were successfully identified by the model.

For the UP class:

Recall = TP / (TP + FN)

## F1 Score

F1 Score combines Precision and Recall into a single metric.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

## Balanced Accuracy

Balanced Accuracy calculates the average recall across the two classes.

Balanced Accuracy = (Recall of Class 0 + Recall of Class 1) / 2

This gives both classes equal importance.

## Macro F1

Macro F1 calculates the F1 score independently for each class and then takes the average.

This prevents the larger class from dominating the metric.

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between the two classes across different classification thresholds.

A value around 0.50 represents performance close to random classification.

A value closer to 1.00 indicates stronger class separation.

---

# Machine Learning Results

The models were evaluated on a held-out test period after training and hyperparameter tuning.

| Model | CV Balanced Accuracy | Test Accuracy | Test Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 57.63% | 56.76% | 62.62% | 0.556 |
| Random Forest | 53.11% | 54.05% | 56.66% | 0.540 |
| XGBoost | 55.73% | 54.73% | 53.22% | 0.532 |

The Logistic Regression model produced the highest test Balanced Accuracy among the tested models for the 5-day prediction experiment.

Its ROC-AUC on the held-out test period was:

0.622

The results indicate that the model captured some signal in this dataset, but the predictive separation remained limited.

These results should therefore be interpreted as experimental results for this dataset and test period rather than as a guarantee of future stock market performance.

---

# 1-Day vs 5-Day Prediction

Two prediction horizons were experimentally evaluated.

| Prediction Horizon | Accuracy | Balanced Accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| 1 Trading Day | 53.69% | 53.40% | 0.534 | 0.503 |
| 5 Trading Days | 56.76% | 62.62% | 0.556 | 0.622 |

The 5-day experiment produced stronger held-out metrics in this particular dataset and experimental setup.

However, this should not be interpreted as evidence that a 5-day strategy will consistently outperform a 1-day strategy in future market conditions.

---

# Final Machine Learning Model

The final prediction system uses the trained:

Logistic Regression

model for the 5-day AAPL direction prediction.

The trained model is stored as:

aapl_5day_model.pkl

---

# Part 8 — Live Machine Learning Prediction

The machine learning component was integrated into the Streamlit application.

Instead of using only the historical static dataset, the ML prediction interface retrieves the latest available AAPL market data using:

yfinance

The application automatically:

1. Retrieves the latest AAPL market data.
2. Detects the latest available trading date.
3. Gets the latest closing price.
4. Calculates the required ML features.
5. Creates the latest feature row.
6. Sends the features to the trained model.
7. Generates the 5-day prediction.
8. Displays UP/DOWN probabilities.

## Live Prediction Features

The application dynamically calculates:

AAPL
AAPL Return
5-Day Moving Average
10-Day Moving Average
Lag 1
Lag 2
Lag 3
5-Day Volatility

This allows the prediction to use the latest available market information rather than relying on a fixed feature row.

---

# Part 9 — Unified Streamlit Application

The project was integrated into a unified Streamlit application.

The application contains two main sections:

📊 Market Dashboard
🤖 ML Prediction

The application uses a custom dark-themed interface with interactive navigation.

## Market Dashboard

The Market Dashboard provides:

- Stock selection.
- Date range filtering.
- Historical performance analysis.
- Normalized stock performance.
- Cumulative returns.
- Sharpe Ratio.
- Risk vs Return.
- Correlation Heatmap.
- Maximum Drawdown.
- Performance Summary Table.
- Key Performance Indicators.

## ML Prediction

The ML Prediction section provides:

- Latest AAPL price.
- Latest available market date.
- 5-day direction prediction.
- UP probability.
- DOWN probability.
- Prediction horizon.
- Machine learning model information.

Example:

AAPL 5-Day Prediction

Prediction: DOWN

DOWN Probability: 56.62%
UP Probability: 43.38%

Forecast Horizon:
Next 5 Trading Days

The prediction is automatically generated using the latest available market data.

---

# Interactive GUI

The final application provides a single user interface combining financial analysis and machine learning.

Users can navigate between:

📊 Market Dashboard
🤖 ML Prediction

The interface was developed using:

Streamlit

The application also uses live market data through:

yfinance

---

# Live Demo

The final Streamlit application will be deployed using Streamlit Community Cloud.

## 🚀 Live Application

[Open Stock Market Intelligence Dashboard](YOUR_STREAMLIT_APP_LINK_HERE)

The final deployment link will be added after the unified application is deployed.

---

# Project Structure

Stock_Market/

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
├── dashboard.py
├── app.py
│
├── aapl_5day_model.pkl
│
├── requirements.txt
├── README.md
│
└── media/

---

# Files Description

## Data

- AAPL.csv — Apple historical stock data.
- AMZN.csv — Amazon historical stock data.
- MSFT.csv — Microsoft historical stock data.
- NVDA.csv — NVIDIA historical stock data.
- TSLA.csv — Tesla historical stock data.
- combined_close_prices.csv — Combined closing prices used for the financial analysis.

## Analysis

- project.ipynb — Main Jupyter Notebook containing data processing, exploratory analysis, risk analysis, performance analysis, and visualizations.
- pred.ipynb — Jupyter Notebook containing the machine learning workflow, feature engineering, model training, validation, tuning, and evaluation.

## Application

- dashboard.py — Original Streamlit market analysis dashboard.
- app.py — Unified Streamlit application containing both the Market Dashboard and ML Prediction interface.

## Machine Learning

- aapl_5day_model.pkl — Trained Logistic Regression model used for AAPL 5-day direction prediction.

## Configuration

- requirements.txt — Required Python libraries.
- README.md — Project documentation.

---

# Technologies Used

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

## Development Environment

- Jupyter Notebook
- Visual Studio Code
- Python

---

# Installation

Clone the repository:

git clone <repository-url>

Navigate to the project directory:

cd Stock_Market

Install the required libraries:

pip install -r requirements.txt

---

# Run the Application

Run the unified Streamlit application:

streamlit run app.py

The application will open in your web browser.

---

# Machine Learning Workflow

The complete machine learning workflow can be summarized as:

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

---

# Complete Project Workflow

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

---

# Disclaimer

This project was developed for educational and analytical purposes only.

The financial analysis, machine learning predictions, probabilities, visualizations, and other outputs should not be considered financial or investment advice.

Machine learning predictions are based on historical data and the features used by the model. Actual future market behavior can differ significantly from model predictions.