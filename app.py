import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Stock Market Intelligence",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
   MAIN APP
   ========================================================= */

.stApp {
    background: #0b0f14;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   MAIN HEADER
   ========================================================= */

.main-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #9ca3af;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}


/* =========================================================
   GENERAL CARDS
   ========================================================= */

.metric-card {
    background: linear-gradient(
        145deg,
        #151a21,
        #10141a
    );

    border: 1px solid #252c36;
    border-radius: 18px;

    padding: 1.4rem;

    box-shadow:
        0 8px 30px rgba(0, 0, 0, 0.25);
}

.metric-label {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}

.metric-value {
    color: #f9fafb;
    font-size: 1.7rem;
    font-weight: 750;
}


/* =========================================================
   PREDICTION CARD
   ========================================================= */

.prediction-card {
    background: linear-gradient(
        145deg,
        #151a21,
        #0f1318
    );

    border: 1px solid #2b333e;
    border-radius: 22px;

    padding: 2rem;

    text-align: center;

    margin-top: 1rem;

    box-shadow:
        0 12px 40px rgba(0, 0, 0, 0.35);
}

.prediction-label {
    color: #9ca3af;
    font-size: 0.95rem;
    margin-bottom: 0.5rem;
}

.prediction-value {
    font-size: 3rem;
    font-weight: 850;
    margin: 0.3rem 0;
}


/* =========================================================
   UP PREDICTION
   ========================================================= */

.prediction-up {
    border: 1px solid #22c55e;

    background: linear-gradient(
        145deg,
        #0d2a1b,
        #101a15
    );
}

.prediction-up .prediction-value {
    color: #4ade80;
}


/* =========================================================
   DOWN PREDICTION
   ========================================================= */

.prediction-down {
    border: 1px solid #ef4444;

    background: linear-gradient(
        145deg,
        #301316,
        #181114
    );
}

.prediction-down .prediction-value {
    color: #f87171;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #11161d 0%,
        #0b0f14 100%
    );

    border-right: 1px solid #20262f;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.8rem;
    padding-left: 1rem;
    padding-right: 1rem;
}


/* =========================================================
   SIDEBAR BRAND
   ========================================================= */

.sidebar-brand {
    text-align: center;

    padding: 1rem 0 1.5rem 0;

    margin-bottom: 1rem;

    border-bottom: 1px solid #252c36;
}

.sidebar-brand-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #f9fafb;
}

.sidebar-brand-subtitle {
    color: #6b7280;
    font-size: 0.78rem;
    margin-top: 0.25rem;
}


/* =========================================================
   NAVIGATION TITLE
   ========================================================= */

.sidebar-nav-title {
    color: #9ca3af;

    font-size: 0.75rem;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 1px;

    margin-bottom: 0.7rem;
}


/* =========================================================
   RADIO GROUP
   ========================================================= */

div[role="radiogroup"] {
    gap: 10px;
}


/* =========================================================
   NAVIGATION BUTTONS
   ========================================================= */

div[role="radiogroup"] label {
    background: #151a21;

    border: 1px solid #252c36;

    border-radius: 14px;

    padding: 15px 16px;

    margin-bottom: 8px;

    cursor: pointer;

    transition:
        background 0.25s ease,
        border-color 0.25s ease,
        transform 0.25s ease,
        box-shadow 0.25s ease;
}


/* =========================================================
   HOVER EFFECT
   ========================================================= */

div[role="radiogroup"] label:hover {
    background: linear-gradient(
        135deg,
        #1c2530,
        #151a21
    );

    border-color: #4b5563;

    transform: translateX(5px);

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.25);
}


/* =========================================================
   NAVIGATION TEXT
   ========================================================= */

div[role="radiogroup"] label p {
    font-size: 0.95rem;

    font-weight: 600;

    color: #d1d5db;
}


/* =========================================================
   SELECTED NAVIGATION
   ========================================================= */

div[role="radiogroup"] label:has(
    input[type="radio"]:checked
) {
    background: linear-gradient(
        135deg,
        #1f2937,
        #17202b
    );

    border-color: #60a5fa;

    box-shadow:
        0 0 0 1px rgba(96, 165, 250, 0.15),
        0 8px 30px rgba(0, 0, 0, 0.3);
}


/* Selected text */

div[role="radiogroup"] label:has(
    input[type="radio"]:checked
) p {
    color: #ffffff;

    font-weight: 700;
}


/* Radio */

div[role="radiogroup"] label input[type="radio"] {
    accent-color: #60a5fa;
}


/* =========================================================
   SIDEBAR STATUS
   ========================================================= */

.sidebar-status {
    margin-top: 2rem;

    padding: 0.9rem;

    background: #111820;

    border: 1px solid #202a34;

    border-radius: 12px;

    text-align: center;
}

.status-dot {
    display: inline-block;

    width: 8px;

    height: 8px;

    background: #22c55e;

    border-radius: 50%;

    margin-right: 6px;
}

.status-text {
    color: #9ca3af;

    font-size: 0.78rem;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;

    color: #6b7280;

    font-size: 0.8rem;

    margin-top: 3rem;

    padding-top: 1rem;

    border-top: 1px solid #20262f;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD ML MODEL
# =========================================================

model = joblib.load("aapl_5day_model.pkl")


# =========================================================
# LOAD HISTORICAL DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "combined_close_prices.csv",
        index_col="Date",
        parse_dates=True
    )

    return df


df = load_data()


# =========================================================
# SIDEBAR BRAND
# =========================================================

st.sidebar.markdown(
    """
<div class="sidebar-brand">
    <div class="sidebar-brand-title">
        🍎 Stock Market
    </div>
    <div class="sidebar-brand-subtitle">
        Intelligence Platform
    </div>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION
# =========================================================

st.sidebar.markdown(
    '<div class="sidebar-nav-title">Navigation</div>',
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Market Dashboard",
        "🤖 ML Prediction"
    ],
    label_visibility="collapsed"
)


# =========================================================
# SIDEBAR STATUS
# =========================================================

st.sidebar.markdown(
    """
<div class="sidebar-status">
    <span class="status-dot"></span>
    <span class="status-text">
        Data & ML Systems Active
    </span>
</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# =========================================================
# MARKET DASHBOARD
# =========================================================
# =========================================================

if page == "📊 Market Dashboard":

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '📊 Stock Market Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Historical analysis of AAPL, MSFT, NVDA, AMZN and TSLA'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # SIDEBAR CONTROLS
    # -----------------------------------------------------

    st.sidebar.markdown(
        "### Dashboard Controls"
    )


    selected_stocks = st.sidebar.multiselect(
        "Select Stocks",
        options=df.columns.tolist(),
        default=df.columns.tolist()
    )


    start_date = st.sidebar.date_input(
        "Start Date",
        value=df.index.min().date()
    )


    end_date = st.sidebar.date_input(
        "End Date",
        value=df.index.max().date()
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if not selected_stocks:

        st.warning(
            "Please select at least one stock."
        )

        st.stop()


    if start_date > end_date:

        st.error(
            "Start date must be before end date."
        )

        st.stop()


    # -----------------------------------------------------
    # FILTER DATA
    # -----------------------------------------------------

    filtered_df = df.loc[
        (df.index >= pd.to_datetime(start_date)) &
        (df.index <= pd.to_datetime(end_date)),
        selected_stocks
    ]


    if filtered_df.empty:

        st.warning(
            "No data available for the selected date range."
        )

        st.stop()


    # -----------------------------------------------------
    # CALCULATIONS
    # -----------------------------------------------------

    daily_returns = (
        filtered_df
        .pct_change()
        .dropna()
    )


    # Cumulative Return

    cumulative_returns = (
        (
            filtered_df.iloc[-1] /
            filtered_df.iloc[0]
        ) - 1
    ) * 100


    # Annualized Volatility

    annual_volatility = (
        daily_returns.std() *
        np.sqrt(252)
    ) * 100


    # Annualized Return

    annual_return = (
        daily_returns.mean() * 252
    )


    # Sharpe Ratio

    sharpe_ratio = (
        annual_return /
        (
            daily_returns.std() *
            np.sqrt(252)
        )
    )


    # Maximum Drawdown

    rolling_max = filtered_df.cummax()

    drawdown = (
        filtered_df /
        rolling_max
    ) - 1

    max_drawdown = drawdown.min() * 100


    # -----------------------------------------------------
    # KPI SECTION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📌 Key Performance Indicators'
        '</div>',
        unsafe_allow_html=True
    )


    best_return_stock = cumulative_returns.idxmax()
    best_return_value = cumulative_returns.max()

    best_sharpe_stock = sharpe_ratio.idxmax()
    best_sharpe_value = sharpe_ratio.max()

    lowest_vol_stock = annual_volatility.idxmin()
    lowest_vol_value = annual_volatility.min()

    lowest_drawdown_stock = max_drawdown.idxmax()
    lowest_drawdown_value = max_drawdown.max()


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Cumulative Return",
        best_return_stock,
        f"{best_return_value:.2f}%"
    )


    col2.metric(
        "Sharpe Ratio",
        best_sharpe_stock,
        f"{best_sharpe_value:.2f}"
    )


    col3.metric(
        "Lowest Volatility",
        lowest_vol_stock,
        f"{lowest_vol_value:.2f}%"
    )


    col4.metric(
        "Lowest Drawdown",
        lowest_drawdown_stock,
        f"{lowest_drawdown_value:.2f}%"
    )


    # -----------------------------------------------------
    # NORMALIZED PERFORMANCE
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📈 Normalized Stock Performance'
        '</div>',
        unsafe_allow_html=True
    )


    normalized = (
        filtered_df /
        filtered_df.iloc[0]
    )


    fig = px.line(
        normalized,
        title="Normalized Stock Performance",
        labels={
            "value": "Normalized Price",
            "Date": "Date"
        }
    )


    fig.update_layout(
        height=500,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # CUMULATIVE RETURNS + SHARPE
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 📊 Cumulative Returns"
        )


        cumulative_df = (
            cumulative_returns
            .reset_index()
        )


        cumulative_df.columns = [
            "Stock",
            "Cumulative Return (%)"
        ]


        fig = px.bar(
            cumulative_df,
            x="Stock",
            y="Cumulative Return (%)",
            text="Cumulative Return (%)",
            title="Cumulative Return Comparison"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        st.markdown(
            "### 📐 Sharpe Ratio"
        )


        sharpe_df = (
            sharpe_ratio
            .reset_index()
        )


        sharpe_df.columns = [
            "Stock",
            "Sharpe Ratio"
        ]


        fig = px.bar(
            sharpe_df,
            x="Stock",
            y="Sharpe Ratio",
            text="Sharpe Ratio",
            title="Sharpe Ratio Comparison"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # RISK VS RETURN
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '⚖️ Risk vs Return Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    risk_return_df = pd.DataFrame({

        "Stock":
            daily_returns.columns,

        "Annualized Return (%)":
            annual_return.values * 100,

        "Annualized Volatility (%)":
            annual_volatility.values

    })


    fig = px.scatter(
        risk_return_df,
        x="Annualized Volatility (%)",
        y="Annualized Return (%)",
        text="Stock",
        size="Annualized Return (%)",
        title="Risk vs Return"
    )


    fig.update_traces(
        textposition="top center"
    )


    fig.update_layout(
        height=500,
        template="plotly_dark"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # -----------------------------------------------------
    # CORRELATION + DRAWDOWN
    # -----------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            "### 🔗 Correlation Heatmap"
        )


        correlation = (
            daily_returns.corr()
        )


        fig = px.imshow(
            correlation,
            text_auto=".2f",
            title="Stock Returns Correlation Matrix",
            aspect="auto"
        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    with col2:

        st.markdown(
            "### 📉 Maximum Drawdown"
        )


        drawdown_df = (
            max_drawdown
            .reset_index()
        )


        drawdown_df.columns = [
            "Stock",
            "Maximum Drawdown (%)"
        ]


        fig = px.bar(
            drawdown_df,
            x="Stock",
            y="Maximum Drawdown (%)",
            text="Maximum Drawdown (%)",
            title="Maximum Drawdown Comparison"
        )


        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )


        fig.update_layout(
            template="plotly_dark"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">'
        '📋 Performance Summary'
        '</div>',
        unsafe_allow_html=True
    )


    summary = pd.DataFrame({

        "Cumulative Return (%)":
            cumulative_returns,

        "Annualized Return (%)":
            annual_return * 100,

        "Annualized Volatility (%)":
            annual_volatility,

        "Sharpe Ratio":
            sharpe_ratio,

        "Maximum Drawdown (%)":
            max_drawdown

    })


    summary = summary.round(2)


    st.dataframe(
        summary,
        use_container_width=True
    )


    # -----------------------------------------------------
    # DASHBOARD FOOTER
    # -----------------------------------------------------

    st.markdown(
        """<div class="footer">
Stock market analysis dashboard for educational and analytical purposes.
</div>""",
        unsafe_allow_html=True
    )


# =========================================================
# =========================================================
# MACHINE LEARNING PREDICTION
# =========================================================
# =========================================================

else:

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '🍎 AAPL Stock Intelligence'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="subtitle">'
        'Machine Learning powered 5-day direction prediction for Apple Inc.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # GET LIVE AAPL DATA
    # -----------------------------------------------------

    ticker = yf.Ticker("AAPL")


    live_data = ticker.history(
        period="1mo",
        interval="1d"
    )


    if live_data.empty:

        st.error(
            "Unable to retrieve AAPL market data. "
            "Please try again later."
        )

        st.stop()


    # Latest Date

    latest_date = live_data.index[-1]


    # Latest Price

    latest_price = (
        live_data["Close"].iloc[-1]
    )


    # -----------------------------------------------------
    # MARKET OVERVIEW
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '📊 Market Overview'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            f"""<div class="metric-card">
<div class="metric-label">Latest AAPL Price</div>
<div class="metric-value">${latest_price:.2f}</div>
</div>""",
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""<div class="metric-card">
<div class="metric-label">Latest Data Date</div>
<div class="metric-value">{latest_date.strftime('%b %d, %Y')}</div>
</div>""",
            unsafe_allow_html=True
        )


    # -----------------------------------------------------
    # FEATURE ENGINEERING
    # -----------------------------------------------------

    prices = (
        live_data["Close"]
        .rename("AAPL")
    )


    features = pd.DataFrame({
        "AAPL": prices
    })


    # Daily Return

    features["AAPL_Return"] = (
        features["AAPL"].pct_change()
    )


    # Moving Average 5

    features["AAPL_MA_5"] = (
        features["AAPL"]
        .rolling(window=5)
        .mean()
    )


    # Moving Average 10

    features["AAPL_MA_10"] = (
        features["AAPL"]
        .rolling(window=10)
        .mean()
    )


    # Lag 1

    features["AAPL_Lag_1"] = (
        features["AAPL"]
        .shift(1)
    )


    # Lag 2

    features["AAPL_Lag_2"] = (
        features["AAPL"]
        .shift(2)
    )


    # Lag 3

    features["AAPL_Lag_3"] = (
        features["AAPL"]
        .shift(3)
    )


    # Volatility 5

    features["AAPL_Volatility_5"] = (
        features["AAPL_Return"]
        .rolling(window=5)
        .std()
    )


    # Remove missing values

    features = features.dropna()


    # Exact features used during training

    latest_features = features[
        [
            "AAPL",
            "AAPL_Return",
            "AAPL_MA_5",
            "AAPL_MA_10",
            "AAPL_Lag_1",
            "AAPL_Lag_2",
            "AAPL_Lag_3",
            "AAPL_Volatility_5"
        ]
    ].iloc[[-1]]


    # -----------------------------------------------------
    # PREDICT BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔮 Predict",
        type="primary"
    ):

        # Model prediction

        prediction = model.predict(
            latest_features
        )[0]


        # Prediction probabilities

        probabilities = model.predict_proba(
            latest_features
        )[0]


        # -------------------------------------------------
        # DIRECTION
        # -------------------------------------------------

        if prediction == 1:

            direction = "UP"
            icon = "📈"
            prediction_class = "prediction-up"

        else:

            direction = "DOWN"
            icon = "📉"
            prediction_class = "prediction-down"


        # -------------------------------------------------
        # PREDICTION RESULT
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🔮 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f"""<div class="prediction-card {prediction_class}">
<div class="prediction-label">Model Prediction</div>
<div class="prediction-value">{icon} {direction}</div>
<div class="prediction-label">Forecast: Next 5 Trading Days</div>
</div>""",
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # PROBABILITIES
        # -------------------------------------------------

        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                f"""<div class="metric-card">
<div class="metric-label">📉 DOWN Probability</div>
<div class="metric-value">{probabilities[0]:.2%}</div>
</div>""",
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                f"""<div class="metric-card">
<div class="metric-label">📈 UP Probability</div>
<div class="metric-value">{probabilities[1]:.2%}</div>
</div>""",
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # FORECAST HORIZON
        # -------------------------------------------------

        st.markdown(
            """<div class="metric-card"
style="margin-top: 1rem; text-align: center;">
<div class="metric-label">Forecast Horizon</div>
<div class="metric-value">Next 5 Trading Days</div>
</div>""",
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # MODEL INFORMATION
        # -----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🧠 Model Information'
            '</div>',
            unsafe_allow_html=True
        )


        model_col1, model_col2, model_col3 = st.columns(3)


        model_col1.metric(
            "Model",
            "Logistic Regression"
        )


        model_col2.metric(
            "Prediction Horizon",
            "5 Trading Days"
        )


        model_col3.metric(
            "Features",
            "8"
        )


        # -------------------------------------------------
        # ML DISCLAIMER
        # -------------------------------------------------

        st.markdown(
            """<div class="footer">
This prediction is generated by a machine learning model
and should not be considered financial advice.
</div>""",
            unsafe_allow_html=True
        )