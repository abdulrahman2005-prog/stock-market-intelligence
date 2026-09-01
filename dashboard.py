import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "combined_close_prices.csv",
        index_col="Date",
        parse_dates=True
    )

    return df


df = load_data()


# ==========================================
# TITLE
# ==========================================

st.title("Stock Market Analysis Dashboard")

st.markdown(
    """
    This dashboard analyzes the historical performance of:

    **Apple (AAPL) | Microsoft (MSFT) | NVIDIA (NVDA) | Amazon (AMZN) | Tesla (TSLA)**
    """
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Dashboard Controls")

selected_stocks = st.sidebar.multiselect(
    "Select Stocks",
    options=df.columns.tolist(),
    default=df.columns.tolist()
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=df.index.min()
)

end_date = st.sidebar.date_input(
    "End Date",
    value=df.index.max()
)


# ==========================================
# FILTER DATA
# ==========================================

filtered_df = df.loc[
    (df.index >= pd.to_datetime(start_date)) &
    (df.index <= pd.to_datetime(end_date)),
    selected_stocks
]


# ==========================================
# CALCULATIONS
# ==========================================

daily_returns = filtered_df.pct_change().dropna()


# Cumulative Return
cumulative_returns = (
    (filtered_df.iloc[-1] / filtered_df.iloc[0]) - 1
) * 100


# Volatility
annual_volatility = (
    daily_returns.std() * np.sqrt(252)
) * 100


# Annualized Return
annual_return = (
    daily_returns.mean() * 252
)


# Sharpe Ratio
sharpe_ratio = (
    annual_return /
    (daily_returns.std() * np.sqrt(252))
)


# Maximum Drawdown
rolling_max = filtered_df.cummax()

drawdown = (
    filtered_df / rolling_max
) - 1

max_drawdown = drawdown.min() * 100


# ==========================================
# KPI SECTION
# ==========================================

st.subheader("Key Performance Indicators")


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
    "↑ Best Cumulative Return",
    f"{best_return_stock}",
    f"{best_return_value:.2f}%"
)

col2.metric(
    "↑ Best Sharpe Ratio",
    f"{best_sharpe_stock}",
    f"{best_sharpe_value:.2f}"
)

col3.metric(
    "↓ Lowest Volatility",
    f"{lowest_vol_stock}",
    f"{lowest_vol_value:.2f}%"
)

col4.metric(
    "↓ Lowest Maximum Drawdown",
    f"{lowest_drawdown_stock}",
    f"{lowest_drawdown_value:.2f}%"
)


# ==========================================
# NORMALIZED PERFORMANCE
# ==========================================

st.divider()

st.subheader("Normalized Stock Performance")

normalized = (
    filtered_df / filtered_df.iloc[0]
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
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# CUMULATIVE RETURNS
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Cumulative Returns")

    cumulative_df = cumulative_returns.reset_index()

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

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# SHARPE RATIO
# ==========================================

with col2:

    st.subheader("Sharpe Ratio")

    sharpe_df = sharpe_ratio.reset_index()

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

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# RISK VS RETURN
# ==========================================

st.divider()

st.subheader("Risk vs Return Analysis")

risk_return_df = pd.DataFrame({

    "Stock": daily_returns.columns,

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
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# CORRELATION HEATMAP
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Correlation Heatmap")

    correlation = daily_returns.corr()

    fig = px.imshow(

        correlation,

        text_auto=".2f",

        title="Stock Returns Correlation Matrix",

        aspect="auto"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================
# MAXIMUM DRAWDOWN
# ==========================================

with col2:

    st.subheader("Maximum Drawdown")

    drawdown_df = max_drawdown.reset_index()

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

    st.plotly_chart(

        fig,

        use_container_width=True

    )


# ==========================================
# SUMMARY TABLE
# ==========================================

st.divider()

st.subheader("Performance Summary Table")


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


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "This dashboard is for educational and analytical purposes only and does not constitute investment advice."
)