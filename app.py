import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="⚡ Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0b1020;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
}

h2 {
    font-weight: 700 !important;
}

h3 {
    font-weight: 650 !important;
}


/* KPI CARDS */

.kpi-card {
    padding: 22px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #111936,
        #18234a
    );
    border: 1px solid #29365f;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    margin-bottom: 10px;
}

.kpi-title {
    font-size: 14px;
    opacity: 0.75;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 28px;
    font-weight: 800;
}

.kpi-icon {
    font-size: 25px;
}


/* SECTION */

.section-box {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(
        135deg,
        #10172d,
        #151f3c
    );
    border: 1px solid #29365f;
    margin-top: 15px;
    margin-bottom: 15px;
}


/* HERO */

.hero {
    padding: 28px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        #101a38,
        #1c2c5c,
        #101a38
    );
    border: 1px solid #334579;
    box-shadow: 0 12px 35px rgba(0,0,0,0.3);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 44px;
    font-weight: 900;
}

.hero-subtitle {
    font-size: 18px;
    opacity: 0.8;
}


/* PREDICTION */

.prediction-box {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        #13233d,
        #183b4b
    );
    border: 1px solid #2d6170;
    text-align: center;
    margin-top: 15px;
    margin-bottom: 20px;
}

.prediction-value {
    font-size: 48px;
    font-weight: 900;
}


/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0a1022,
        #111a35
    );
}


/* BUTTON */

.stButton > button {
    border-radius: 12px;
    font-weight: 700;
    min-height: 45px;
}


/* DOWNLOAD BUTTON */

.stDownloadButton > button {
    border-radius: 12px;
    font-weight: 700;
}


/* INFO */

.info-box {
    padding: 18px;
    border-radius: 15px;
    background: #111b35;
    border: 1px solid #2b3d68;
}


/* FOOTER */

.footer {
    text-align: center;
    opacity: 0.55;
    padding: 20px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

try:

    data = pd.read_csv("energy_data.csv")

except FileNotFoundError:

    st.error(
        "❌ energy_data.csv not found. "
        "Place it in the same folder as app.py."
    )

    st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================

data["datetime"] = pd.to_datetime(
    data["datetime"]
)

data = data.sort_values(
    "datetime"
).reset_index(drop=True)


# =========================================================
# TIME FEATURES
# =========================================================

data["year"] = data["datetime"].dt.year

data["month"] = data["datetime"].dt.month

data["day"] = data["datetime"].dt.day

data["hour"] = data["datetime"].dt.hour

data["day_of_week"] = (
    data["datetime"].dt.dayofweek
)


# =========================================================
# LAG FEATURES
# =========================================================

data["lag_1"] = (
    data["energy_consumption"].shift(1)
)

data["lag_24"] = (
    data["energy_consumption"].shift(24)
)

data["lag_168"] = (
    data["energy_consumption"].shift(168)
)


# =========================================================
# ROLLING AVERAGE
# =========================================================

data["rolling_24h"] = (

    data[
        "energy_consumption"
    ]
    .shift(1)
    .rolling(24)
    .mean()

)


# Remove missing values

data = data.dropna().reset_index(
    drop=True
)


# =========================================================
# FEATURES
# =========================================================

features = [

    "hour",

    "day_of_week",

    "month",

    "lag_1",

    "lag_24",

    "lag_168",

    "rolling_24h"

]


X = data[features]

y = data["energy_consumption"]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

split = int(
    len(data) * 0.80
)


X_train = X.iloc[:split]

X_test = X.iloc[split:]


y_train = y.iloc[:split]

y_test = y.iloc[split:]


# =========================================================
# MODEL
# =========================================================

model = RandomForestRegressor(

    n_estimators=120,

    random_state=42,

    n_jobs=-1

)


# Train

model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTION
# =========================================================

test_prediction = model.predict(
    X_test
)


# =========================================================
# METRICS
# =========================================================

mae = mean_absolute_error(
    y_test,
    test_prediction
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_prediction
    )
)


r2 = r2_score(
    y_test,
    test_prediction
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## ⚡ ENERGY AI"
    )

    st.caption(
        "Machine Learning Energy Forecasting"
    )

    st.divider()

    st.markdown(
        "### 📊 Dataset"
    )

    st.write(
        f"Records: **{len(data):,}**"
    )

    st.write(
        f"Start: **{data['datetime'].min().date()}**"
    )

    st.write(
        f"End: **{data['datetime'].max().date()}**"
    )

    st.divider()

    st.markdown(
        "### 🤖 Model"
    )

    st.write(
        "Random Forest Regression"
    )

    st.write(
        "Features: **7**"
    )

    st.write(
        "Temperature: **Not Used**"
    )

    st.divider()

    st.markdown(
        "### 🎯 Purpose"
    )

    st.caption(
        "Predict electricity consumption "
        "using historical consumption "
        "and time-based patterns."
    )


# =========================================================
# HERO
# =========================================================

st.markdown("""

<div class="hero">

<div class="hero-title">
⚡ ENERGY AI
</div>

<div class="hero-subtitle">
Real-World Electricity Consumption Forecasting
using Random Forest Machine Learning
</div>

</div>

""", unsafe_allow_html=True)


# =========================================================
# TOP KPIs
# =========================================================

st.subheader(
    "📊 System Overview"
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    st.markdown(
        f"""
        <div class="kpi-card">
        <div class="kpi-icon">⚡</div>
        <div class="kpi-title">
        TOTAL RECORDS
        </div>
        <div class="kpi-value">
        {len(data):,}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k2:

    st.markdown(
        f"""
        <div class="kpi-card">
        <div class="kpi-icon">📈</div>
        <div class="kpi-title">
        AVERAGE ENERGY
        </div>
        <div class="kpi-value">
        {data['energy_consumption'].mean():.2f}
        </div>
        <small>kWh</small>
        </div>
        """,
        unsafe_allow_html=True
    )


with k3:

    st.markdown(
        f"""
        <div class="kpi-card">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-title">
        R² SCORE
        </div>
        <div class="kpi-value">
        {r2:.3f}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with k4:

    st.markdown(
        f"""
        <div class="kpi-card">
        <div class="kpi-icon">📅</div>
        <div class="kpi-title">
        DATA PERIOD
        </div>
        <div class="kpi-value">
        2006–2010
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.divider()

st.header(
    "🤖 Machine Learning Performance"
)


m1, m2, m3 = st.columns(3)


with m1:

    st.metric(
        "MAE",
        f"{mae:.3f} kWh"
    )


with m2:

    st.metric(
        "RMSE",
        f"{rmse:.3f} kWh"
    )


with m3:

    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )


# =========================================================
# CONSUMPTION TREND
# =========================================================

st.divider()

st.header(
    "📈 Energy Consumption Trend"
)


# Daily average

daily_data = (

    data

    .set_index("datetime")

    ["energy_consumption"]

    .resample("D")

    .mean()

    .reset_index()

)


trend_fig = go.Figure()


trend_fig.add_trace(

    go.Scatter(

        x=daily_data["datetime"],

        y=daily_data[
            "energy_consumption"
        ],

        mode="lines",

        name="Daily Average",

        line=dict(
            width=2
        )

    )

)


trend_fig.update_layout(

    title=
    "Daily Average Energy Consumption",

    xaxis_title="Date",

    yaxis_title=
    "Energy Consumption (kWh)",

    template="plotly_dark",

    height=450

)


st.plotly_chart(

    trend_fig,

    use_container_width=True

)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

st.header(
    "🎯 Actual vs Predicted"
)


comparison = pd.DataFrame({

    "datetime":
        data[
            "datetime"
        ].iloc[split:],

    "actual":
        y_test.values,

    "predicted":
        test_prediction

})


# Display only last 500 test points
comparison_display = comparison.tail(
    500
)


actual_fig = go.Figure()


actual_fig.add_trace(

    go.Scatter(

        x=comparison_display[
            "datetime"
        ],

        y=comparison_display[
            "actual"
        ],

        mode="lines",

        name="Actual"

    )

)


actual_fig.add_trace(

    go.Scatter(

        x=comparison_display[
            "datetime"
        ],

        y=comparison_display[
            "predicted"
        ],

        mode="lines",

        name="Predicted"

    )

)


actual_fig.update_layout(

    title=
    "Actual vs Predicted Energy",

    xaxis_title="Date",

    yaxis_title=
    "Energy Consumption (kWh)",

    template="plotly_dark",

    height=450

)


st.plotly_chart(

    actual_fig,

    use_container_width=True

)


# =========================================================
# DATE / TIME PREDICTION
# =========================================================

st.divider()

st.header(
    "🎯 Predict Energy Consumption"
)


st.write(
    "Choose a date and time from the available "
    "historical period."
)


p1, p2 = st.columns(2)


with p1:

    prediction_date = st.date_input(

        "📅 Select Date",

        value=
        data["datetime"].max().date(),

        min_value=
        data["datetime"].min().date(),

        max_value=
        data["datetime"].max().date()

    )


with p2:

    prediction_time = st.time_input(

        "⏰ Select Time",

        value=
        pd.Timestamp("12:00").time()

    )


selected_datetime = pd.Timestamp(

    prediction_date

).replace(

    hour=
    prediction_time.hour,

    minute=
    prediction_time.minute

)


if st.button(

    "⚡ GENERATE PREDICTION",

    use_container_width=True

):

    previous = data[

        data["datetime"]
        <
        selected_datetime

    ]


    if len(previous) < 168:

        st.error(
            "Not enough historical data."
        )

    else:

        input_data = pd.DataFrame({

            "hour": [
                selected_datetime.hour
            ],

            "day_of_week": [
                selected_datetime.dayofweek
            ],

            "month": [
                selected_datetime.month
            ],

            "lag_1": [

                previous[
                    "energy_consumption"
                ].iloc[-1]

            ],

            "lag_24": [

                previous[
                    "energy_consumption"
                ].iloc[-24]

            ],

            "lag_168": [

                previous[
                    "energy_consumption"
                ].iloc[-168]

            ],

            "rolling_24h": [

                previous[
                    "energy_consumption"
                ].iloc[-24:].mean()

            ]

        })


        prediction = model.predict(
            input_data
        )[0]


        st.markdown(

            f"""
            <div class="prediction-box">

            <div>
            ⚡ PREDICTED ENERGY
            </div>

            <div class="prediction-value">
            {prediction:.2f} kWh
            </div>

            <div>
            {selected_datetime.strftime(
                "%d %B %Y • %H:%M"
            )}
            </div>

            </div>
            """,

            unsafe_allow_html=True

        )


        # Actual value

        actual = data[

            data["datetime"]
            ==
            selected_datetime

        ]


        if len(actual) > 0:

            actual_value = (

                actual[
                    "energy_consumption"
                ].iloc[0]

            )


            difference = abs(

                actual_value
                -
                prediction

            )


            a1, a2, a3 = st.columns(3)


            with a1:

                st.metric(

                    "Actual",

                    f"{actual_value:.2f} kWh"

                )


            with a2:

                st.metric(

                    "Predicted",

                    f"{prediction:.2f} kWh"

                )


            with a3:

                st.metric(

                    "Difference",

                    f"{difference:.2f} kWh"

                )


# =========================================================
# 24-HOUR FORECAST
# =========================================================

st.divider()

st.header(
    "🔮 Next 24-Hour Forecast"
)


st.write(
    "Forecast generated recursively from the "
    "latest available historical consumption."
)


# Latest values

last_value = data[
    "energy_consumption"
].iloc[-1]


history_24 = list(

    data[
        "energy_consumption"
    ].iloc[-24:]

)


history_168 = list(

    data[
        "energy_consumption"
    ].iloc[-168:]

)


last_datetime = data[
    "datetime"
].iloc[-1]


future_times = pd.date_range(

    start=
    last_datetime
    +
    pd.Timedelta(hours=1),

    periods=24,

    freq="h"

)


future_predictions = []


# =========================================================
# RECURSIVE FORECAST
# =========================================================

for future_time in future_times:

    lag_1 = last_value

    lag_24 = history_24[-24]

    lag_168 = history_168[-168]

    rolling = np.mean(
        history_24[-24:]
    )


    model_input = pd.DataFrame({

        "hour": [
            future_time.hour
        ],

        "day_of_week": [
            future_time.dayofweek
        ],

        "month": [
            future_time.month
        ],

        "lag_1": [
            lag_1
        ],

        "lag_24": [
            lag_24
        ],

        "lag_168": [
            lag_168
        ],

        "rolling_24h": [
            rolling
        ]

    })


    predicted = model.predict(
        model_input
    )[0]


    future_predictions.append(
        predicted
    )


    history_24.append(
        predicted
    )

    history_168.append(
        predicted
    )

    last_value = predicted


forecast = pd.DataFrame({

    "datetime":
        future_times,

    "predicted_energy":
        future_predictions

})


# =========================================================
# FORECAST CHART
# =========================================================

forecast_fig = go.Figure()


forecast_fig.add_trace(

    go.Scatter(

        x=forecast["datetime"],

        y=forecast[
            "predicted_energy"
        ],

        mode="lines+markers",

        name="Forecast",

        line=dict(
            width=4
        )

    )

)


forecast_fig.update_layout(

    title=
    "⚡ Next 24 Hours Energy Forecast",

    xaxis_title=
    "Date & Time",

    yaxis_title=
    "Predicted Energy (kWh)",

    template="plotly_dark",

    height=500

)


st.plotly_chart(

    forecast_fig,

    use_container_width=True

)


# =========================================================
# FORECAST KPIs
# =========================================================

f1, f2, f3, f4 = st.columns(4)


forecast_average = forecast[
    "predicted_energy"
].mean()


forecast_peak = forecast[
    "predicted_energy"
].max()


forecast_low = forecast[
    "predicted_energy"
].min()


forecast_total = forecast[
    "predicted_energy"
].sum()


with f1:

    st.metric(

        "Average",

        f"{forecast_average:.2f} kWh"

    )


with f2:

    st.metric(

        "🔥 Peak",

        f"{forecast_peak:.2f} kWh"

    )


with f3:

    st.metric(

        "🟢 Lowest",

        f"{forecast_low:.2f} kWh"

    )


with f4:

    st.metric(

        "⚡ 24h Total",

        f"{forecast_total:.2f} kWh"

    )


# =========================================================
# PEAK ANALYSIS
# =========================================================

peak_row = forecast.loc[

    forecast[
        "predicted_energy"
    ].idxmax()

]


low_row = forecast.loc[

    forecast[
        "predicted_energy"
    ].idxmin()

]


st.subheader(
    "🔥 Forecast Intelligence"
)


c1, c2 = st.columns(2)


with c1:

    st.warning(

        f"""
        **Peak Consumption**

        ⚡ {peak_row['predicted_energy']:.2f} kWh

        🕐 {peak_row['datetime'].strftime(
            '%d %B %Y • %H:%M'
        )}
        """

    )


with c2:

    st.success(

        f"""
        **Lowest Consumption**

        🟢 {low_row['predicted_energy']:.2f} kWh

        🕐 {low_row['datetime'].strftime(
            '%d %B %Y • %H:%M'
        )}
        """

    )


# =========================================================
# SMART RECOMMENDATION
# =========================================================

st.subheader(
    "💡 Smart Energy Recommendation"
)


if forecast_peak > forecast_average * 1.30:

    st.info(

        "⚠️ Higher consumption is expected during "
        "peak hours. If possible, shift flexible "
        "activities such as washing, charging, or "
        "other high-energy tasks toward lower-demand "
        "hours."

    )

else:

    st.success(

        "✅ Consumption is expected to remain "
        "relatively stable during the next 24 hours."

    )


# =========================================================
# FORECAST TABLE
# =========================================================

st.subheader(
    "📋 Detailed Forecast"
)


forecast_table = forecast.copy()


forecast_table["datetime"] = (

    forecast_table[
        "datetime"
    ].dt.strftime(
        "%Y-%m-%d %H:%M"
    )

)


forecast_table[
    "predicted_energy"
] = (

    forecast_table[
        "predicted_energy"
    ].round(3)

)


forecast_table = forecast_table.rename(

    columns={

        "datetime":
            "Date & Time",

        "predicted_energy":
            "Predicted Energy (kWh)"

    }

)


st.dataframe(

    forecast_table,

    use_container_width=True,

    hide_index=True

)


# =========================================================
# DOWNLOAD
# =========================================================

st.subheader(
    "📥 Export Forecast"
)


download_csv = forecast.to_csv(
    index=False
)


st.download_button(

    label=
    "📥 DOWNLOAD FORECAST CSV",

    data=
    download_csv,

    file_name=
    "energy_forecast.csv",

    mime=
    "text/csv",

    use_container_width=True

)


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

with st.expander(
    "ℹ️ About This Project"
):

    st.write(

        """
        **Energy AI** is a machine-learning based
        electricity consumption forecasting system.

        **Machine Learning Algorithm:**
        Random Forest Regression

        **Input Features:**
        - Hour
        - Day of Week
        - Month
        - Previous Hour Consumption
        - Previous 24-Hour Consumption
        - Previous 7-Day Consumption
        - 24-Hour Rolling Average

        **Target:**
        Electricity Energy Consumption (kWh)

        **Temperature Feature:**
        Not used.

        **Dataset:**
        Real-world household electricity
        consumption data converted from
        minute-level measurements into
        hourly consumption.

        **Interface:**
        Streamlit

        **Visualization:**
        Plotly
        """

    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(

    """
    <div class="footer">

    ⚡ ENERGY AI • Random Forest Forecasting •
    Real Electricity Data • No Temperature

    </div>
    """,

    unsafe_allow_html=True

)