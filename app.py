import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


st.set_page_config(
    page_title="Stock Predictor",
    layout="wide"
)



@st.cache_data
def search_stocks(query):

    search = yf.Search(query)

    results = search.quotes

    stocks = []

    for stock in results:

        symbol = stock.get("symbol")
        name = stock.get("longname") or stock.get("shortname")

        if symbol and name:
            stocks.append({
                "name": name,
                "symbol": symbol
            })

    return stocks


@st.cache_data
def fetch_data(symbol):

    stock = yf.Ticker(symbol)

    data = stock.history(period="5y")

    if data.empty:
        return None

    return data[["Close"]].dropna()


def normalize_data(df):

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        df[["Close"]]
    )

    return scaled_data, scaler


def prepare_data(scaled_data, time_steps=60):

    X = []
    y = []

    for i in range(time_steps, len(scaled_data)):

        X.append(
            scaled_data[i - time_steps:i, 0]
        )

        y.append(
            scaled_data[i, 0]
        )

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    return X, y


def build_model(input_shape):

    model = Sequential([

        LSTM(
            50,
            return_sequences=True,
            input_shape=input_shape
        ),

        LSTM(
            50,
            return_sequences=False
        ),

        Dense(25),

        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


def predict_future(
    model,
    scaled_data,
    scaler,
    days
):

    input_sequence = scaled_data[-60:].copy()

    predictions = []

    for i in range(days):

        model_input = input_sequence.reshape(
            1,
            60,
            1
        )

        predicted = model.predict(
            model_input,
            verbose=0
        )[0][0]

        predictions.append(predicted)

        input_sequence = np.append(
            input_sequence[1:],
            [[predicted]],
            axis=0
        )

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions.flatten()



st.title("Stock Price Predictor")

st.write(
    "Search for a company and predict its future stock price using an LSTM model."
)

st.divider()



st.sidebar.header("Stock Selection")

search_query = st.sidebar.text_input(
    "Search company",
    placeholder="Example: Apple, Reliance, Tesla"
)


selected_stock = None
selected_name = None


if search_query:

    results = search_stocks(search_query)

    if results:

        options = [
            f"{stock['name']} ({stock['symbol']})"
            for stock in results
        ]

        selected = st.sidebar.selectbox(
            "Select company",
            options
        )

        selected_index = options.index(selected)

        selected_stock = results[selected_index]["symbol"]
        selected_name = results[selected_index]["name"]

    else:

        st.sidebar.error(
            "No stocks found."
        )




prediction_days = st.sidebar.slider(
    "Prediction Days",
    1,
    10,
    5
)




if selected_stock:

    with st.spinner("Fetching stock data..."):

        data = fetch_data(selected_stock)


    if data is None:

        st.error(
            "Unable to fetch data for this stock."
        )

        st.stop()


    latest_price = data["Close"].iloc[-1]



    st.subheader(
        f"{selected_name} ({selected_stock})"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Price",
            f"₹{latest_price:.2f}"
        )


    with col2:

        previous_price = data["Close"].iloc[-2]

        change = (
            (latest_price - previous_price)
            / previous_price
        ) * 100

        st.metric(
            "Daily Change",
            f"{change:.2f}%"
        )


    with col3:

        st.metric(
            "Data Points",
            len(data)
        )


    st.divider()


    st.subheader(
        "Historical Stock Price"
    )


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Close Price"
        )
    )


    fig.update_layout(
        height=450,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    if st.button(
        "Train Model and Predict",
        type="primary"
    ):

        with st.spinner(
            "Training model. This may take a minute..."
        ):

            scaled_data, scaler = normalize_data(
                data
            )


            X, y = prepare_data(
                scaled_data
            )



            split = int(
                len(X) * 0.8
            )


            X_train = X[:split]
            X_test = X[split:]

            y_train = y[:split]
            y_test = y[split:]




            model = build_model(
                (
                    X_train.shape[1],
                    X_train.shape[2]
                )
            )



            model.fit(
                X_train,
                y_train,
                epochs=5,
                batch_size=32,
                validation_data=(
                    X_test,
                    y_test
                ),
                verbose=0
            )



            test_predictions = model.predict(
                X_test,
                verbose=0
            )


            test_predictions = scaler.inverse_transform(
                test_predictions
            )


            actual_prices = scaler.inverse_transform(
                y_test.reshape(-1, 1)
            )


            rmse = np.sqrt(
                mean_squared_error(
                    actual_prices,
                    test_predictions
                )
            )


            mae = mean_absolute_error(
                actual_prices,
                test_predictions
            )



            future_prices = predict_future(
                model,
                scaled_data,
                scaler,
                prediction_days
            )



        st.divider()

        st.subheader(
            "Future Price Prediction"
        )


        final_prediction = future_prices[-1]


        expected_change = (
            (final_prediction - latest_price)
            / latest_price
        ) * 100


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Current Price",
                f"${latest_price:.2f}"
            )


        with col2:

            st.metric(
                f"Predicted Price ({prediction_days} days)",
                f"${final_prediction:.2f}"
            )


        with col3:

            st.metric(
                "Expected Change",
                f"{expected_change:.2f}%"
            )



        if expected_change > 2:

            signal = "BULLISH"

        elif expected_change < -2:

            signal = "BEARISH"

        else:

            signal = "NEUTRAL"


        st.subheader(
            f"Market Signal: {signal}"
        )


        st.subheader(
            "Model Performance"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "RMSE",
                f"${rmse:.2f}"
            )


        with col2:

            st.metric(
                "MAE",
                f"${mae:.2f}"
            )


    

        dates = pd.bdate_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=prediction_days
        )


        prediction_df = pd.DataFrame({

            "Date": dates,

            "Predicted Price":
                future_prices

        })


        st.subheader(
            "Predicted Prices"
        )


        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )



        st.subheader(
            "Future Price Forecast"
        )


        forecast_fig = go.Figure()


        forecast_fig.add_trace(
            go.Scatter(
                x=data.index[-100:],
                y=data["Close"].iloc[-100:],
                mode="lines",
                name="Historical Price"
            )
        )


        forecast_fig.add_trace(
            go.Scatter(
                x=dates,
                y=future_prices,
                mode="lines+markers",
                name="Predicted Price"
            )
        )


        forecast_fig.update_layout(
            height=500,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price"
        )


        st.plotly_chart(
            forecast_fig,
            use_container_width=True
        )


else:

    st.info(
        "Search for a company from the sidebar to begin."
    )