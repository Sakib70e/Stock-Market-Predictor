# Stock Price Predictor

A simple machine learning project that predicts future stock prices using an LSTM model.

## What it does

* Search for a company by name
* Select a stock from the search results
* Fetch historical stock data using Yahoo Finance
* Train an LSTM model using historical closing prices
* Predict the stock price for the next few days
* Display predicted prices in a table
* Show historical and predicted prices on a graph
* Calculate basic model performance using RMSE and MAE
* Give a simple Bullish, Bearish, or Neutral signal based on the prediction

## Tech Stack

* Python
* Streamlit
* yFinance
* Pandas
* NumPy
* Plotly
* TensorFlow / Keras
* Scikit-learn

## How It Works

The application fetches the last 5 years of closing-price data for the selected stock from Yahoo Finance.

The data is scaled and divided into sequences of 60 days. The LSTM model uses these 60 days to predict the next closing price.

The model is trained on historical data and tested on a separate portion of the data. RMSE and MAE are used to evaluate the predictions.

After training, the model predicts prices for the selected number of future days.

## Run Locally

Clone the repository:

```bash
git clone <your-github-repository-url>
cd Stock-Market
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## Live Application

You can try the deployed application here:

**https://stock-market-predictor1.streamlit.app/**

## Project Structure

```text
Stock-Market/
│
├── app.py
├── requirements.txt
└── README.md
```

## Disclaimer

This project is built for learning and experimentation with machine learning and financial data. Stock price predictions are not guaranteed to be accurate and this application should not be used as financial advice.
