import yfinance as yf
import pandas as pd
import numpy as np

def load_data(tickers_input, start=None):

    if isinstance(tickers_input, dict):
        name_to_symbol = tickers_input
        symbols = list(name_to_symbol.values())
    elif isinstance(tickers_input, str):
        name_to_symbol = {tickers_input: tickers_input}
        symbols = [tickers_input]
    elif isinstance(tickers_input, (list, tuple, set)):
        name_to_symbol = {sym: sym for sym in tickers_input}
        symbols = list(tickers_input)
    else:
        raise TypeError("`tickers_input` doit être un str, une liste, un tuple ou un dict.")

    data = yf.download(symbols, start=start, progress=False)

    if isinstance(data, pd.DataFrame):
        if 'Adj Close' in data.columns:
            price_df = data['Adj Close']
        elif 'Close' in data.columns:
            price_df = data['Close']
        else:
            raise KeyError("Ni 'Adj Close' ni 'Close' disponibles.")
    else:
        raise ValueError("Les données ne sont pas un DataFrame.")

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame(name=symbols[0])

    returns = price_df.pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    latest_price = price_df.iloc[-1]

    if isinstance(tickers_input, dict):
        rename_map = {v: k for k, v in name_to_symbol.items()}
        price_df.columns = [rename_map[s] for s in price_df.columns]
        vol.index = [rename_map[s] for s in vol.index]
        latest_price.index = [rename_map[s] for s in latest_price.index]

    rows = []
    for ticker in price_df.columns:
        row = {
            'Stock': ticker,
            'Spot (€)': round(latest_price[ticker], 2),
            'Annualized Volatility (%)': round(vol[ticker] * 100, 2)
        }
        rows.append(row)
    stocks = pd.DataFrame(rows)

    return price_df, vol, latest_price, stocks
