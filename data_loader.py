import yfinance as yf
import pandas as pd
import numpy as np

def load_data(tickers, start):

    if isinstance(tickers, str):
        tickers = [tickers]
    data = yf.download(tickers, start=start, progress=False)
    if isinstance(data, pd.DataFrame):
        if 'Adj Close' in data.columns:
            price_df = data['Adj Close']
        elif 'Close' in data.columns:
            price_df = data['Close']
        else:
            raise KeyError("Ni 'Adj Close' ni 'Close' n'ont été trouvés dans les données téléchargées.")
    else:
        raise ValueError("Les données téléchargées ne sont pas au format DataFrame.")

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame(name=tickers[0])

    returns = price_df.pct_change().dropna()

    vol = returns.std() * np.sqrt(252)

    latest_price = price_df.iloc[-1]

    return price_df, vol, latest_price

