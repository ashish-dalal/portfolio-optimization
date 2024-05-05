import pandas as pd
import numpy as np
import yfinance as yf

## A FUNCTION TO RETRIEVE INFORMATION FOR A GIVEN LIST OF STOCKS
def fetch_data(stock_list, start_date, end_date):
    
    if type(stock_list)!=list:
        stock_list = [stock_list]
    
    for i in range(len(stock_list)):
        stock_list[i] = stock_list[i].upper()

    stock_data = {}
    for stock in stock_list:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
        
    return pd.DataFrame(stock_data)