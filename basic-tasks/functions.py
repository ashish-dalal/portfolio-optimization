import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

"""
## A FUNCTION TO RETRIEVE INFORMATION FOR A GIVEN LIST OF STOCKS
def fetch_data(stock_list, start_date, end_date):
    
    if type(stock_list)!=list:
        stock_list = [stock_list]
    
    for i in range(len(stock_list)):
        stock_list[i] = stock_list[i].upper()

    stock_data = {}
    for i in tqdm(range(len(stock_list))):
        ticker = yf.Ticker(stock_list[i])
        stock_data[stock_list[i]] = ticker.history(start=start_date, end=end_date)['Close']
        
    return pd.DataFrame(stock_data)
    """

    ## A FUNCTION TO RETRIEVE INFORMATION FOR A GIVEN LIST OF STOCKS

def fetch_data(stock_list, start_date, end_date, div=True):
    
    """Returns stock_price_df, stock_dividend_df"""
    
    if type(stock_list)!=list:
        stock_list = [stock_list]
    
    for i in range(len(stock_list)):
        stock_list[i] = stock_list[i].upper()

    stock_price = {}
    stock_dividend = {}
    for i in tqdm(range(len(stock_list))):
        ticker = yf.Ticker(stock_list[i])
        stock_data = ticker.history(start=start_date, end=end_date)
        stock_price[stock_list[i]] = stock_data['Close']
        stock_dividend[stock_list[i]] = stock_data['Dividends']
        
    return pd.DataFrame(stock_price), pd.DataFrame(stock_dividend)

## FUNCTION TO CONVERT WEIGHTS TO DATAFRAME

def weights_to_dataframe(arr, cols):
    
    if arr.ndim != 2:
        raise ValueError("Input error must have only 2 dimensions")
    
    for i in range(arr.shape[1]):
        if i==0:
            df = pd.DataFrame(arr[:,i])
        else:
            temp = pd.DataFrame(arr[:,i])
            df = pd.concat([df,temp], axis=1)
    df.columns = cols
    return df

## TO CALCULATE BETA FOR A STOCK

def calculate_beta(log_return_portfolio, log_return_market):
    
    covariance = np.cov(log_return_portfolio, log_return_market.iloc[:,0])[0,1]
    variance_market = log_return_market.var()
    
    beta = covariance/variance_market
    
    return beta[0]

    