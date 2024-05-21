import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px

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
        if div:
            stock_dividend[stock_list[i]] = stock_data['Dividends']
    if div:    
        return pd.DataFrame(stock_price), pd.DataFrame(stock_dividend)
    else:
        return pd.DataFrame(stock_price)

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

## PLOT STOCK PRICES
#1 single stock

def plot_price(price_data, dividend_data=None, stock_name=None, figure=None):
    
    """Plotly graph, make sure you use plotly elements"""
    
    if figure is None:
        fig = go.Figure()
    else:
        fig = figure
    
    if stock_name is None:
        stock = price_data.name
    else:
        stock = stock_name
    
    ## PLOTTING THE PRICES
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data, mode='lines', name=stock))
    
    if dividend_data is not None:
        ## PLOTTING THE DIVIDEND ROLLOUT
        dividend_rollout_days = dividend_data[dividend_data != 0].index
        price_when_dividend = price_data.loc[dividend_rollout_days]
        dividend_amounts = dividend_data.loc[dividend_rollout_days]
        
        # Constructing hover text without date
        hover_text = [f"Dividend Amount: {amount:.2f}" for amount in dividend_amounts]
        
        fig.add_trace(go.Scatter(x=dividend_rollout_days, y=price_when_dividend, 
                                 mode='markers', marker=dict(symbol='circle-open-dot', size=10), 
                                 showlegend=False, name="{} Dividend".format(stock),
                                 hovertext=hover_text))
    
    fig.update_layout(
        title='Stock Price with Dividend Rollout',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        hovermode="x unified"
        #hoverlabel=dict(bgcolor='white', font=dict(color='black'))
    )
    
    return fig

#2 plot multiple stock_price

def plot_multi_price(df_price, df_dividend, figure=None):
    
    """input dataframes"""
    
    if figure is None:
        fig = go.Figure()
    else:
        fig = figure
    
    for stock in df_price.columns:
        fig = plot_price(df_price[stock], df_dividend[stock], figure=fig)
        
    return fig

## FOR % COMPARISONS

def plot_single_price_comparison(price_data, dividend_data, figure=None, stock=None):
    
    if figure is None:
        fig = go.Figure()
    else:
        fig = figure
    
    price_data = 100*price_data/price_data[0] - 100
        
    fig = plot_price(price_data,dividend_data)
    fig.add_trace(go.Scatter(x=price_data.index, 
                             y=np.ones(len(price_data))*price_data[0], 
                             mode='lines', line=dict(width=2,dash='dash'), showlegend=False))
    return fig

## STACKED HORIZONTAL BAR GRAPH FOR PORTFOLIO WEIGHTS

def plot_weights_horzontal_bar_stacked(weights, stock_price_column, y_label_text='Simulation', figure=None):
    
    if figure is None:
        fig = go.Figure()
    else:
        fig = figure
    
    ## DATA
    x_data = 100*weights[len(weights)::-1]
    y_data = ["{} #{}".format(y_label_text,i) for i in range(len(x_data),0,-1)]
    top_labels = stock_price_column


    # Colors for bars
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.G10 + px.colors.qualitative.T10 + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Dark2 + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel2 + px.colors.qualitative.Set3 + px.colors.qualitative.Antique + px.colors.qualitative.Bold + px.colors.qualitative.Pastel + px.colors.qualitative.Prism + px.colors.qualitative.Safe + px.colors.qualitative.vivid 

    # Add bars
    for i in range(len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]],
                y=[yd],
                orientation='h',
                marker=dict(color=colors[i], line=dict(color='rgb(248, 248, 249)', width=1))
            ))

    # Update layout
    fig.update_layout(
        xaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False, domain=[0.15, 1]),
        yaxis=dict(showgrid=False, showline=False, showticklabels=False, zeroline=False),
        barmode='stack',
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        margin=dict(l=120, r=10, t=140, b=80),
        showlegend=False
    )

    # Add annotations for percentage labels
    annotations = []
    for yd, xd in zip(y_data, x_data):
        annotations.append(dict(xref='paper', yref='y', x=0.14, y=yd, xanchor='right',
                                text=str(yd), font=dict(family='Arial', size=14, color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        space = 0
        for i, val in enumerate(xd):
            annotations.append(dict(xref='x', yref='y', x=space + (val / 2), y=yd,
                                    text=f"{val:.2f}%", font=dict(family='Arial', size=14, color='rgb(248, 248, 255)'),
                                    showarrow=False))
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper', x=space + (val / 2), y=1.1,
                                        text=str(top_labels[i]), font=dict(family='Arial', size=14, color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += val

    fig.update_layout(annotations=annotations)
    
    return fig