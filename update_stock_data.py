#!/usr/bin/env python
# coding: utf-8

# In[34]:


# Suppress pandas warnings 
import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd

import sqlite3

import yahooquery  
import yfinance as yf

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.svm import LinearSVC
from sklearn import preprocessing 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


GET_NEW_DATA = False


# In[12]:


# Relative Strength Index (RSI)
def RSI(df, window=14):
    Gain=df['Close'].copy()
    Loss=df['Close'].copy()
    Avg_gain=df['Close'].copy()
    Avg_loss=df['Close'].copy()
    rsi=df['Close'].copy()

    Gain[:]=0.0
    Loss[:]=0.0
    Avg_gain[:]=0.0
    Avg_loss[:]=0.0
    rsi[:]=0.0
    
    #idx = df['Close']>df['Close'].shift(1)

    for i in range(1,len(df)):
        if df.loc[i,'Close']>df.loc[i-1,'Close']:
            Gain[i]=df.loc[i,'Close']-df.loc[i-1,'Close']
        else:
            # For loss save the absolute value on loss
            Loss[i]=abs(df.loc[i,'Close']-df.loc[i-1,'Close'])
        if i>window:
            Avg_gain[i]=(Avg_gain[i-1]*(window-1)+Gain[i])/window
            Avg_loss[i]=(Avg_loss[i-1]*(window-1)+Loss[i])/window
            rsi[i]=(100*Avg_gain[i]/(Avg_gain[i]+Avg_loss[i])).round(2)
            
    # 50-day simple moving average
    sma_50 = df['Close'].rolling(window=50).mean().round(2)
    
    # 200-day simple moving average
    sma_200 = df['Close'].rolling(window=200).mean()
    
    # 9-day exponential moving average (aka signal line for MACD indicator)
    ema_9 = df['Close'].ewm(span=9, adjust=False).mean().round(2)
    
    # 26-day EMA
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    # 12-day EMA
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    
    # MACD = 12-Period EMA − 26-Period EMA
    macd = (ema_12 - ema_26)
    
    # MACD histogram: Difference between MACD line and singal line
    # Positive values as green bars
    # Negative values as red bars
    diff_macd = macd - ema_9
    
    
    return rsi, sma_50, sma_200, ema_9, macd, diff_macd

def get_technical_data(symbols, con):

    # Drop table if exists, to prevent overwriting
    con.execute('DROP table IF EXISTS stock_technical_data')
    
    for symbol in symbols:

        stock = yf.Ticker(symbol)
        df = stock.history(period='2500d', actions=False) # minimum 100 days of data for good RSI accuracy
        df = df.reset_index()
        df.drop(df.columns.difference(['Date','Close']), 1, inplace=True)

        rsi, sma_50, sma_200, ema_9, macd, diff_macd = RSI(df)

        # Add RSI to the dataframe
        df['RSI'] = rsi

        # Add 9-day EMA to the dataframe
        df['EMA_9'] = ema_9

        # Add MACD to the dataframe
        df['MACD'] = macd.round(2)

        # Add signal to the dataframe
        df['signal'] = macd.ewm(span=9, adjust=False).mean().round(2)

        # Add diff_MACD to the dataframe
        df['diff_MACD'] = (df['MACD'] - df['signal'])

        # Add ticker symbol to the dataframe
        df['ticker'] = symbol

        # Write dataframe to database
        df.to_sql("stock_technical_data", con, if_exists="append")

def get_fundamental_data(symbols, con):

    # Drop table if exists, to prevent overwriting
    con.execute('DROP table IF EXISTS stock_fundamental_data')    
    
    for symbol in symbols:

    stock = yf.Ticker(symbol)
    
    # Get revenue and earnings data
    df_earnings = stock.earnings
    
    # Get R&D and Income Before Tax data
    df_financials = stock.financials
    try:
        df_financials.columns = df_financials.columns.year # Change to year only format
    except:
        print('symbol '+symbol+' issue')
        continue
    
    df_financials = df_financials.T # transpose for years as rows
    # Drop all columns except R&D and IBT
    df_financials.drop(df_financials.columns.difference(['Research Development', 'Income Before Tax']), 1, inplace=True)
    df_financials = df_financials.iloc[::-1] # Reverse the order
    
    # Merge the dataframes
    df = df_earnings.merge(df_financials, how='outer', left_index=True, right_index=True)
    
    # Add ticker symbol to the dataframe
    df['ticker'] = symbol
    
    # Write dataframe to database
    try:
        df.to_sql("stock_fundamental_data", con, if_exists="append")
    except:
        print('Unable to write '+symbol+' to database')
        continue
        
def get_financial_data(symbols):
    sp500 = yahooquery.Ticker(symbols, asynchronous=False)
    sp500_all_financial_data = sp500.all_financial_data()
    sp500_all_financial_data['ResearchAndDevelopmentPerShare'] = sp500_all_financial_data['ResearchAndDevelopment']/sp500_all_financial_data['BasicAverageShares']
    sp500_all_financial_data['TotalRevenuePerShare'] = sp500_all_financial_data['TotalRevenue']/sp500_all_financial_data['BasicAverageShares']
    sp500_all_financial_data['DebtToEquity'] = sp500_all_financial_data['TotalDebt']/sp500_all_financial_data['StockholdersEquity']

    sp500_financial_data = sp500_all_financial_data[['asOfDate',
                                                     'BasicAverageShares',
                                                     'BasicEPS', 
                                                     'ResearchAndDevelopmentPerShare',
                                                     'TotalRevenuePerShare',
                                                     'DebtToEquity']]

    sp500_financial_data = sp500_financial_data.reset_index()
    
    return sp500_financial_data


# In[13]:


#get symbols
symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].values.tolist()

con = sqlite3.connect('stock_data_500.db')

if GET_NEW_DATA:
    get_technical_data(symbols, con)
    get_fundamental_data(symbols, con)


# In[14]:


df_tech = pd.read_sql("""SELECT * FROM stock_technical_data""",con)
df_tech['Date'] = pd.to_datetime(df_tech['Date'])


# In[15]:


if GET_NEW_DATA:
    sp500_financial_data = get_finanical_data(symbols)
    df_all = df_tech.merge(
        sp500_financial_data,
        how = 'left',
        left_on=['ticker','Date'],
        right_on=['symbol','asOfDate']
    )
    df_all['Price/Earnings Ratio'] = df_all['Close']/df_all['BasicEPS']
    df_all['Price/Revenue Ratio'] = df_all['Close']/df_all['TotalRevenuePerShare']
    df_all['Research/Revenue Ratio'] = df_all['ResearchAndDevelopmentPerShare']/df_all['TotalRevenuePerShare']
    
    df_all = df_all.drop(
        columns=[
            'symbol',
            'asOfDate',
            'BasicAverageShares',
            'BasicEPS', 
            'ResearchAndDevelopmentPerShare',
            'TotalRevenuePerShare',
        ]
    )
    
    df_all.to_sql("stock_all_financial_data", con, if_exists="replace")



# In[17]:


df_all = pd.read_sql("""SELECT * FROM stock_all_financial_data""",con)
df = df_all.copy()
df2 = df_all.copy()


# In[23]:


# Create the input dataframe with RSI and MACD
df['return10']=(df['Close'].shift(-10) - df['Close'])*100/df['Close']

# Remove rows with RSI <10
df = df[df['RSI']>=10]

# Drop all rows with Nan
df = df.dropna(subset=['return10'])

x=df[['RSI','diff_MACD']].values

df['y10'] = df['return10'].apply(lambda x:1 if x>0 else 0)


# In[24]:


# Create the input dataframe with RSI and MACD
df2['return10']=(df2['Close'].shift(-10) - df2['Close'])*100/df2['Close']

# Remove rows with RSI <10
df2 = df2[df2['RSI']>=10]

# Drop all rows with Nan
df2 = df2.dropna(subset=['return10','DebtToEquity',
       'Price/Earnings Ratio','Price/Revenue Ratio',
       'Research/Revenue Ratio'])

x2=df2[['RSI','diff_MACD','DebtToEquity',
       'Price/Earnings Ratio','Price/Revenue Ratio',
       'Research/Revenue Ratio']].values

df2['y10'] = df2['return10'].apply(lambda x:1 if x>0 else 0)


# In[35]:


# Using 10-day returns to create final logistic regression model
y=df['y10'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
logmodel


predictions=logmodel.predict(X_test)
print('Logistic Regression Accuracy score of 10 day returns is: '+str(accuracy_score(y_test,predictions)))

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

predictions=knn.predict(X_test)
print('KNN Accuracy score of 10 day returns is: '+str(accuracy_score(y_test,predictions)))


linsvc = LinearSVC()
scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
linsvc.fit(X_train,y_train)

predictions=linsvc.predict(scaler.transform(X_test))
print('Linear Support Vector Classification Accuracy score of 10 day returns is: '+str(accuracy_score(y_test,predictions)))


# In[36]:


# Using 10-day returns to create final logistic regression model
y=df2['y10'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(x2, y, test_size=0.3, random_state=42)
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
logmodel

predictions=logmodel.predict(X_test)
print('Logistic Regression Accuracy score of 10 day returns is: '+str(accuracy_score(y_test,predictions)))

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

predictions=knn.predict(X_test)
print('KNN Accuracy score of 10 day returns is: '+str(accuracy_score(y_test,predictions)))


linsvc = LinearSVC()
scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
linsvc.fit(X_train,y_train)

predictions=linsvc.predict(scaler.transform(X_test))
print('Linear Support Vector Classification Accuracy score of 10 day returns is: '+str(accuracy_score(y_test,predictions)))


# In[ ]:





# In[58]:


last_day = df_all['Date'].max()
df_last_day = df_all[df_all['Date'] == last_day]
df_last_day = df_last_day.dropna(subset=['RSI','diff_MACD','DebtToEquity',
       'Price/Earnings Ratio','Price/Revenue Ratio',
       'Research/Revenue Ratio'])

x_last_day = df_last_day[['RSI','diff_MACD','DebtToEquity',
       'Price/Earnings Ratio','Price/Revenue Ratio',
       'Research/Revenue Ratio']]

predictions=logmodel.predict(x_last_day)
#print(predictions)
df_last_day['prob'] = logmodel.predict_proba(x_last_day)[:,1]


# In[59]:


df_last_day = df_last_day.sort_values('prob', ascending=False)
df_top_ten = df_last_day.iloc[0:10][['ticker',  'prob']]
df_top_ten.to_sql("top_ten_list", con, if_exists="replace")


# In[61]:


df_top_ten


# In[60]:


#close connection
con.close()


# In[ ]:




