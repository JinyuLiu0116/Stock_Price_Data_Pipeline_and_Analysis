import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
import yfinance as yf
import datetime as dt

engine=sqlalchemy.create_engine('mysql+pymysql://root:8551649@localhost:3306/stock_data')

now=dt.datetime.now()
df=yf.download('WFC', start='2024-01-01', end=now)

#df.to_sql('wfc',engine)
max_date=pd.read_sql('SELECT MAX(Date) FROM wfc',engine).values[0][0]

df[df.index > max_date].to_sql('wfc',engine,if_exists='append')

df1=pd.read_sql('wfc',engine)
print(df1)
