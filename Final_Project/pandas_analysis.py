import pandas as pd
import sqlalchemy
import numpy as np
from sqlalchemy import create_engine

#connect to MySQL database
engine = sqlalchemy.create_engine('mysql+pymysql://root:8551649@localhost:3306/stock_data')
# use pandas read data from MySQL stock table
df = pd.read_sql_table('crwd', engine)
#over view of data
print(df)
print(df.columns)
print(df.describe())

# ensure the format is correct and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df1=df['Close']
# set the range of date that I want to look at
start=pd.to_datetime("2024-01-02").date()
end=pd.to_datetime("2024-07-19").date()
df1.loc[start:end]
print(df1.index)

#to see different percent of price
print(df1.describe(percentiles=[0.1,0.3,0.7,0.9]))


#to see the change of price
df1=df1.tail(15)
stock_return=np.log(df1).diff()
print(stock_return) 





















# start=pd.Timestamp('2024-07-05')
# end=pd.Timestamp('2024-07-19')
# pandas_dataframe=pandas_dataframe.loc[start,end]
#df=df.round({'Open':4,'High':4,'Low':4,'Close':4,'Adj Close':4})

# show data
