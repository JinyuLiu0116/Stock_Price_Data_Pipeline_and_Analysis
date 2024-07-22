import pandas as pd
import sqlalchemy
import numpy as np
from sqlalchemy import create_engine

#connect to MySQL database
engine = sqlalchemy.create_engine('mysql+pymysql://root:8551649@localhost:3306/stock_data')
# use pandas read data from MySQL stock table
pandas_dataframe = pd.read_sql_table('crwd', engine)

# use pandas to analys data
 
 # To view the last 5 rows of data
print(pandas_dataframe.tail(10))

 # To see the change of price
CRWD_close=pandas_dataframe['Close']
CRWD_return=np.log(CRWD_close).diff()
print(CRWD_return.tail(10))

 # To view the data from the 'Close' column
print(pandas_dataframe['Close'].tail(10))

 # To see describe of the data we want to analysis
CRWD_close=CRWD_close.tail(10)
print(CRWD_close.describe())




















# start=pd.Timestamp('2024-07-05')
# end=pd.Timestamp('2024-07-19')
# pandas_dataframe=pandas_dataframe.loc[start,end]
#df=df.round({'Open':4,'High':4,'Low':4,'Close':4,'Adj Close':4})

# show data
