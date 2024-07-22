import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

#connect to MySQL database
engine = sqlalchemy.create_engine('mysql+pymysql://root:8551649@localhost:3306/stock_data')
# use pandas read data from MySQL stock table
pandas_dataframe = pd.read_sql_table('crwd', engine)

# use pandas to analys data
 
 # To view the last 5 rows of data
print(pandas_dataframe.tail(5))

 # To see the change of price
CRWD_close=pandas_dataframe['Close']
CRWD_return=np.log(CRWD_close).diff()
print(CRWD_return.tail(10))

 # To view the data from the 'Close' column
print(pandas_dataframe['Close'].tail(10))

 # To see describe of the data we want to analysis
CRWD_close=CRWD_close.tail(10)
print(CRWD_close.describe())
