import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

#connect to MySQL database
engine = sqlalchemy.create_engine('mysql+pymysql://root:8551649@localhost:3306/stock_data')
# use pandas read data from MySQL stock table
pandas_dataframe = pd.read_sql_table('crwd', engine)

# use pandas to analys data
 
 # The over view of 'CRWD' price data
print(pandas_dataframe)

 # To view the first 5 rows of data
print(pandas_dataframe.head(5))

 # To view the last 5 rows of data
print(pandas_dataframe.tail(5))

 # To view the data from the 'Close' column
print(pandas_dataframe['Close'].tail(10))
 # This data on price changes is what we want to analyze.
