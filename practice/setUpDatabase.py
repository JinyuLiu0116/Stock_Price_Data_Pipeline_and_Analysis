import sqlalchemy
import pymysql
import sqlalchemy.exc

pymysql.install_as_MySQLdb()
indices = ['WFC','IBM','AAPL']#array

#a function create schema
def schemarcreator(index):
        engine = sqlalchemy.create_engine('mysql://root:8551649@localhost:3306/')
        with engine.connect() as connection:# use connect()function execute 'execute' attribute.
            connection.execute(sqlalchemy.schema.CreateSchema(index))
        print(f"Schema '{index}' created successfully.")
        
for index in indices:#for loop to execute schemarcreator function
    schemarcreator(index)