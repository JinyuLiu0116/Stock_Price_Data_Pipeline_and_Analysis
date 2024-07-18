import apache_beam as beam
from apache_beam,options.pipeline_options import PipelineOptions
import csv
import mysql.connector


class ReadFromCSV(beam.DoFn):#reading data from csv file
    def __init__(self, file_path): #constructor method that initializes the class with csv file

        def process(self, element):
            with open(self.file_path, 'r') as csvfile: #opens the csv file
                reader=csv.DictReader(csvfile) #create csv reader object
                for row in reader:# next stage of the pipeline
                yield row

                class TransformatData(beam.DoFn):#manupulating and formating
                    def process(self,element):#process each element(row) and format
                        formatted_data={
                            'symbol':element['symbol'],
                            'timestamp':element['timstamp'],
                            'open':element['open'],
                            'high':element['high'],
                            'low':element['low'],
                            'close':element['close'],
                            'volume':element['volume'],
                        }
                        return [formatted_data]#return  the formatted datat as list

                    class WriteToMySQL(beam.DoFn):
                        def __init__(self,host,database,user,password):#database connection parameters
                            self.host=host
                            self.database=database
                            self.user=user
                            self.password=password 

                            def start_bundle(self): #this method establish a connection to the Mysql database and prepare an inser statement before processing data
                            
                                self.connection=mysql.connector.connect(  #connects to the mysql database
                                    host=self.host,
                                    database=self.database,
                                    user=self.user,
                                    password=self.password                           
                                )
                                self.cursor=self.connection.cursor()#create to navigate object to excute sql statement
                                    self.insert_statement=( #prepare sql inser statement
                                        "INSERT INTO stock_prices"
                                        "(sysmbol, timestamp,open, high,low,close,volume)"
                                        "VALUES (%(symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)"
                                    )

                                    def process(self,element):
                                        self.cursor.execute(self.insert_statement,element)#excute the insert data
                                        self.connection.commit() # commit the transcation to save the change

                                        def finish_bundkle(self): #close the database connection and cursor after processing data
                                            self.cursor.close()
                                            self.connection.close()

                                            def run():
                                                file_path="https://raw.githubusercontent.com/JinyuLiu0116/Stock_Price_Data_Pipeline_and_Analysis/main/daily_IBM.csv?token=GHSAT0AAAAAACUKGJGDABWDBMJGQ5RGBEZIZUYTE3A"
                                                
                                                options=PipelineOptions()
                                                with beam.Pipeline( options= options) as p:#create and run hte pipeline
                                                    (
                                                        p
                                                        |'start'>>beam.create([None])#start the pipleine
                                                        |'Read from CSV'>>beam.ParDo(ReadFromCSV(file_path))#read the data from csv file
                                                        |'Trasform and format data'>>beam.ParDo(TransformatData())
                                                        |'Write to MySQL'>>beam.ParDo(WriteToMySQL(
                                                            host='',
                                                            database='stock_data',
                                                            user='root'
                                                            password='root'
                                                        ))
                                                    )

                                                    if __name__== '__main__':
                                                        run

                                
