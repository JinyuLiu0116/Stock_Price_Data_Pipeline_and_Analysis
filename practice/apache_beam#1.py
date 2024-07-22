import apache_beam as beam
import requests
import time

#in apache beam, everything is pipeline
pipe=beam.Pipeline()
#read data from source: file, API, and act
pi=(
    pipe #'|'is apply function
    |beam.io.ReadAllFromText("here is for csv file",skip_header_lines=True)
    |beam.Map(lambda x:x.split(",")) # map is used for transformation, here is apply',' to split data into list
    |beam.Filter(lambda x:x[1]=='data') # in this line, x is a list
    |beam.combiners.Count.Globally()# combin all elements in subsystem and count the total(Globally())
    |beam.Map(print)#this the will print out the information this pipeline has selected
)
#every pipeline is completed, use run function to run
pipe.run()

#to simplefly the code, we can write the same pipeline use 'with'
#in 'with', the code will execute and close
with beam.Pipeline() as pipe1:
    pi1=(pipe1
    |beam.io.ReadAllFromText("here is for csv file",skip_header_lines=True)
    |beam.Map(lambda x:x.split(",")) 
    |beam.Filter(lambda x:x[1]=='data') 
    |beam.combiners.Count.Globally()
    |beam.Map(print)
)# we combin three steps into one step by writing pipeline in 'with'