#!pip install pandas
#!pip install numpy
#!pip install matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. basic graph
x = [1,2,3,4,5] #array for x-asis
y = [0,20,40,60,80] #array for y-asis
# x y need to have same dimension

plt.figure(figsize=(5,3), dpi=150)# concle the size of graph and sould be excueted befor plot()

plt.plot(x,y, color='green', label='IBM',linewidth=2,marker='.', markersize = 5) # I run matplotlib as plt, so i need to add plt. befor plot. 
#color can change the color of graph line
 

plt.title('Stock Price Graph')# add title to the graph
plt.xlabel('Date') # label below the graph
#plt.xlabel('date', fontdict={'fontname':'Comic San MS','fontsize':20}) change fontdict and size
plt.ylabel('Price $')# label on the left hand side of the graph

plt.xticks([1,2,3,4,5])# keep number as index withou decimal

plt.show() #this can remove header from the graph

