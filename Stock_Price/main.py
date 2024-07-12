import numpy as np
import pandas as pd
#pip install tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt

#Libraries to be used for Model Evaluation and scaling of data

from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_squared_error

#Libraries to be used for the model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

#Used to turn off warnings

import warnings
warnings.filterwarnings('ignore')

#Tensorflow Warning Suppression
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
df = pd.read_csv("C:/Projects/Huge Stock Market Dataset/Stocks/tsla.us.txt")
df.head()
Date	Open	High	Low	Close	Volume	OpenInt
0	2010-06-28	17.00	17.00	17.00	17.00	0	0
1	2010-06-29	19.00	25.00	17.54	23.89	18783276	0
2	2010-06-30	25.79	30.42	23.30	23.83	17194394	0
3	2010-07-01	25.00	25.92	20.27	21.96	8229863	0
4	2010-07-02	23.00	23.10	18.71	19.20	5141807	0
df.tail()
Date	Open	High	Low	Close	Volume	OpenInt
1853	2017-11-06	307.00	307.50	299.01	302.78	6482486	0
1854	2017-11-07	301.02	306.50	300.03	306.05	5286320	0
1855	2017-11-08	305.50	306.89	301.30	304.31	4725510	0
1856	2017-11-09	302.50	304.46	296.30	302.99	5440335	0
1857	2017-11-10	302.50	308.36	301.85	302.99	4621912	0
def check_df(dataframe, head=5):
    print("---------------- Shape ----------------")
    print(dataframe.shape)

    print("---------------- Types ----------------")
    print(dataframe.dtypes)

    print("---------------- Head ----------------")
    print(dataframe.head(head))

    print("---------------- Tail ----------------")
    print(dataframe.tail(head))

    print("---------------- NA ----------------")
    print(dataframe.isnull().sum())

    print("---------------- Quantiles ----------------")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
---------------- Shape ----------------
(1858, 7)
---------------- Types ----------------
Date        object
Open       float64
High       float64
Low        float64
Close      float64
Volume       int64
OpenInt      int64
dtype: object
---------------- Head ----------------
         Date   Open   High    Low  Close    Volume  OpenInt
0  2010-06-28  17.00  17.00  17.00  17.00         0        0
1  2010-06-29  19.00  25.00  17.54  23.89  18783276        0
2  2010-06-30  25.79  30.42  23.30  23.83  17194394        0
3  2010-07-01  25.00  25.92  20.27  21.96   8229863        0
4  2010-07-02  23.00  23.10  18.71  19.20   5141807        0
---------------- Tail ----------------
            Date    Open    High     Low   Close   Volume  OpenInt
1853  2017-11-06  307.00  307.50  299.01  302.78  6482486        0
1854  2017-11-07  301.02  306.50  300.03  306.05  5286320        0
1855  2017-11-08  305.50  306.89  301.30  304.31  4725510        0
1856  2017-11-09  302.50  304.46  296.30  302.99  5440335        0
1857  2017-11-10  302.50  308.36  301.85  302.99  4621912        0
---------------- NA ----------------
Date       0
Open       0
High       0
Low        0
Close      0
Volume     0
OpenInt    0
dtype: int64
---------------- Quantiles ----------------
          0.00        0.05        0.50          0.95          0.99  \
Open     16.14      22.177      184.44  3.295575e+02  3.670842e+02   
High     16.63      23.000      188.66  3.320488e+02  3.718088e+02   
Low       8.03      21.500      181.45  3.232701e+02  3.622350e+02   
Close    15.80      22.207      184.85  3.278490e+02  3.656543e+02   
Volume    0.00  569400.300  3421025.50  1.205163e+07  2.226238e+07   
OpenInt   0.00       0.000        0.00  0.000000e+00  0.000000e+00   

                 1.00  
Open     3.866900e+02  
High     3.896100e+02  
Low      3.793450e+02  
Close    3.850000e+02  
Volume   3.714989e+07  
OpenInt  0.000000e+00  
df["Date"]=pd.to_datetime(df["Date"])
df.head()
Date	Open	High	Low	Close	Volume	OpenInt
0	2010-06-28	17.00	17.00	17.00	17.00	0	0
1	2010-06-29	19.00	25.00	17.54	23.89	18783276	0
2	2010-06-30	25.79	30.42	23.30	23.83	17194394	0
3	2010-07-01	25.00	25.92	20.27	21.96	8229863	0
4	2010-07-02	23.00	23.10	18.71	19.20	5141807	0
Preparing the Data
stock_df=df[["Date","Close"]]
stock_df.head()
Date	Close
0	2010-06-28	17.00
1	2010-06-29	23.89
2	2010-06-30	23.83
3	2010-07-01	21.96
4	2010-07-02	19.20
print("Min. Date - ", stock_df["Date"].min())
print("Max. Date - ", stock_df["Date"].max())
Min. Date -  2010-06-28 00:00:00
Max. Date -  2017-11-10 00:00:00
stock_df.index=stock_df["Date"]
stock_df
Date	Close
Date		
2010-06-28	2010-06-28	17.00
2010-06-29	2010-06-29	23.89
2010-06-30	2010-06-30	23.83
2010-07-01	2010-07-01	21.96
2010-07-02	2010-07-02	19.20
...	...	...
2017-11-06	2017-11-06	302.78
2017-11-07	2017-11-07	306.05
2017-11-08	2017-11-08	304.31
2017-11-09	2017-11-09	302.99
2017-11-10	2017-11-10	302.99
1858 rows × 2 columns

stock_df.drop("Date",axis=1,inplace=True)
stock_df
Close
Date	
2010-06-28	17.00
2010-06-29	23.89
2010-06-30	23.83
2010-07-01	21.96
2010-07-02	19.20
...	...
2017-11-06	302.78
2017-11-07	306.05
2017-11-08	304.31
2017-11-09	302.99
2017-11-10	302.99
1858 rows × 1 columns

result_df=stock_df.copy()

plt.figure(figsize=(12,6))
plt.plot(stock_df["Close"],color="blue")
plt.ylabel("Stock Price")
plt.xlabel("Time")
plt.title("Tesla Stock Price")
plt.show()

Converting to Numpy Array

stock_df=stock_df.values
stock_df[0:5]
array([[17.  ],
       [23.89],
       [23.83],
       [21.96],
       [19.2 ]])
Defining float 32 for Neural Network

stock_df=stock_df.astype("float32")
Let's make Train-Test distinction as a function

def split_data(dataframe,test_size):
    pos=int(round(len(dataframe)*(1-test_size)))
    train=dataframe[:pos]
    test=dataframe[pos:]
    return train,test,pos

train,test,pos=split_data(stock_df,0.20)
print(train.shape,test.shape)
(1486, 1) (372, 1)
scaler_train=MinMaxScaler(feature_range=(0,1))
train=scaler_train.fit_transform(train)
train[0:5]
array([[0.00444049],
       [0.02993634],
       [0.02971432],
       [0.02279454],
       [0.01258141]], dtype=float32)
scaler_test=MinMaxScaler(feature_range=(0,1))
test=scaler_test.fit_transform(test)
test[0:5]
array([[0.17912066],
       [0.187325  ],
       [0.21454191],
       [0.20432329],
       [0.2052567 ]], dtype=float32)
def create_features(data,lookback):
    x,y=[],[]
    for i in range(lookback,len(data)):
        x.append(data[i-lookback:i,0])
        y.append(data[i,0])
    return np.array(x),np.array(y)

lookback=30
Train & Test Data Set

x_train,y_train=create_features(train,lookback)
x_test,y_test=create_features(test,lookback)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
(1456, 30) (1456,) (342, 30) (342,)
There are 30 features in total here, so we added the day 30 days before the relevant date to the data set

x_train[0:5]
array([[0.00444049, 0.02993634, 0.02971432, 0.02279454, 0.01258141,
        0.00114713, 0.        , 0.00614268, 0.00592066, 0.00462551,
        0.00865897, 0.01494967, 0.01513469, 0.01791   , 0.02260953,
        0.01665186, 0.01635583, 0.01924215, 0.02031527, 0.01905713,
        0.01757696, 0.01820603, 0.01683688, 0.01531971, 0.01894612,
        0.02275755, 0.02020426, 0.01720693, 0.01402457, 0.01406157],
       [0.02993634, 0.02971432, 0.02279454, 0.01258141, 0.00114713,
        0.        , 0.00614268, 0.00592066, 0.00462551, 0.00865897,
        0.01494967, 0.01513469, 0.01791   , 0.02260953, 0.01665186,
        0.01635583, 0.01924215, 0.02031527, 0.01905713, 0.01757696,
        0.01820603, 0.01683688, 0.01531971, 0.01894612, 0.02275755,
        0.02020426, 0.01720693, 0.01402457, 0.01406157, 0.01195234],
       [0.02971432, 0.02279454, 0.01258141, 0.00114713, 0.        ,
        0.00614268, 0.00592066, 0.00462551, 0.00865897, 0.01494967,
        0.01513469, 0.01791   , 0.02260953, 0.01665186, 0.01635583,
        0.01924215, 0.02031527, 0.01905713, 0.01757696, 0.01820603,
        0.01683688, 0.01531971, 0.01894612, 0.02275755, 0.02020426,
        0.01720693, 0.01402457, 0.01406157, 0.01195234, 0.00777087],
       [0.02279454, 0.01258141, 0.00114713, 0.        , 0.00614268,
        0.00592066, 0.00462551, 0.00865897, 0.01494967, 0.01513469,
        0.01791   , 0.02260953, 0.01665186, 0.01635583, 0.01924215,
        0.02031527, 0.01905713, 0.01757696, 0.01820603, 0.01683688,
        0.01531971, 0.01894612, 0.02275755, 0.02020426, 0.01720693,
        0.01402457, 0.01406157, 0.01195234, 0.00777087, 0.00666074],
       [0.01258141, 0.00114713, 0.        , 0.00614268, 0.00592066,
        0.00462551, 0.00865897, 0.01494967, 0.01513469, 0.01791   ,
        0.02260953, 0.01665186, 0.01635583, 0.01924215, 0.02031527,
        0.01905713, 0.01757696, 0.01820603, 0.01683688, 0.01531971,
        0.01894612, 0.02275755, 0.02020426, 0.01720693, 0.01402457,
        0.01406157, 0.01195234, 0.00777087, 0.00666074, 0.00932504]],
      dtype=float32)
y_train[0:5]
array([0.01195234, 0.00777087, 0.00666074, 0.00932504, 0.01102723],
      dtype=float32)
Stock Closing Prices

x_train=np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
x_test=np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
(1456, 1, 30) (1456, 1) (342, 1, 30) (342, 1)
Modeling
model=Sequential()
model.add(LSTM(units=100,
               activation="relu",
               input_shape=(x_train.shape[1],lookback)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.summary()
Model: "sequential_10"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_10 (LSTM)              (None, 100)               52400     
                                                                 
 dropout_10 (Dropout)        (None, 100)               0         
                                                                 
 dense_10 (Dense)            (None, 1)                 101       
                                                                 
=================================================================
Total params: 52,501
Trainable params: 52,501
Non-trainable params: 0
_________________________________________________________________
model.compile(loss="mean_squared_error",optimizer="adam")
callbacks=[EarlyStopping(monitor="val_loss",patience=3,verbose=1,mode="min"),ModelCheckpoint(filepath="mymodel.h5",monitor="val_loss",mode="min",save_best_only=True,save_weights_only=False,verbose=1)]
history = model.fit(x=x_train,
                    y=y_train,
                    epochs=200,
                    batch_size=30,
                    validation_data=(x_test,y_test),
                    callbacks=callbacks,
                    shuffle=False)
Epoch 1/200
35/49 [====================>.........] - ETA: 0s - loss: 0.0057    
Epoch 1: val_loss improved from inf to 0.00898, saving model to mymodel.h5
49/49 [==============================] - 2s 9ms/step - loss: 0.0083 - val_loss: 0.0090
Epoch 2/200
35/49 [====================>.........] - ETA: 0s - loss: 0.0044
Epoch 2: val_loss improved from 0.00898 to 0.00815, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0065 - val_loss: 0.0082
Epoch 3/200
36/49 [=====================>........] - ETA: 0s - loss: 0.0034    
Epoch 3: val_loss improved from 0.00815 to 0.00747, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0052 - val_loss: 0.0075
Epoch 4/200
47/49 [===========================>..] - ETA: 0s - loss: 0.0053
Epoch 4: val_loss improved from 0.00747 to 0.00692, saving model to mymodel.h5
49/49 [==============================] - 0s 5ms/step - loss: 0.0056 - val_loss: 0.0069
Epoch 5/200
46/49 [===========================>..] - ETA: 0s - loss: 0.0042
Epoch 5: val_loss improved from 0.00692 to 0.00635, saving model to mymodel.h5
49/49 [==============================] - 0s 5ms/step - loss: 0.0047 - val_loss: 0.0064
Epoch 6/200
39/49 [======================>.......] - ETA: 0s - loss: 0.0037    
Epoch 6: val_loss improved from 0.00635 to 0.00555, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0048 - val_loss: 0.0056
Epoch 7/200
40/49 [=======================>......] - ETA: 0s - loss: 0.0032    
Epoch 7: val_loss improved from 0.00555 to 0.00525, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0041 - val_loss: 0.0053
Epoch 8/200
38/49 [======================>.......] - ETA: 0s - loss: 0.0028    
Epoch 8: val_loss improved from 0.00525 to 0.00487, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0035 - val_loss: 0.0049
Epoch 9/200
37/49 [=====================>........] - ETA: 0s - loss: 0.0026    
Epoch 9: val_loss improved from 0.00487 to 0.00463, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0039 - val_loss: 0.0046
Epoch 10/200
42/49 [========================>.....] - ETA: 0s - loss: 0.0038    
Epoch 10: val_loss improved from 0.00463 to 0.00450, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0044 - val_loss: 0.0045
Epoch 11/200
45/49 [==========================>...] - ETA: 0s - loss: 0.0033    
Epoch 11: val_loss did not improve from 0.00450
49/49 [==============================] - 0s 3ms/step - loss: 0.0036 - val_loss: 0.0048
Epoch 12/200
44/49 [=========================>....] - ETA: 0s - loss: 0.0028    
Epoch 12: val_loss improved from 0.00450 to 0.00406, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0031 - val_loss: 0.0041
Epoch 13/200
47/49 [===========================>..] - ETA: 0s - loss: 0.0031    
Epoch 13: val_loss improved from 0.00406 to 0.00395, saving model to mymodel.h5
49/49 [==============================] - 0s 4ms/step - loss: 0.0032 - val_loss: 0.0040
Epoch 14/200
43/49 [=========================>....] - ETA: 0s - loss: 0.0032    
Epoch 14: val_loss did not improve from 0.00395
49/49 [==============================] - 0s 4ms/step - loss: 0.0034 - val_loss: 0.0040
Epoch 15/200
23/49 [=============>................] - ETA: 0s - loss: 2.5507e-04
Epoch 15: val_loss improved from 0.00395 to 0.00383, saving model to mymodel.h5
49/49 [==============================] - 0s 3ms/step - loss: 0.0034 - val_loss: 0.0038
Epoch 16/200
41/49 [========================>.....] - ETA: 0s - loss: 0.0024    
Epoch 16: val_loss improved from 0.00383 to 0.00377, saving model to mymodel.h5
49/49 [==============================] - 0s 5ms/step - loss: 0.0029 - val_loss: 0.0038
Epoch 17/200
37/49 [=====================>........] - ETA: 0s - loss: 0.0025    
Epoch 17: val_loss did not improve from 0.00377
49/49 [==============================] - 0s 4ms/step - loss: 0.0034 - val_loss: 0.0039
Epoch 18/200
34/49 [===================>..........] - ETA: 0s - loss: 0.0017    
Epoch 18: val_loss did not improve from 0.00377
49/49 [==============================] - 0s 4ms/step - loss: 0.0031 - val_loss: 0.0038
Epoch 19/200
24/49 [=============>................] - ETA: 0s - loss: 2.0378e-04
Epoch 19: val_loss did not improve from 0.00377
49/49 [==============================] - 0s 4ms/step - loss: 0.0032 - val_loss: 0.0039
Epoch 19: early stopping
plt.figure(figsize=(20,5))
plt.subplot(1,2,2)
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.legend(loc="upper right")
plt.xlabel("Epoch",fontsize=16)
plt.ylabel("Loss",fontsize=16)
plt.ylim([0,max(plt.ylim())])
plt.title("Training and Validation Loss",fontsize=16)
plt.show()

The expression loss is the mean squared error value, that is, the mean squared expression of the actual values and the estimated values

Evaluation
loss=model.evaluate(x_test,y_test,batch_size=30)
print("\nTest loss:%.1f%%"%(100.0*loss))
12/12 [==============================] - 0s 1ms/step - loss: 0.0039

Test loss:0.4%
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)
46/46 [==============================] - 0s 1ms/step
11/11 [==============================] - 0s 0s/step
train_predict=scaler_train.inverse_transform(train_predict)
test_predict=scaler_test.inverse_transform(test_predict)
y_train=scaler_train.inverse_transform(y_train)
y_test=scaler_test.inverse_transform(y_test)
RMSE value to train dataset

train_rmse=np.sqrt(mean_squared_error(y_train,train_predict))
test_rmse=np.sqrt(mean_squared_error(y_test,test_predict))
print(f"Train RMSE - {train_rmse}")
print(f"Test RMSE - {test_rmse}")
Train RMSE - 9.909297943115234
Test RMSE - 12.756081581115723
The mistake Tesla Stock Market will be made in the next period is 13 dollars

train_prediction_data=result_df[lookback:pos]
train_prediction_data["Predicted"]=train_predict
train_prediction_data.head()
Close	Predicted
Date		
2010-08-10	19.03	22.340857
2010-08-11	17.90	22.261044
2010-08-12	17.60	22.241150
2010-08-13	18.32	22.225958
2010-08-16	18.78	22.224825
test_prediction_data=result_df[pos+lookback:]
test_prediction_data["Predicted"]=test_predict
test_prediction_data.head()
Close	Predicted
Date		
2016-07-07	215.94	212.811951
2016-07-08	216.78	213.702942
2016-07-11	224.78	213.510590
2016-07-12	224.65	215.208267
2016-07-13	222.53	217.051941
plt.figure(figsize=(14,5))
plt.plot(result_df,label="Real Values")
plt.plot(train_prediction_data["Predicted"],color="blue",label="Train Predicted")
plt.plot(test_prediction_data["Predicted"],color="red",label="Test Predicted")
plt.xlabel("Time")
plt.ylabel("Stock Values")
plt.legend()
plt.show()

