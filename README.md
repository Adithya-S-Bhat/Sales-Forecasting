# Sales-Analysis-Using-ML

Sales forecasting is the process of estimating future sales. Accurate sales forecasts enable companies to make informed business decisions and predict short-term and long-term performance. 
This project uses two Machine Learning models to predict sales. One model predicts sales using Artificial Neural Network and the other one is SARIMAX. The reppective files are as follows:
1. project.py -> SARIMA

2. salesann.py -> Artificial Neural Network

# DataSet
* The dataset used is a Superstore dataset.
* It has different parameters like Shipment mode, Profit , Discount, Quantity,etc
Label encoding , One Hot Encoder is used to convert categorical data to numerical data.
* Feature Scaling is done to give equal weightage to all the parameters.

# SARIMA
SARIMA is used for a non-stationary series of dataset.It can identify the trend and seasonality of the data unlike ANN,is a forecasting algorithm based on the idea that the information in the past values of the time series can alone be used to predict the future values.

SARIMA stands for Seasonal Autoregressive Integrated Moving Average
* Auto regressive(AR) model can be either simple,multiple or non-linear regression.
* Differentiation step I eliminates non-stationarity
* Moving average(MA) model uses weighting factors,where the recent observations are given higher weight.To identify the recent trend in the sales
* Seasonal ARIMA can support a time series without a repeating cycle  

SARIMA Equation:
![image](https://user-images.githubusercontent.com/56101205/120184816-077fce80-c22f-11eb-9d62-de63a0f3a171.png)


