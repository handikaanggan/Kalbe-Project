#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install scikit-learn')


# In[95]:


#library untuk pengolahan data
import pandas as pd 
import numpy as np

#library untuk visualisasi data
import matplotlib.pyplot as plt 
import seaborn as sns

#library untuk machine learning model
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot


# In[96]:


#UPLOAD DATA


# In[97]:


#upload data dan define nya sebagai df_
df_transaction = pd.read_csv ('Transaction.csv', delimiter =';')
df_product = pd.read_csv ('Product.csv', delimiter =';')
df_store = pd.read_csv ('Store.csv', delimiter =';')
df_customer = pd.read_csv ('Customer.csv', delimiter =';')


# In[98]:


#untuk mengetahui bentuk dataframe (basis, kolom)
df_transaction.shape, df_store.shape, df_store.shape, df_customer.shape


# In[99]:


df_transaction.head()


# In[100]:


df_store.head()


# In[101]:


df_product.head()


# In[102]:


df_customer.head()


# In[103]:


#mendapatkan informasi data
df_transaction.info()
df_store.info()
df_product.info()
df_customer.info()


# In[104]:


#data cleansing df transaction, kolom Date di ubah menjadi tipe data time menggunakan pd.to_datetime() 
df_transaction ['Date'] = pd.to_datetime(df_transaction ['Date'])


# In[105]:


#data cleansing df customer
df_customer ['Income'] = df_customer ['Income'].replace('[,]', '.', regex = True).astype('float')


# In[106]:


#data cleansing df store
df_store ['Latitude'] = df_store ['Latitude'].replace('[,]', '.', regex = True).astype('float')
df_store ['Longitude'] = df_store ['Longitude'].replace('[,]', '.', regex = True).astype('float')


# In[107]:


#merge data transactiom dan customer didasarkan pada kolom customer id
df_merge = pd.merge (df_transaction, df_customer, on = ['CustomerID'])
#merge data hasil merge dengan product, menghilangkan kolom price biar tidak double
df_merge = pd.merge (df_merge, df_product.drop (columns=['Price']), on =['ProductID'])
#merge data hasil merge 2 dengan store didasarkan pada kolom store id
df_merge = pd.merge (df_merge, df_store, on = ['StoreID'])


# In[108]:


df_merge.head()


# In[109]:


#ambil data yang penting
df_regresi = df_merge.groupby(['Date']).agg({
    'Qty':'sum'
}).reset_index()



# In[110]:


df_regresi


# In[ ]:





# In[111]:


decomposed = seasonal_decompose(df_regresi.set_index('Date'))

plt.figure(figsize=(8,8))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title ('Trend')

plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title ('Seasonality')

plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title ('Residuals')

plt.tight_layout()


# 

# In[112]:


# Split data
cut_off = round(df_regresi.shape[0] * 0.9)
df_train = df_regresi.iloc[:cut_off]
df_test = df_regresi.iloc[cut_off:]



# Cek bentuk (shape) dari data training dan data testing
print

(df_train.shape, df_test.shape)


# In[113]:


df_train


# In[114]:


df_test


# In[115]:


plt.figure(figsize=(20,5))
sns.lineplot(data = df_train, x=df_train['Date'], y=df_train['Qty']);
sns.lineplot(data = df_test, x=df_test['Date'], y=df_test['Qty']);


# In[116]:


#identifikasi pola korelasi data
autocorrelation_plot(df_regresi['Qty']);


# In[119]:


def rmse(y_actual, y_pred):
    
    #Function to calculate Root Mean Squared Error (RMSE)
    
    rmse_value = mean_squared_error(y_actual, y_pred, squared=False)
    print(f'RMSE value: {rmse_value}')

def eval(y_actual, y_pred):
   
    #Function to evaluate machine learning modeling
   
    rmse(y_actual, y_pred)
    mae_value = mean_absolute_error(y_actual, y_pred)
    print(f'MAE value: {mae_value}')


# In[ ]:





# In[118]:


df_train.set_index ('Date')
df_test.set_index ('Date')

y = df_train['Qty']

ARIMAmodel = ARIMA(y, order=(40, 2, 1))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(df_test))
                                     
y_pred_df = y_pred.conf_int()
y_pred_df['predictions'] = ARIMAmodel.predict(start =y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = df_test.index
y_pred_out = y_pred_df['predictions']
eval(df_test['Qty'], y_pred_out)

plt.figure(figsize=(20,5))
plt.plot(df_train['Qty'])
plt.plot(df_test['Qty'], color='red')
plt.plot(y_pred_out, color='black', label= 'Arima Prediction')
plt.legend()
                                     


# In[ ]:





# In[ ]:




