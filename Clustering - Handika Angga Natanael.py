#!/usr/bin/env python
# coding: utf-8

# In[45]:


#library untuk pengolahan data
import pandas as pd 
import numpy as np

#library untuk visualisasi data
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[46]:


#upload data dan set dataframe
df_transaction = pd.read_csv ('Transaction.csv', delimiter =';')
df_product = pd.read_csv ('Product.csv', delimiter =';')
df_store = pd.read_csv ('Store.csv', delimiter =';')
df_customer = pd.read_csv ('Customer.csv', delimiter =';')


# In[47]:


#untuk mengetahui bentuk dataframe (basis, kolom)
df_transaction.shape, df_store.shape, df_store.shape, df_customer.shape


# In[48]:


df_transaction.head()


# In[49]:


df_store.head()


# In[50]:


df_product.head()


# In[51]:


df_customer.head()


# In[52]:


#mendapatkan informasi data type
df_transaction.info()
df_store.info()
df_product.info()
df_customer.info()


# In[83]:


#data cleansing df transaction, kolom Date di ubah menjadi tipe data datetime menggunakan pd.to_datetime() 
df_transaction ['Date'] = pd.to_datetime(df_transaction ['Date'])


# In[54]:


#data cleansing df customer, mengubah koma menjadi titik
df_customer ['Income'] = df_customer ['Income'].replace('[,]', '.', regex = True).astype('float')


# In[55]:


#data cleansing df store, mengubah koma menjadi titik
df_store ['Latitude'] = df_store ['Latitude'].replace('[,]', '.', regex = True).astype('float')
df_store ['Longitude'] = df_store ['Longitude'].replace('[,]', '.', regex = True).astype('float')


# In[56]:


#merge data transactiom dan customer didasarkan pada kolom customer id
df_merge = pd.merge (df_transaction, df_customer, on = ['CustomerID'])
#merge data hasil merge dengan product, menghilangkan kolom price biar tidak double
df_merge = pd.merge (df_merge, df_product.drop (columns=['Price']), on =['ProductID'])
#merge data hasil merge 2 dengan store didasarkan pada kolom store id
df_merge = pd.merge (df_merge, df_store, on = ['StoreID'])


# In[57]:


df_merge


# In[ ]:





# In[58]:


#mengambil data yang penting
df_cluster = df_merge.groupby(['CustomerID']).agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()


# In[59]:


df_cluster


# In[60]:


#drop data yang tidak diperlukan
data_cluster = df_cluster.drop(columns = 'CustomerID')
#normalisasi data
data_normalize = preprocessing.normalize(data_cluster)


# In[61]:


data_normalize


# In[62]:


#algoritma K-Means dengan menggunakan metode "Elbow" dan metode "Silhouette Score"
K = range (2, 8)
fits = []
score = []

for k in K:
    model = KMeans (n_clusters = k, random_state = 0, n_init = 'auto').fit(data_normalize)
    
    fits.append(model)
    
    score.append(silhouette_score(data_normalize, model.labels_, metric = 'euclidean'))


# In[63]:


sns.lineplot(x = K, y = score);


# In[64]:


#menambah kolom baru cluster model yang isinya fits dari hasil kmeans
df_cluster['cluster_label'] = fits[2].labels_


# In[79]:


df_segmen = df_cluster.groupby (('cluster_label')).agg({
    'CustomerID' : 'count',
    'TransactionID' : 'mean',
    'Qty' : 'mean',
    'TotalAmount' : 'mean'

}).reset_index()


# In[80]:


df_segmen


# In[82]:


#grafik interpretasi data
sns.lmplot(x='Qty', y='TotalAmount', data=df_segmen, fit_reg=False, hue='cluster_label', height=5, palette='Dark2')
plt.title('Quantity vs TotalAmount', fontsize=16)
plt.xlabel('Quantity', fontsize=13)
plt.ylabel('TotalAmount', fontsize=13)
plt.show()


# In[ ]:





# In[ ]:




