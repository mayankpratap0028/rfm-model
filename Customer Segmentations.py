#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler


# In[2]:


orc = pd.read_excel("Online Retail.xlsx")


# In[3]:


orc.head()


# In[4]:


orc.isna().sum()


# In[5]:


orc['Description'] = orc['Description'].fillna('Unknown')


# In[6]:


orc.dropna(inplace=True)


# In[7]:


orc.duplicated().sum()


# In[8]:


orc.drop_duplicates(keep='last',inplace=True)


# In[9]:


orc.shape


# In[10]:


orc.info()


# In[11]:


orc.describe()


# In[12]:


plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
plt.hist(orc['Quantity'])
plt.title("Histogram of Quantity")

plt.subplot(1,2,2)
sns.boxplot(data=orc, x='Quantity')
plt.title("Boxplot of Quantity")

plt.show()


# In[13]:


plt.figure(figsize=(16,5))

plt.subplot(1,2,1)
plt.hist(orc['UnitPrice'])
plt.title("Histogram of Unitprice")

plt.subplot(1,2,2)
sns.boxplot(data=orc, x='UnitPrice')
plt.title("Boxplot of unit price")

plt.show()


# In[16]:


orc['Country'].value_counts().iloc[:10].plot(kind='bar')


# In[42]:


orc[orc['InvoiceNo'].str.contains("C",na=False)].shape[0]


# In[52]:


import datetime as dt


# In[56]:


# Recency
orc['daypassed']  = orc['InvoiceDate'].max() - orc['InvoiceDate']
orc['daypassed'] = orc['daypassed'].dt.days
recency = orc.groupby(['CustomerID',])['daypassed'].min().to_frame()
recency.head()


# In[62]:


frequency = orc.groupby('CustomerID')['InvoiceNo'].count().to_frame()
frequency.head()


# In[59]:


orc['SalesAmount'] = orc['Quantity']* orc['UnitPrice']


# In[63]:


monetary = orc.groupby('CustomerID')['SalesAmount'].sum().to_frame()
monetary.head()


# In[67]:


rfm = recency.merge(frequency,how = 'inner',on='CustomerID').merge(monetary,how ='inner',on='CustomerID')
rfm.head()


# In[69]:


clusters = [2,3,4,5,6,7,8]

for n_clusters in clusters:
    kmean = KMeans(n_clusters=n_clusters,random_state=5)
    cluster_labels = kmean.fit_predict(rfm)
    silhouette_avg = silhouette_score(rfm,cluster_labels)
    print("for n_cluster = ",n_clusters, "Average Silhouette Score is = ",silhouette_avg)


# In[70]:


# for cluster 4
kmeans = KMeans(n_clusters=4,random_state=5)
cluster_labels = kmeans.fit_predict(rfm)


# In[71]:


rfm['Cluster'] = cluster_labels


# In[73]:


rfm['Cluster'].value_counts()


# In[74]:


rfm.groupby('Cluster').mean()


# In[75]:


pca = make_pipeline(StandardScaler(),
                   PCA(n_components=2,random_state=5))
rfm_transformed = pca.fit_transform(rfm)


# In[77]:


rfm_transformed


# In[84]:


plt.figure(figsize=(10,8))
sns.scatterplot(x=rfm_transformed[:,0],y= rfm_transformed[:,1],hue=cluster_labels)
plt.title("Clusters",fontsize=14)
plt.xlabel("1st PC")
plt.ylabel("2nd PC")


# In[ ]:




