#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
from sklearn import neighbors, datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
get_ipython().system('pip install seaborn')
get_ipython().system('pip install yellowbrick')
import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
np.random.seed(42)


# In[5]:


df = pd.read_csv('C:\\Users\\Admin\\Desktop\\BA\\part3\\data mining\\marketing_campaign.csv', sep='\t')


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.show()


# In[10]:


corr_matrix =df.corr()


# In[11]:


print(corr_matrix)


# In[12]:


df["Year_Birth"].value_counts()


# In[13]:


df["Education"].value_counts()


# In[14]:


df["Marital_Status"].value_counts()


# In[15]:


df["Kidhome"].value_counts()


# In[16]:


df["Teenhome"].value_counts()


# In[17]:


print(df.isnull().sum())


# In[18]:


df=df.dropna(subset=["Income"]) 


# In[19]:


print(df.isnull().sum())


# In[20]:


from pandas.plotting import scatter_matrix
attributes = ["Income", "MntWines", "MntFruits",
 "MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds"]
scatter_matrix(df[attributes], figsize=(12, 8))


# In[23]:


df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
dates = []
for i in df["Dt_Customer"]:
    i = i.date()
    dates.append(i)  
#Dates of the newest and oldest recorded customer
print("The newest customer's enrolment date in therecords:",max(dates))
print("The oldest customer's enrolment date in the records:",min(dates))


# In[25]:


#creation d'une feature "Customer_For"
days = []
d1 = max(dates) #taking it to be the newest customer
for i in dates:
    delta = d1 - i
    days.append(delta)
df["Customer_For"] = days
df["Customer_For"] = pd.to_numeric(df["Customer_For"], errors="coerce")


# In[26]:


df.describe()


# In[27]:


df["Age"] = 2024-df["Year_Birth"]


# In[28]:


df["Spent"] = df["MntWines"]+ df["MntFruits"]+ df["MntMeatProducts"]+ df["MntFishProducts"]+ df["MntSweetProducts"]+ df["MntGoldProds"]

#Deriving living situation by marital status"Alone"
df["Living_With"]=df["Marital_Status"].replace({"Married":"Partner", "Together":"Partner", "Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone",})


# In[29]:


df["Children"]=df["Kidhome"]+df["Teenhome"]


# In[30]:


df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner":2})+ df["Children"]


# In[31]:


df["Is_Parent"] = np.where(df.Children> 0, 1, 0)


# In[32]:


to_drop = ["Marital_Status", "Dt_Customer","Year_Birth", "ID"]
data = df.drop(to_drop, axis=1)


# In[34]:


#Dropping the outliers by setting a cap on Age and income. 
data = data[(data["Age"]<90)]
data = data[(data["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(data))


# In[39]:


#Get list of categorical variables
s = (data.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables in the dataset:", object_cols)


# In[40]:


#Label Encoding the object dtypes.
LE=LabelEncoder()
for i in object_cols:
    data[i]=data[[i]].apply(LE.fit_transform)
    
print("All features are now numerical")


# In[60]:


#Creating a copy of data
ds = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Complain', 'Response']
ds = ds.drop(cols_del, axis=1)
ds_2=data.copy()
#Scaling
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )
scaler.fit(ds_2)
scaled_ds2 = pd.DataFrame(scaler.transform(ds_2),columns= ds_2.columns )
print("All features are now scaled")


# In[61]:


scaled_ds.head()


# In[56]:


#Creating a feature to get a sum of accepted promotions 
data["Total_Promos"] = data["AcceptedCmp1"]+ data["AcceptedCmp2"]+ data["AcceptedCmp3"]+ data["AcceptedCmp4"]+ data["AcceptedCmp5"]


# In[77]:


data.head()


# In[ ]:




