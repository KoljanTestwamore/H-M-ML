#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time


# In[2]:


from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(data))

    fig = plt.figure(1, figsize=(10,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


# ## Articles data

# In[3]:


articles = pd.read_csv('articles.csv.zip')


# In[4]:


articles[:5]


# In[5]:


articles.info()


# In[6]:


articles.index_name.unique()


# In[7]:


articles.shape


# In[8]:


show_wordcloud(articles['detail_desc'], 'Wordcloud from product name')


# Seems like each variables "_no" or "_id" or "_code" correspond to the "_name" <br>
# Drop all column with "_no", "_id", "_code" to prevent the machine thinking that "1" is more than "2", except "articel_id"

# In[9]:


articles1 = articles.iloc[:, 1:]

articles1.drop(articles1.columns[articles1.columns.str.contains('_no|_code|id')], axis=1, inplace=True)

articles2 = pd.concat([articles1, articles['article_id']], axis =1)
articles2


# In[10]:


# Check for missing value

articles2.isnull().sum()


# In[11]:


# drop column "detail desc". Other column already give almost the same description already
articles2.drop('detail_desc', axis= 1, inplace= True)
articles2.head(10)


# In[12]:


# There are three redundant columns: 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name'
# Checking the value of each column to see if they are the same or similar --> may delete if thet are similar

print(articles2['colour_group_name'].value_counts())
print(articles2['perceived_colour_value_name'].value_counts())
print(articles2['perceived_colour_master_name'].value_counts())


# In[13]:


# they all explain the color in a similar way 
# possibly when customer search, the result will be 'perceived_colour_value_name' + 'perceived_colour_master_name' = 'colour_group_name
# then no use for 'colour_group_name'

articles2.drop('colour_group_name', axis=1, inplace=True)
articles2.head()


# In[14]:


# See if "index_name" and "index_group_name" are redundant
print(articles2['index_name'].value_counts())
print(articles2['index_group_name'].value_counts())


# In[15]:


# "index_name" is more than "index_group_name" and explain things the same, so delete "index_name"
articles2.drop('index_name', axis=1, inplace=True)
articles2.head()


# In[16]:


#'garment_group_name' and department_name are the same, so remove department_name
articles2.drop('department_name', axis=1, inplace=True)
articles2.head()


# In[17]:


temp = articles.groupby(['product_group_name'])['product_type_name'].nunique()
df = pd.DataFrame({'Product group': temp.index,
                   'Product types number': temp.values
                  })
df = df.sort_values(['Product types number'], ascending=False)
plt.figure(figsize = (8,6))
plt.title('Number of Product types per each Product group')
s = sns.barplot(x = 'Product group', y='Product types number', data=df)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
locs, labels = plt.xticks()
plt.show()


# In[18]:


temp = articles.groupby(['perceived_colour_value_name'])['article_id'].nunique()
df = pd.DataFrame({'Perceived colour value name': temp.index,
                   'Num of articles': temp.values
                  })
df = df.sort_values(['Num of articles'], ascending=False)
plt.figure(figsize = (12,6))
plt.title(f'Number of Articles per each Perceived colour value Name')
s = sns.barplot(x = 'Perceived colour value name', y='Num of articles', data=df)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
locs, labels = plt.xticks()
plt.show()


# In[19]:


temp = articles.groupby(['perceived_colour_master_name'])['article_id'].nunique()
df = pd.DataFrame({'Perceived colour master name': temp.index,
                   'Num of articles': temp.values
                  })
df = df.sort_values(['Num of articles'], ascending=False)
plt.figure(figsize = (12,6))
plt.title(f'Number of Articles per each Perceived colour master Name')

s = sns.barplot(x = 'Perceived colour master name', y='Num of articles', data=df)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
locs, labels = plt.xticks()
plt.show()


# In[20]:


temp = articles2.groupby(["product_group_name"])["product_type_name"].nunique()
df = pd.DataFrame({'Product Group': temp.index,
                   'Product Types': temp.values
                  })
df = df.sort_values(['Product Types'], ascending=False)
plt.figure(figsize = (12,6))
plt.title(f'Number of Product Types per each Product Group')

s = sns.barplot(x = 'Product Group', y='Product Types', data=df)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
locs, labels = plt.xticks()
plt.show()


# In[21]:


temp = articles2.groupby(["index_group_name"])["article_id"].nunique()
df = pd.DataFrame({'Index Group Name': temp.index,
                   'Articles': temp.values
                  })
df = df.sort_values(['Articles'], ascending=False)
plt.title(f'Number of Articles per each Index Group Name')

s = sns.barplot(x = 'Index Group Name', y='Articles', data=df)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
locs, labels = plt.xticks()
plt.show()


# In[22]:


temp = articles2.groupby(["product_type_name"])["article_id"].nunique()
df = pd.DataFrame({'Product Type': temp.index,
                   'Articles': temp.values
                  })
total_types = len(df['Product Type'].unique())
df = df.sort_values(['Articles'], ascending=False)[0:50]
plt.figure(figsize = (12,6))
plt.title(f'Number of Articles per each Product Type')

s = sns.barplot(x = 'Product Type', y='Articles', data=df)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
locs, labels = plt.xticks()
plt.show()


# In[23]:


_, ax = plt.subplots(figsize=(15, 7))
ax = sns.histplot(data=articles2, y='garment_group_name', color='orange', 
                  hue='index_group_name', multiple='stack')
ax.set_xlabel('Count by garment group')
ax.set_ylabel('Garment group')
plt.show()


# ## Customers data

# In[24]:


customers = pd.read_csv('customers.csv.zip')
customers[:10]


# <ul>
# <li>Cannot find how to read the postal_code in the given format --> plan to group based on city or country, but no luck trying to read it, so will drop 'postal_code'.</li>
# <li>Seems like 'FN' = if a customer get Fashion News newsletter</li>
# <li>Check if there is more value than 'Regulary' and 'NONE' -> if only these two then will drop 'fashion_news_frequency' as FN is already binary</li>
# <li>Have check the value for both 'club_member_status and 'Active' first</li>
# </ul>

# In[25]:


customers.info()


# In[26]:


customers.shape


# In[27]:


# delete 'postal_code'
customers.drop('postal_code', axis= 1, inplace= True)
customers.head()


# In[28]:


customers.customer_id = customers.customer_id.apply(lambda x: int(x[-16:],16) ).astype('int64')
customers.target = customers.target.astype('int8')

print(f'There are {len(targets.customer_ID.unique())} unique short train targets')


# In[29]:


customers[:10]


# In[30]:


# Dealing with missing value
customers['FN'] = customers['FN'].fillna(0)
customers['Active'] = customers['Active'].fillna(0)


# In[31]:


# Check for missing value
print(customers['club_member_status'].value_counts())
print('number of missing value =', customers['club_member_status'].isnull().sum())


# In[32]:


# Dealing with missing value
customers['club_member_status'] = customers['club_member_status'].fillna(customers['club_member_status'].mode()[0])
print(customers['club_member_status'].value_counts())
print('number of missing value =', customers['club_member_status'].isnull().sum())


# In[33]:


# Check for missing value
print(customers['fashion_news_frequency'].value_counts())
print('number of missing value =', customers['fashion_news_frequency'].isnull().sum())


# In[34]:


# Dealing with missing value
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].replace('None', 'NONE')

# replacing missing value with mode
customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna(customers['fashion_news_frequency'].mode()[0])

# Check for missing value
print(customers['fashion_news_frequency'].value_counts())
print('number of missing value =', customers['fashion_news_frequency'].isnull().sum())


# In[35]:


# Dealing with missing value
customers.dropna(axis = 0, inplace=True)
customers.isnull().sum()


# In[36]:


sns.set_style('darkgrid')

_, ax = plt.subplots(figsize=(10,5))
ax = sns.histplot(data=customers, x='age', bins=100, color='green')
ax.set_xlabel('Distribution of the customers age')
plt.show()


# In[37]:


customers.to_csv('customers_short.csv.zip', index=False, compression={'method': 'zip', 'archive_name': 'train_labels_short.csv'})


# In[38]:


import os

file_name1 = "customers.csv.zip"
file_name2 = "customers_short.csv.zip"

file_stats1 = os.stat(file_name1)
file_stats2 = os.stat(file_name2)
print(f'Size of customers.csv.zip: {file_stats1.st_size / (1024 * 1024)}')
print(f'Size of customers_short.csv.zip: {file_stats2.st_size / (1024 * 1024)}')


# ## Transactions data

# In[39]:


transactions = pd.read_csv('transactions_train.csv.zip')


# In[40]:


transactions.head()


# In[41]:


merged = transactions[['customer_id', 'article_id', 
                                   'price', 't_dat']].merge(articles[['article_id', 'prod_name', 
                                                                      'product_type_name', 'product_group_name', 
                                                                      'index_name']], on='article_id', how='left')


# In[42]:


articles_index = merged[['index_name', 'price']].groupby('index_name').mean()
sns.set_style('darkgrid')
_, ax = plt.subplots(figsize=(10,5))
ax = sns.barplot(x=articles_index.price, y=articles_index.index, color='green', alpha=0.8)
ax.set_xlabel('Price by index')
ax.set_ylabel('Index')
plt.show()


# In[43]:


from datetime import datetime

grouped = transactions.sample(200000).groupby(['t_dat', 'sales_channel_id'])['article_id'].count().reset_index()
grouped['t_dat'] = grouped['t_dat'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

grouped.columns = ['Date', 'Sales Channel Id', "Transactions"]

_, ax = plt.subplots(1, 1, figsize=(16,6))
g1 = ax.plot(grouped.loc[grouped['Sales Channel Id']==1, 'Date'], 
             grouped.loc[grouped['Sales Channel Id']==1, 'Transactions'], label='Sales Channel 1')

g2 = ax.plot(grouped.loc[grouped["Sales Channel Id"]==2, 'Date'], 
             grouped.loc[grouped["Sales Channel Id"]==2, 'Transactions'], label='Sales Channel 2')

plt.xlabel('Date')
plt.ylabel('Num of transactions')
ax.legend()
plt.title(f'Number of transactions per day, grouped by Sales channel (200k sample)')
plt.show()


# In[44]:


transactions = transactions.sample(n=10000, random_state=0)


# In[45]:


transactions.shape


# In[46]:


base_path = ''

transactions_path = f'{base_path}transactions_train.csv.zip'
customers_path = f'{base_path}customers.csv.zip'
articles_path = f'{base_path}articles.csv.zip'

transactions = pd.read_csv(transactions_path, dtype={'article_id': str}, parse_dates=['t_dat'])

customers = pd.read_csv(customers_path)
articles = pd.read_csv(articles_path, dtype={'article_id': str})

ALL_USERS = customers['customer_id'].unique().tolist()
ALL_ITEMS = articles['article_id'].unique().tolist()

user_to_customer_map = {customer_id: user_id for user_id, customer_id in enumerate(ALL_USERS)}
item_to_article_map = {article_id: item_id for item_id, article_id in enumerate(ALL_ITEMS)}

del customers, articles

transactions


# In[47]:


print(f'There are {len(ALL_USERS)} unique users the dataset.')
print(f'There are {len(ALL_ITEMS)} unique items the dataset.')


# In[48]:


START_DATE = '2020-08-21'

transactions_small = transactions[transactions['t_dat'] > START_DATE].copy()


# In[49]:


transactions_small.loc[:,'user_id'] = transactions_small['customer_id'].map(user_to_customer_map)
transactions_small.loc[:,'item_id'] = transactions_small['article_id'].map(item_to_article_map)
transactions_small


# In[50]:


print(f'There are {len(transactions_small.user_id.unique())} unique users with purchases after {START_DATE}.')
print(f'There are {len(transactions_small.item_id.unique())} unique items in the smaller dataset.')


# In[1]:


transactions['t_dat'].min(), transactions['t_dat'].max()

test_days = 30

test_cut = transactions['t_dat'].max() - pd.Timedelta(test_days)

train = transactions[transactions['t_dat'] < test_cut]
test = transactions[transactions['t_dat'] >= test_cut]


# In[ ]:


from scipy.sparse import coo_matrix

users = transactions_small['user_id'].values
items = transactions_small['item_id'].values

purchases = np.ones(transactions_small.shape[0])

csr_train = coo_matrix((purchases, (users, items))).tocsr()
csr_train


# In[ ]:


print(csr_train[0])


# In[ ]:


pip install implicit


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from implicit.als import AlternatingLeastSquares\n\nmodel = AlternatingLeastSquares(factors=7, iterations=2,\n                                regularization=0.01, \n                                random_state=42)\nmodel.fit(csr_train)')


# In[ ]:


from implicit.evaluation import mean_average_precision_at_k

map_12 = mean_average_precision_at_k(model, csr_train, csr_train, K=12, num_threads=4)
map_12


# In[ ]:


user_id = 1019194
N = 12

recommendations = model.recommend(user_id, csr_train[user_id], N=N, filter_already_liked_items=False)
dict(zip(recommendations[0], recommendations[1]))


# In[ ]:


item_id = 97632

related_items = model.similar_items(item_id, N=5)
dict(zip(related_items[0], related_items[1]))


# In[ ]:





# In[ ]:


transactions_small.head()


# In[ ]:


transactions_small_sales_1 = transactions_small[transactions_small['sales_channel_id']==1]


# In[ ]:


transactions_small_sales_1.head()


# In[ ]:


START_DATE = '2020-06-21'

transactions_small_sales_chanel_1 = transactions_small_sales_1[transactions_small_sales_1['t_dat'] > START_DATE].copy()


# In[ ]:


transactions_small_sales_chanel_1.loc[:,'user_id'] = transactions_small_sales_chanel_1['customer_id'].map(user_to_customer_map)
transactions_small_sales_chanel_1.loc[:,'item_id'] = transactions_small_sales_chanel_1['article_id'].map(item_to_article_map)
transactions_small_sales_chanel_1


# In[ ]:


print(f'There are {len(transactions_small_sales_chanel_1.user_id.unique())} unique users with purchases after {START_DATE}.')
print(f'There are {len(transactions_small_sales_chanel_1.item_id.unique())} unique items in the smaller dataset.')


# In[ ]:


transactions['t_dat'].min(), transactions['t_dat'].max()

test_days = 90

test_cut = transactions['t_dat'].max() - pd.Timedelta(test_days)

train = transactions[transactions['t_dat'] < test_cut]
test = transactions[transactions['t_dat'] >= test_cut]


# In[ ]:


from scipy.sparse import coo_matrix

users = transactions_small_sales_chanel_1['user_id'].values
items = transactions_small_sales_chanel_1['item_id'].values

purchases = np.ones(transactions_small_sales_chanel_1.shape[0])

csr_train = coo_matrix((purchases, (users, items))).tocsr()
csr_train


# In[ ]:


print(csr_train[0])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from implicit.als import AlternatingLeastSquares\n\nmodel = AlternatingLeastSquares(factors=7, iterations=2,\n                                regularization=0.01, \n                                random_state=42)\nmodel.fit(csr_train)')


# In[ ]:


from implicit.evaluation import mean_average_precision_at_k

map_12 = mean_average_precision_at_k(model, csr_train, csr_train, K=12, num_threads=4)
map_12


# In[ ]:


user_id = 1369338
N = 12

recommendations = model.recommend(user_id, csr_train[user_id], N=N, filter_already_liked_items=False)
dict(zip(recommendations[0], recommendations[1]))


# In[ ]:


item_id = 81825

related_items = model.similar_items(item_id, N=5)
dict(zip(related_items[0], related_items[1]))


# In[ ]:





# In[ ]:


transactions_small_sales_2 = transactions_small[transactions_small['sales_channel_id']==2]


# In[ ]:


transactions_small_sales_2.head()


# In[ ]:


START_DATE = '2020-06-21'

transactions_small_sales_chanel_2 = transactions_small_sales_2[transactions_small_sales_2['t_dat'] > START_DATE].copy()


# In[ ]:


transactions_small_sales_chanel_2.loc[:,'user_id'] = transactions_small_sales_chanel_2['customer_id'].map(user_to_customer_map)
transactions_small_sales_chanel_2.loc[:,'item_id'] = transactions_small_sales_chanel_2['article_id'].map(item_to_article_map)
transactions_small_sales_chanel_2


# In[ ]:


print(f'There are {len(transactions_small_sales_chanel_1.user_id.unique())} unique users with purchases after {START_DATE}.')
print(f'There are {len(transactions_small_sales_chanel_1.item_id.unique())} unique items in the smaller dataset.')


# In[ ]:




