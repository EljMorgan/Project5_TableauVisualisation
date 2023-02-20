import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import datetime as dt

# Set dfs
customers_df= pd.read_csv('Dataset/olist_customers_dataset.csv')
geolocation_df= pd.read_csv("Dataset/olist_geolocation_dataset.csv")
items_df= pd.read_csv('Dataset/olist_order_items_dataset.csv')
payments_df= pd.read_csv('Dataset/olist_order_payments_dataset.csv')
reviews_df= pd.read_csv('Dataset/olist_order_reviews_dataset.csv')
orders_df= pd.read_csv('Dataset/olist_orders_dataset.csv')
products_df= pd.read_csv('Dataset/olist_products_dataset.csv')
sellers_df= pd.read_csv('Dataset/olist_sellers_dataset.csv')
category_translation_df= pd.read_csv('Dataset/product_category_name_translation.csv')

#customers_df.info()

#print(reviews_df.isnull().sum())

datasets = [customers_df, geolocation_df, items_df, payments_df, reviews_df, orders_df, products_df, sellers_df, category_translation_df]
titles = ["customers", "geolocation", "items", "payments", "reviews", "orders", "products", "sellers", "category_translation"]

data_summary = pd.DataFrame({},)
data_summary['datasets'] = titles
data_summary['columns'] = [', '.join([col for col in data.columns]) for data in datasets]
data_summary['total_rows'] = [data.shape[0] for data in datasets]
data_summary['total_cols'] = [data.shape[1] for data in datasets]
data_summary['total_duplicate'] = [data.duplicated().sum() for data in datasets]
data_summary['total_null'] = [data.isnull().sum().sum() for data in datasets]
data_summary['null_cols'] = [', '.join([col for col, null in data.isnull().sum().items() if null>0])
for data in datasets]
cm = sns.color_palette("blend:white,green", as_cmap=True)
data_summary.style.background_gradient(cmap=cm)


for i in datasets:
    i.dropna(inplace=True)
    i.drop(i[i.duplicated()].index, axis=0, inplace=True)

datasets = [customers_df, geolocation_df, items_df, payments_df, reviews_df, orders_df, products_df, sellers_df, category_translation_df]
titles = ["customers", "geolocation", "items", "payments", "reviews", "orders", "products", "sellers", "category_translation"]

data_summary = pd.DataFrame({},)
data_summary['datasets']= titles
data_summary['columns'] = [', '.join([col for col, null in data.isnull().sum().items() ]) for data in datasets]
data_summary['total_rows']= [data.shape[0] for data in datasets]
data_summary['total_cols']= [data.shape[1] for data in datasets]
data_summary['total_duplicate']= [len(data[data.duplicated()]) for data in datasets]
data_summary['total_null']= [data.isnull().sum().sum() for data in datasets]
data_summary['null_cols'] = [', '.join([col for col, null in data.isnull().sum().items() if null > 0]) for data in datasets]
data_summary.style.background_gradient(cmap='YlGnBu')

merged_df= pd.merge(customers_df, orders_df, on="customer_id")
merged_df= merged_df.merge(reviews_df, on="order_id")
merged_df= merged_df.merge(items_df, on="order_id")
merged_df= merged_df.merge(products_df, on="product_id")
merged_df= merged_df.merge(payments_df, on="order_id")
merged_df= merged_df.merge(sellers_df, on='seller_id')
merged_df= merged_df.merge(category_translation_df, on='product_category_name')
print(merged_df.shape)
#print(merged_df.head(10))

#print(merged_df.isnull().sum())

time_columns= ['order_purchase_timestamp', 'order_approved_at','order_delivered_carrier_date','order_delivered_customer_date',
               'order_estimated_delivery_date', 'review_creation_date', 'review_answer_timestamp', 'shipping_limit_date']

sp_data=merged_df[merged_df.customer_state=='SP']
merged_df[time_columns]=merged_df[time_columns].apply(pd.to_datetime)

###########        RFM  Analysis ###############
#The “RFM” in RFM analysis stands for recency, frequency and monetary value. RFM analysis is a way to use data based 
# on existing customer behavior to predict how a new customer is likely to act in the future.

# 1. Recency value: This refers to the amount of time since a customer’s last interaction with a brand
# 2. Frequency value: This refers to the number of times a customer has made a purchase or otherwise
#  interacted with your brand during a particular period of time
# 3.  Monetary value: This refers to the total amount a customer has spent purchasing products and services from 
# your brand over a particular period of time.
present_day = merged_df['order_purchase_timestamp'].max()+ dt.timedelta(days=2)

print("Present day: ",present_day)
print("Latest date in dataset: ", merged_df['order_purchase_timestamp'].max())

recency_df= pd.DataFrame(merged_df.groupby(by='customer_unique_id', as_index=False)['order_purchase_timestamp'].max())
print(recency_df.head())

rfm= merged_df.groupby('customer_unique_id').agg({'order_purchase_timestamp': lambda date: (present_day - date.max()).days,
                                                  'order_id': lambda num: len(num), 
                                                  'payment_value': lambda price: price.sum()})
# Rename the columns 
rfm.columns=['Recency','Frequency','Monetary']
rfm['Recency'] = rfm['Recency'].astype(int)
rfm['Monetary'] = rfm['Monetary'].astype(int)
print(rfm.head())

#### The graph below depicts information about the distribution of Recency (how recently),
#  Frequency(how often), Monetary(how much they spent).
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1); sns.distplot(rfm['Recency'])
plt.subplot(3, 1, 2); sns.distplot(rfm['Frequency'])
plt.subplot(3, 1, 3); sns.distplot(rfm['Monetary'])
#plt.show()

##################
####################" OUTLIERS"
rfm.plot(
    kind='box', 
    subplots=True, 
    sharey=False, 
    figsize=(10, 6)
)

# increase spacing between subplots
plt.subplots_adjust(wspace=0.5) 
#plt.show()

fig, axs = plt.subplots(1, 3, figsize=(14, 7))

sns.histplot(data=rfm, x="Recency", kde=True, color="skyblue", ax=axs[0])
sns.histplot(data=rfm, x="Frequency", kde=True, color="olive", ax=axs[1])
sns.histplot(data=rfm, x="Monetary", kde=True, color="gold", ax=axs[2])

#plt.show()

# Calculate Z scores to normalize the data
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(rfm))
#print(z)

################################################333
########## Removing Outliers ##################
rfm_clean = rfm[(z < 3).all(axis=1)]
#print(rfm.shape)

# Create box plots to check for outliers
rfm_clean.plot(
    kind='box', 
    subplots=True, 
    sharey=False, 
    figsize=(10, 6)
)

# increase spacing between subplots
plt.subplots_adjust(wspace=0.5) 
#plt.show()

fig, axs = plt.subplots(1, 3, figsize=(14, 7))

sns.histplot(data=rfm_clean, x="Recency", kde=True, color="skyblue", ax=axs[0])
sns.histplot(data=rfm_clean, x="Frequency", kde=True, color="olive", ax=axs[1])
sns.histplot(data=rfm_clean, x="Monetary", kde=True, color="gold", ax=axs[2])

#plt.show()

#################################################
#########" RFM SCORE" ###########################
# Use quintiles to to make 5 equal parts based on the available values. Each quintiles contains 20% of the population. 
quintiles = rfm_clean[['Recency', 'Frequency', 'Monetary']].quantile([.2, .4, .6, .8]).to_dict()
print(quintiles)

def r_score(x):
    if x <= quintiles['Recency'][.2]:
        return 5
    elif x <= quintiles['Recency'][.4]:
        return 4
    elif x <= quintiles['Recency'][.6]:
        return 3
    elif x <= quintiles['Recency'][.8]:
        return 2
    else:
        return 1
    
def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5  

# Calculate RFM score for each customer

rfm_clean['R'] = rfm_clean['Recency'].apply(lambda x: r_score(x))
rfm_clean['F'] = rfm_clean['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
rfm_clean['M'] = rfm_clean['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))

# Combine the scores
rfm_clean['RFM Score'] = rfm_clean['R'].map(str) + rfm_clean['F'].map(str) + rfm_clean['M'].map(str)
print(rfm_clean.head())

# Created 6 segments based on R and F scores

segments = {
    '[1-2][1-4]': 'at risk',
    '[1-2]5': 'can\'t loose',
    '3[1-3]': 'needs attention',
    '[3-4][4-5]': 'loyal customers',
    '[4-5]1': 'new customers',
    '[4-5][2-5]': 'champions'
    
}

rfm_clean['Segment'] = rfm_clean['R'].map(str) + rfm_clean['F'].map(str)
rfm_clean['Segment'] = rfm_clean['Segment'].replace(segments, regex=True)
print(rfm_clean.head())

# count the number of customers in each segment
segments_counts = rfm_clean['Segment'].value_counts().sort_values(ascending=True)

print(segments_counts)

# Distribution of Segments
fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='#11399c')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['champions', 'loyal customers']:
            bar.set_color('green')
        if segments_counts.index[i] in ['at risk', 'needs attention']:
            bar.set_color('#df4947')
        if segments_counts.index[i] in ['new customers']:
            bar.set_color('#ede35c')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )

plt.show()