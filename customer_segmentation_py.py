
# coding: utf-8

# # CUSTOMER SEGMENTATION - STEP-BY-STEP TUTORIAL

# ##### We will consider an case study for hotel booking companies like say booking.com, agoda.com, etc. #####
# objective:- To identify on how the companies can improve their overall booking
# My Procedure:-
# 1: Identify the underperforming & overperforming segments
# 2: Focus on tailoring new marketing campaigns for different cities in the coming few months ahead.
# 3: Focus on how the user booking rate can be improved?
# # Data Description 
# 'date_time': Date of booking
# 'user_location_country', 'user_location_region','user_location_city':  user related geographic information 
# 'orig_destination_distance': destination info
# 'is_mobile': whether cust. has booked using mobile or not
# 'is_package': whether cust. has opted for any package
# 'channel': channel used by customer
# 'srch_ci': check-in date
# 'srch_co': check-out date
# 'srch_adults_cnt': num of persons
# 'srch_children_cnt': children count
# 'srch_rm_cnt': num of rooms
# 'is_booking': whether customer has booked or not  

# In[123]:

# let's get started:-
__author__ = 'Ajay Arunachalam'

# import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl
from scipy import stats
from sklearn.externals.six import StringIO
from sklearn import preprocessing
from sklearn import cluster, tree, decomposition
import matplotlib.pyplot as plt
import pydot
get_ipython().magic('matplotlib inline')


# In[124]:

# set display right
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)


# In[125]:

# Load the booking dataset
try:
    sample = pd.read_csv('sample', error_bad_lines=False)
    print("Booking dataset has {} samples with {} features each.".format(*sample.shape))
    print("Feature names:")
    print(sample.columns)
except:
    print("Dataset could not be loaded. Check whether csv is available")


# In[126]:

# drop the column 'Unnamed: 0'
sample = sample.drop('Unnamed: 0',1)


# In[127]:

# create a 'target' column for our own convenience
sample['target'] = sample['is_booking'].astype('category')
print("Target variable: '{}' -> '{}'".format('is_booking', 'target'))


# # step1:-  lets explore the data

# In[128]:

##################### peek into data s#####################
sample.head(2)


# In[129]:

sample.dtypes


# In[130]:

sample.info()


# In[131]:

# check the booking dates i.e., first & last day of booking
print("First booking period",sample['date_time'].min()), print("Last booking period",sample['date_time'].max())


# In[10]:

# check if missing values present
def missing(x):
    return sum(x.isnull())


# In[132]:

#Applying per column:
print("Missing values per column:")
print(sample.apply(missing, axis=0))   #axis=0 defines that function is to be applied on each column

#Applying per row:
print("\nMissing values per row:")
print(sample.apply(missing, axis=1).head()) #axis=1 defines that function is to be applied on each row

# we observe that 3 variables have missing values, i.e., orig_destination_distance, srch_ci, srch_co


# In[133]:

sample.columns


# In[134]:

# define function to check unique values for all columns of the dataset excluding the new created target column
# unique counts
def unique_counts(sample):
    for i in sample.columns[:-1]:
        count = len(sample[i].unique().tolist())
        print(i, ": ", count)


# In[135]:

# check unique counts for each attributes in the dataset
print("Unique counts for each column")
print(unique_counts(sample))


# In[136]:

# lets see if there is any pattern in the choice of num of rooms prefered by customers who have done booking or haven't booked
pd.crosstab(sample['is_booking'], sample['srch_rm_cnt'])


# In[137]:

# mean of num. of rooms by booking status
sample.groupby('srch_rm_cnt')['is_booking'].mean()


# In[138]:

## What is correlation?
# A correlation coefficient measures the extent to which two variables tend to change together. 

## Pearson vs. Spearman difference 
# The Pearson correlation evaluates the linear relationship between two continuous variables.
# The Spearman correlation coefficient is based on the ranked values for each variable rather than the raw data. 
# Spearman correlation is often used to evaluate relationships involving ordinal variables

# check if any correlation between children count & booking status
sample['srch_children_cnt'].corr(sample['is_booking'])


# In[139]:

# check if any correlation between adult count & booking status
sample['srch_adults_cnt'].corr(sample['is_booking'])


# In[140]:

# check if any correlation between rooms preferred & booking status
sample['srch_rm_cnt'].corr(sample['is_booking'])


# In[141]:

# lets see the complete correlation matrix
sample.corr()


# In[142]:

# way to select all integer columns at once. The same can be done to get categorical columns specify 'object' & 
# for float columns just specify 'float64'
int_columns = sample.select_dtypes(['int64']).columns
print(int_columns)


# In[22]:

# EDA (EXPLORATORY DATA ANALYSIS)
# here i will shown you an easy way to explore the dataset with the library pandas_profiling
# https://pypi.org/project/pandas-profiling/1.4.0/
import pandas_profiling
# To generate Inline report without saving object
pandas_profiling.ProfileReport(sample[int_columns])
# to know more in detail about other options I shown an demo pls. check out https://github.com/ajayarunachalam/EDA


# In[143]:

##################### explore the data - doing descriptive statistics #####################
sample.describe()


# In[144]:

# if you don't want to use the above mentioned package & want to still plot the histogram you can manage it this way
# sample.hist()
sample[['srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt']].hist()


# In[25]:

# lets check the distribution of number of booking attempts
sample.groupby('user_id')['is_booking']   .agg({'num_of_bookings':'count'}).reset_index()   .groupby('num_of_bookings')['user_id']   .agg('count').reset_index().plot(x='num_of_bookings', y='user_id')


# In[145]:

sample = sample.merge(sample.groupby('user_id')['is_booking']
    .agg(['count']).reset_index())


# In[146]:

sample.columns


# In[147]:

sample.head(3)


# In[29]:

# lets check the distribution of booking rate
sample.groupby('user_id')['is_booking']   .agg(['mean']).reset_index()   .groupby('mean')['user_id']   .agg('count').reset_index().plot(x='mean', y='user_id')


# # explore the data for business logic validation #
# HERE WE WILL VALIDATE 3 THINGS
# 1) Check-in date should be > booking_date
# 2) Check-out date should be > check-in date
# 3) No. of guests should be > 0 

# In[148]:

# checking number of guests need to be > 0
pd.crosstab(sample['srch_adults_cnt'], sample['srch_children_cnt'])


# In[149]:

# we see that there are invalid entries, we need to drop them
sample.drop(sample[sample['srch_adults_cnt'] + sample['srch_children_cnt']==0].index, inplace=True)


# In[150]:

# reverifying the scenario
# checking number of guests need to be > 0
pd.crosstab(sample['srch_adults_cnt'], sample['srch_children_cnt'])


# In[151]:

# convert to date time
sample['srch_co'] = pd.to_datetime(sample['srch_co'])
sample['srch_ci'] = pd.to_datetime(sample['srch_ci'])
sample['date_time'] = pd.to_datetime(sample['date_time'])
sample['date'] = pd.to_datetime(sample['date_time'].apply(lambda x: x.date()))

# filter cases where the condition fails: Check-out date need to be later than check-in date;
print(sample[sample['srch_co'] < sample['srch_ci']][['srch_co', 'srch_ci']].count())
print(sample[sample['srch_co'] < sample['srch_ci']][['srch_co', 'srch_ci']])


# In[152]:

# filter cases where condition fails: Check-in date need to be later than booking date
print(sample[sample['srch_ci'] < sample['date']][['srch_ci', 'date']].count())
sample[sample['srch_ci'] < sample['date']][['srch_ci', 'date']]


# # create new features that might be useful - Feature Engineering #
# We create new variables like the days of stay (duration), how many days in advance the booking was made (days_in_advance)

# In[153]:

def duration(row):
    delta = (row['srch_co'] - row['srch_ci'])/np.timedelta64(1, 'D')
    if delta <= 0:
        return np.nan
    else:
        return delta


# In[154]:

# apply the function on the dataset
sample['duration'] = sample.apply(duration, axis=1)


# In[155]:

# check column duration entries
sample.duration.head(3)


# In[156]:

def days_in_advance(row):
    delta = (row['srch_ci'] - row['date'])/np.timedelta64(1, 'D') #Timedelta is a subclass of datetime.timedelta , it allows compatibility with np.timedelta64 & to convert to days we use (1,'D')
    if delta < 0:
        return np.nan
    else:
        return delta


# In[157]:

# apply the function on the dataset
sample['days_in_advance'] = sample.apply(days_in_advance, axis=1)


# In[158]:

# check the column days_in_advance 
sample['days_in_advance'].head(3)


# # Objective1: Identify the underperforming / overperforming segments #
# Based on booking rate

# In[159]:

cat_list = ['site_name', 'posa_continent',
       'user_location_country', 'user_location_region',
       'user_location_city', 'channel',
       'srch_destination_id', 'srch_destination_type_id',
        'hotel_continent', 'hotel_country', 'hotel_market',
       'hotel_cluster']


# In[160]:

# for all columns
for i in cat_list:
    print(sample.groupby(i)['is_booking']
          .agg({'booking_rate': 'mean', 'num_of_bookings': 'sum'})
          .reset_index()
          .sort_values(by='booking_rate'))


# In[161]:

# recheck again the booking rate per channel
sample.groupby('channel')['is_booking']    .agg({'booking_rate': 'mean', 'num_of_bookings': 'sum'})    .reset_index()    .sort_values(by='booking_rate')


# In[162]:

# this info is already available in our report provided by pandas_profiling
print(sample['is_booking'].mean())


# # TWO-SAMPLED t-test 
# #### to understand the underperforming & overperforming segments
# 
# * We use Two sample t-test to check whether the outperformance is statistically significant, i.e., we have to check if is booking rate for city 1 greater than other cities. Hypothesis test for the equality of the booking rate in two binomial samples (One Segment vs. all other Segment)
# 
# * We calculate the z-score. To find the z-score of a sample, we need to find the mean, variance and standard deviation of the sample. To calculate the z-score, we will find the difference between a value in the sample and the mean, and divide it by the standard deviation.
# 
# * Then, we calculate the probability score. We then mark it as significant (1) if prob. score > 0.9 , 0 in other cases, while if prob. score < 0.1 then we mark it as "-1"
# 
# * if x < 0.1 -> Not enough samples to check significance so we mark "-1"
#   if x > 0.9 -> we mark "1"
#   else       -> we mark "0"

# In[53]:

# functn for two-sampled t-test
def stats_comparison(i):
    cat = sample.groupby(i)['is_booking']        .agg({
            'sub_average': 'mean',
            'sub_bookings': 'count'
       }).reset_index()
    cat['overall_average'] = sample['is_booking'].mean()
    cat['overall_bookings'] = sample['is_booking'].count()
    cat['rest_bookings'] = cat['overall_bookings'] - cat['sub_bookings']
    cat['rest_average'] = (cat['overall_bookings']*cat['overall_average']                      - cat['sub_bookings']*cat['sub_average'])/cat['rest_bookings']
    cat['z_score'] = (cat['sub_average']-cat['rest_average'])/        np.sqrt(cat['overall_average']*(1-cat['overall_average'])
            *(1/cat['sub_bookings']+1/cat['rest_bookings']))
    cat['prob'] = np.around(stats.norm.cdf(cat.z_score), decimals = 10)
    cat['significant'] = [(lambda x: 1 if x > 0.9 else -1 if x < 0.1 else 0)(i) for i in cat['prob']]
    print(cat)


# In[54]:

stats_comparison('user_location_city')


# # Objective2: Focus on tailoring new marketing campaigns for different cities in the coming few months ahead. #
# we want to undestand what cities we can focus on for setting campaigns

# In[46]:

############## clustering - what are the similar user cities? ##############

# Step 1: what are the features we are going to use (that makes sense)?
# What features may distinguish cities? based on business sense and exploratory analysis

num_list = ['duration', 'days_in_advance', 'orig_destination_distance', 'is_mobile',
            'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt']
city_data = sample.dropna(axis=0)[num_list + ['user_location_city']]
city_groups = city_data.groupby('user_location_city').mean().reset_index().dropna(axis=0)


# In[47]:

city_groups.shape


# In[48]:

city_groups.columns


# In[49]:

city_groups.head(2)


# In[50]:

# Step 2: Standardise the data
# What is the magnitude of data range?
city_groups_std = city_groups.copy()
for i in num_list:
    city_groups_std[i] = preprocessing.scale(city_groups_std[i])


# In[51]:

city_groups_std.head(2)


# In[52]:

'''
K-MEANS ANALYSIS - INITIAL CLUSTER SET
'''
# Step 3: select clustering method and number of clusters
# The Elbow method choose a K so that the sum of the square error of the distances decrease drastically
# there are methods to help derive the optimal number for k
# k-means cluster analysis for 1-10 clusters in random at first                                                       
from scipy.spatial.distance import cdist
clusters=range(1,11)
meandist=[]

# loop through each cluster and fit the model to the train set
# generate the predicted cluster assingment and append the mean distance my taking the sum divided by the shape
for k in clusters:
    model=cluster.KMeans(n_clusters=k)
    model.fit(city_groups_std)
    clusassign=model.predict(city_groups_std)
    meandist.append(sum(np.min(cdist(city_groups_std, model.cluster_centers_, 'euclidean'), axis=1))
    / city_groups_std.shape[0])


# In[53]:

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method') # pick the fewest number of clusters that reduces the average distance


# In[54]:

# Let us Interpret 3 cluster solution
km = cluster.KMeans(n_clusters=3, max_iter=300, random_state=None)
city_groups_std['cluster'] = km.fit_predict(city_groups_std[num_list])


# In[55]:

city_groups_std.columns


# In[56]:

city_groups_std.head(2)


# In[57]:

# check the cluster size population
plt.figure(figsize=(8, 6))
city_groups_std['cluster'].value_counts().plot.bar()


# In[58]:

city_groups_std.columns


# In[59]:

# Principal Component Analysis
pca = decomposition.PCA(n_components=2)
pca.fit(city_groups[num_list])
city_groups_std['x'] = pca.fit_transform(city_groups_std[num_list])[:, 0]
city_groups_std['y'] = pca.fit_transform(city_groups_std[num_list])[:, 1]
plt.figure(figsize=(8, 6))
plt.scatter(city_groups_std['x'], city_groups_std['y'],c=city_groups_std['cluster'])
plt.show()


# In[60]:

# other way to recheck
''' Canonical Discriminant Analysis for variable reduction:
1. creates a smaller number of variables
2. linear combination of clustering variables
3. Canonical variables are ordered by proportion of variance accounted for
4. most of the variance will be accounted for in the first few canonical variables
'''
from sklearn.decomposition import PCA # CA from PCA function
pca_2 = PCA(2) # return 2 first canonical variables
plot_columns = pca_2.fit_transform(city_groups_std) # fit CA to the dataset
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=km.labels_,) # plot 1st canonical variable on x axis, 2nd on y-axis
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show() # close or overlapping clusters indicate correlated variables with low in-class variance, but not good separation. 
           # 2 cluster might be better or appropriate in such case.


# In[61]:

city_groups_std.columns


# In[62]:

city_groups_std.head(2)


# In[63]:

# Step 4: profile the clusters
# merging the two dataframes based on a common column user_location_city
profile_cluster = city_groups.merge(city_groups_std[['user_location_city', 'cluster']])    .groupby('cluster')    .mean() # for every column


# In[64]:

profile_cluster


# In[65]:

profile_cluster.columns


# In[66]:

city_data.head(2)


# In[67]:

city_groups.head(2)


# In[68]:

city_groups_std.head(2)


# In[69]:

# mapping cluster to city_data
city_data['clusters'] = city_groups_std['cluster']


# In[70]:

city_data.head(2)


# In[71]:

sample.head(2)


# In[72]:

# check the user_location_city in each clusters
group = city_groups_std[['user_location_city','cluster']]
from collections import defaultdict
d = defaultdict(list)
for each in group.iterrows():
    d[each[1]['cluster']].append(each[1]['user_location_city'])


# In[73]:

d


# In[74]:

# write to file
import csv
with open('user_location_city_by_cluster.csv', 'w') as f:  
    w = csv.writer(f)
    w.writerows(d.items())


# # Objective3: Focus on how the user booking rate can be improved?
# ############### We need to understand what lead to a higher chance of booking for individuals? ###############

# In[75]:

sample.columns


# In[76]:

# we will build a decision tree to understand what is that factor that pushes people to do booking
from sklearn.cross_validation import train_test_split


# In[77]:

# choose a cluster 
sample = sample.merge(city_groups_std[['user_location_city', 'cluster']], left_on='user_location_city', right_on='user_location_city', how='outer')
sample.groupby('cluster')['is_booking'].count()


# In[78]:

# choose one of the city clusters to analyze, here we choose cluster 0 as it has most data points
tree_data = sample.dropna(axis = 0)[sample['cluster']==0]


# In[79]:

# split into test and train
tree_train, tree_test = train_test_split(tree_data, test_size=0.2, random_state=1, stratify=tree_data['is_booking'])


# In[80]:

# build the decision tree model
clf = tree.DecisionTreeClassifier(max_leaf_nodes=6, min_samples_leaf=200)
clf = clf.fit(tree_train[num_list], tree_train['is_booking'])


# In[81]:

# scoring of the prediction model
clf.score(tree_test[num_list], tree_test['is_booking'])


# In[82]:

# explicitly reassign the path to be used with notebook
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# visualize the decision tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names =['duration', 'days_in_advance', 'orig_destination_distance', 'is_mobile',
            'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt'], filled=True, rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("booking_tree.pdf")


# In[83]:

print(clf.tree_.feature)


# In[84]:

print(clf.tree_.threshold)


# In[85]:

print(clf.tree_.value)


# In[86]:

print(clf.tree_.children_left),print(clf.tree_.children_right)


# In[87]:

# pseudocode adopted from stackoverflow
def get_pseudocode(tree, feature_names):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def parsetree(left, right, threshold, features, node):
                if (threshold[node] != -2):
                        print("if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                        if left[node] != -1:
                                parsetree (left, right, threshold, features,left[node])
                        print("} else {")
                        if right[node] != -1:
                                parsetree (left, right, threshold, features,right[node])
                        print("}")
                else:
                        print("return " + str(value[node]))

        parsetree(left, right, threshold, features, 0)


# In[88]:

get_pseudocode(clf, feature_names =['duration', 'days_in_advance', 'orig_destination_distance', 'is_mobile',
            'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt'])


# # Objective3: Focus on how the user booking rate can be improved?
# ############### We need to understand what lead to a higher chance of booking for individuals? ###############

# In[89]:

# we will build logistic regression model to examine what leads to people doing booking
sample.columns


# In[90]:

# check the statistics by channel for top5 results
# you can also use sample['channel'].value_counts() to get results. I just wanted to show another way
channel_stats = sample.groupby('channel').apply(lambda x: len(x))
channel_stats.nlargest(5)


# In[91]:

#create dummy variable channel
# we keep top5 channels as it is and put the rest as 'other'
sample['new_channel'] = [i if i in [9, 0, 1, 2,5] else 'other' for i in sample['channel']]


# In[92]:

# check new_channel distribution
sample['new_channel'].value_counts()


# In[93]:

# one-hot encoding using pd.get_dummies
dummy_channels = pd.get_dummies(sample['new_channel'], prefix='channel')


# In[94]:

dummy_channels


# In[95]:

# merge the newly encoding dummy variable to original sample file
sample = sample.join(dummy_channels.ix[:, :])


# In[96]:

sample.columns


# In[97]:

sample.head(2)


# In[99]:

# check orig_destination_distance distribution
sample['orig_destination_distance'].hist()


# In[100]:

# we do log transformation to make highly skewed distributions less skewed
sample['log_orig_destination_distance'] = [np.log(i) for i in sample['orig_destination_distance']]


# In[101]:

# check distribution of newly created variable with log transformation
sample['log_orig_destination_distance'].hist()


# In[102]:

# select features
var_list = ['duration', 'days_in_advance', 'log_orig_destination_distance', 'is_mobile',
            'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'channel_0', 'channel_1', 'channel_2', 'channel_5', 
            'channel_9', 'channel_other']


# In[103]:

# create a logistic estimator
logit = sm.Logit(sample['is_booking'], sample[var_list], missing='drop')
result = logit.fit()
result.summary()


# In[107]:

result.params


# In[104]:

# convert logit to odds ratio
np.exp(result.params)
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
np.exp(conf)


# In[111]:

# predicted values
sample['pred'] = result.predict(sample[var_list])


# In[112]:

# plot variable VS. marketing_channel
sample.groupby('channel')['is_booking'].mean()


# In[113]:

def plot_booking_vs_marketing_channel(variable):
    grouped = pd.pivot_table(sample[(sample['pred'].isnull() == False)], 
                             values = ['pred'], index=[variable, 'new_channel'], aggfunc=np.mean)
    colors = 'rbgyrbgy'
    for col in sample.new_channel.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0), plt_data['pred'])
    pl.xlabel(variable)
    pl.ylabel('prob booking = 1')
    pl.legend(['channel_'+str(i) for i in list(sample.new_channel.unique())], loc='upper right', title='new channel')


# In[114]:

plot_booking_vs_marketing_channel('days_in_advance')


# In[119]:

def plot_booking_vs_isPackage(variable):
    grouped = pd.pivot_table(sample[(sample['pred'].isnull() == False)], 
                             values = ['pred'], index=[variable, 'is_package'], aggfunc=np.mean)
    colors = 'rbgyrbgy'
    for col in sample.is_package.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0), plt_data['pred'])
    pl.xlabel(variable)
    pl.ylabel('prob booking = 1')
    pl.legend(['is_package_'+str(i) for i in list(sample.is_package.unique())], loc='upper right', title='Cust has taken package')


# In[120]:

plot_booking_vs_isPackage('days_in_advance')


# In[121]:

def plot_booking_vs_isMobile(variable):
    grouped = pd.pivot_table(sample[(sample['pred'].isnull() == False)], 
                             values = ['pred'], index=[variable, 'is_mobile'], aggfunc=np.mean)
    colors = 'rbgyrbgy'
    for col in sample.is_mobile.unique():
        plt_data = grouped.ix[grouped.index.get_level_values(1)==col]
        pl.plot(plt_data.index.get_level_values(0), plt_data['pred'])
    pl.xlabel(variable)
    pl.ylabel('prob booking = 1')
    pl.legend(['is_mobile_'+str(i) for i in list(sample.is_mobile.unique())], loc='upper right', title='Booking done using mobile')


# In[122]:

plot_booking_vs_isMobile('days_in_advance')


# In[ ]:



