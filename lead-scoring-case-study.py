#!/usr/bin/env python
# coding: utf-8

# # LEAD SCORING CASE STUDY

# ### Problem Statement
# 
# An education company named X Education sells online courses to industry professionals. On any given day, many professionals who are interested in the courses land on their website and browse for courses. 
# 
# The company markets its courses on several websites and search engines like Google. Once these people land on the website, they might browse the courses or fill up a form for the course or watch some videos. When these people fill up a form providing their email address or phone number, they are classified to be a lead. Moreover, the company also gets leads through past referrals. Once these leads are acquired, employees from the sales team start making calls, writing emails, etc. Through this process, some of the leads get converted while most do not. The typical lead conversion rate at X education is around 30%. 
# 
# There are a lot of leads generated in the initial stage, but only a few of them come out as paying customers. In the middle stage, you need to nurture the potential leads well (i.e. educating the leads about the product, constantly communicating etc. ) in order to get a higher lead conversion. 
# 
# X Education has appointed you to help them select the most promising leads, i.e. the leads that are most likely to convert into paying customers. The company requires you to **build a model wherein you need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance**. The CEO, in particular, has given a ballpark of the target lead conversion rate to be around 80%.

# ### Goals of the Case Study
# 
# - Build a **logistic regression model** to assign a lead score between 0 and 100 to each of the leads which can be used by the company to target potential leads. A higher score would mean that the lead is hot, i.e. is most likely to convert whereas a lower score would mean that the lead is cold and will mostly not get converted.

# In[1]:


#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler


# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# In[3]:


#importing dataset to csv

leads=pd.read_csv("Leads.csv")
leads.head()


# In[4]:


#checking total rows and cols in dataset
leads.shape


# This dataset has:
# - 9240 rows,
# - 37 columns

# In[5]:


#basic data check
leads.info()


# In[6]:


leads.describe()


# In[7]:


#check for duplicates
sum(leads.duplicated(subset = 'Prospect ID')) == 0


# **No duplicate values in Prospect ID**

# In[8]:


#check for duplicates
sum(leads.duplicated(subset = 'Lead Number')) == 0


# **No duplicate values in Lead Number**

# Clearly Prospect ID & Lead Number are two variables that are just indicative of the ID number of the Contacted People & can be dropped.

# ## EXPLORATORY DATA ANALYSIS

# ## Data Cleaning & Treatment:

# In[9]:


#dropping Lead Number and Prospect ID since they have all unique values

leads.drop(['Prospect ID', 'Lead Number'], 1, inplace = True)


# In[10]:


#Converting 'Select' values to NaN.

leads = leads.replace('Select', np.nan)


# In[11]:


#checking null values in each rows

leads.isnull().sum()


# In[12]:


#checking percentage of null values in each column

round(100*(leads.isnull().sum()/len(leads.index)), 2)


# In[13]:


#dropping cols with more than 45% missing values

cols=leads.columns

for i in cols:
    if((100*(leads[i].isnull().sum()/len(leads.index))) >= 45):
        leads.drop(i, 1, inplace = True)


# In[14]:


#checking null values percentage

round(100*(leads.isnull().sum()/len(leads.index)), 2)


# ## Categorical Attributes Analysis:

# In[15]:


#checking value counts of Country column

leads['Country'].value_counts(dropna=False)


# In[16]:


#plotting spread of Country columnn 
plt.figure(figsize=(15,5))
s1=sns.countplot(leads.Country, hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[17]:


# Since India is the most common occurence among the non-missing values we can impute all missing values with India

leads['Country'] = leads['Country'].replace(np.nan,'India')


# In[18]:


#plotting spread of Country columnn after replacing NaN values

plt.figure(figsize=(15,5))
s1=sns.countplot(leads.Country, hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# **As we can see the Number of Values for India are quite high (nearly 97% of the Data), this column can be dropped**

# In[19]:


#creating a list of columns to be droppped

cols_to_drop=['Country']


# In[20]:


#checking value counts of "City" column

leads['City'].value_counts(dropna=False)


# In[21]:


leads['City'] = leads['City'].replace(np.nan,'Mumbai')


# In[22]:


#plotting spread of City columnn after replacing NaN values

plt.figure(figsize=(10,5))
s1=sns.countplot(leads.City, hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[23]:


#checking value counts of Specialization column

leads['Specialization'].value_counts(dropna=False)


# In[24]:


# Lead may not have mentioned specialization because it was not in the list or maybe they are a students 
# and don't have a specialization yet. So we will replace NaN values here with 'Not Specified'

leads['Specialization'] = leads['Specialization'].replace(np.nan, 'Not Specified')


# In[25]:


#plotting spread of Specialization columnn 

plt.figure(figsize=(15,5))
s1=sns.countplot(leads.Specialization, hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# We see that specialization with **Management** in them have higher number of leads as well as leads converted.
# So this is definitely a significant variable and should not be dropped.

# In[26]:


#combining Management Specializations because they show similar trends

leads['Specialization'] = leads['Specialization'].replace(['Finance Management','Human Resource Management',
                                                           'Marketing Management','Operations Management',
                                                           'IT Projects Management','Supply Chain Management',
                                                    'Healthcare Management','Hospitality Management',
                                                           'Retail Management'] ,'Management_Specializations')  


# In[27]:


#visualizing count of Variable based on Converted value


plt.figure(figsize=(15,5))
s1=sns.countplot(leads.Specialization, hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[28]:


#What is your current occupation

leads['What is your current occupation'].value_counts(dropna=False)


# In[29]:


#imputing Nan values with mode "Unemployed"

leads['What is your current occupation'] = leads['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[30]:


#checking count of values
leads['What is your current occupation'].value_counts(dropna=False)


# In[31]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(leads['What is your current occupation'], hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# - Working Professionals going for the course have high chances of joining it.
# - Unemployed leads are the most in terms of Absolute numbers.

# In[32]:


#checking value counts

leads['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[33]:


#replacing Nan values with Mode "Better Career Prospects"

leads['What matters most to you in choosing a course'] = leads['What matters most to you in choosing a course'].replace(np.nan,'Better Career Prospects')


# In[34]:


#visualizing count of Variable based on Converted value

s1=sns.countplot(leads['What matters most to you in choosing a course'], hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[35]:


#checking value counts of variable
leads['What matters most to you in choosing a course'].value_counts(dropna=False)


# In[36]:


#Here again we have another Column that is worth Dropping. So we Append to the cols_to_drop List
cols_to_drop.append('What matters most to you in choosing a course')
cols_to_drop


# In[37]:


#checking value counts of Tag variable
leads['Tags'].value_counts(dropna=False)


# In[38]:


#replacing Nan values with "Not Specified"
leads['Tags'] = leads['Tags'].replace(np.nan,'Not Specified')


# In[39]:


#visualizing count of Variable based on Converted value

plt.figure(figsize=(15,5))
s1=sns.countplot(leads['Tags'], hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[40]:


#replacing tags with low frequency with "Other Tags"
leads['Tags'] = leads['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)',
                                     'Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking',
                                    'Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch',
                                    'Recognition issue (DEC approval)','Want to take admission but has financial problems',
                                    'University not recognized'], 'Other_Tags')

leads['Tags'] = leads['Tags'].replace(['switched off',
                                      'Already a student',
                                       'Not doing further education',
                                       'invalid number',
                                       'wrong number given',
                                       'Interested  in full time MBA'] , 'Other_Tags')


# In[41]:


#checking percentage of missing values
round(100*(leads.isnull().sum()/len(leads.index)), 2)


# In[42]:


#checking value counts of Lead Source column

leads['Lead Source'].value_counts(dropna=False)


# In[43]:


#replacing Nan Values and combining low frequency values
leads['Lead Source'] = leads['Lead Source'].replace(np.nan,'Others')
leads['Lead Source'] = leads['Lead Source'].replace('google','Google')
leads['Lead Source'] = leads['Lead Source'].replace('Facebook','Social Media')
leads['Lead Source'] = leads['Lead Source'].replace(['bing','Click2call','Press_Release',
                                                     'youtubechannel','welearnblog_Home',
                                                     'WeLearn','blog','Pay per Click Ads',
                                                    'testone','NC_EDM'] ,'Others')                                                   


# We can group some of the lower frequency occuring labels under a common label 'Others' 

# In[44]:


#visualizing count of Variable based on Converted value
plt.figure(figsize=(15,5))
s1=sns.countplot(leads['Lead Source'], hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# #### Inference
# - Maximum number of leads are generated by Google and Direct traffic.
# - Conversion Rate of reference leads and leads through welingak website is high.
# - To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.

# In[45]:


# Last Activity:

leads['Last Activity'].value_counts(dropna=False)


# In[46]:


#replacing Nan Values and combining low frequency values

leads['Last Activity'] = leads['Last Activity'].replace(np.nan,'Others')
leads['Last Activity'] = leads['Last Activity'].replace(['Unreachable','Unsubscribed',
                                                        'Had a Phone Conversation', 
                                                        'Approached upfront',
                                                        'View in browser link Clicked',       
                                                        'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails',
                                                         'Visited Booth in Tradeshow'],'Others')


# In[47]:


# Last Activity:

leads['Last Activity'].value_counts(dropna=False)


# In[48]:


#Check the Null Values in All Columns:
round(100*(leads.isnull().sum()/len(leads.index)), 2)


# In[49]:


#Drop all rows which have Nan Values. Since the number of Dropped rows is less than 2%, it will not affect the model
leads = leads.dropna()


# In[50]:


#Checking percentage of Null Values in All Columns:
round(100*(leads.isnull().sum()/len(leads.index)), 2)


# In[51]:


#Lead Origin
leads['Lead Origin'].value_counts(dropna=False)


# In[52]:


#visualizing count of Variable based on Converted value

plt.figure(figsize=(8,5))
s1=sns.countplot(leads['Lead Origin'], hue=leads.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# #### Inference
# - API and Landing Page Submission bring higher number of leads as well as conversion.
# - Lead Add Form has a very high conversion rate but count of leads are not very high.
# - Lead Import and Quick Add Form get very few leads.
# - In order to improve overall lead conversion rate, we have to improve lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.

# In[53]:


#Do Not Email & Do Not Call
#visualizing count of Variable based on Converted value

plt.figure(figsize=(15,5))

ax1=plt.subplot(1, 2, 1)
ax1=sns.countplot(leads['Do Not Call'], hue=leads.Converted)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax2=plt.subplot(1, 2, 2)
ax2=sns.countplot(leads['Do Not Email'], hue=leads.Converted)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
plt.show()


# In[54]:


#checking value counts for Do Not Call
leads['Do Not Call'].value_counts(dropna=False)


# In[55]:


#checking value counts for Do Not Email
leads['Do Not Email'].value_counts(dropna=False)


# We Can append the **Do Not Call** Column to the list of Columns to be Dropped since > 90% is of only one Value

# In[56]:


cols_to_drop.append('Do Not Call')
cols_to_drop


# In[57]:


# IMBALANCED VARIABLES THAT CAN BE DROPPED


# In[58]:


leads.Search.value_counts(dropna=False)


# In[59]:


leads.Magazine.value_counts(dropna=False)


# In[60]:


leads['Newspaper Article'].value_counts(dropna=False)


# In[61]:


leads['X Education Forums'].value_counts(dropna=False)


# In[62]:


leads['Newspaper'].value_counts(dropna=False)


# In[63]:


leads['Digital Advertisement'].value_counts(dropna=False)


# In[64]:


leads['Through Recommendations'].value_counts(dropna=False)


# In[65]:


leads['Receive More Updates About Our Courses'].value_counts(dropna=False)


# In[66]:


leads['Update me on Supply Chain Content'].value_counts(dropna=False)


# In[67]:


leads['Get updates on DM Content'].value_counts(dropna=False)


# In[68]:


leads['I agree to pay the amount through cheque'].value_counts(dropna=False)


# In[69]:


leads['A free copy of Mastering The Interview'].value_counts(dropna=False)


# In[70]:


#adding imbalanced columns to the list of columns to be dropped

cols_to_drop.extend(['Search','Magazine','Newspaper Article','X Education Forums','Newspaper',
                 'Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                 'Update me on Supply Chain Content',
                 'Get updates on DM Content','I agree to pay the amount through cheque'])


# In[71]:


#checking value counts of last Notable Activity
leads['Last Notable Activity'].value_counts()


# In[72]:


#clubbing lower frequency values

leads['Last Notable Activity'] = leads['Last Notable Activity'].replace(['Had a Phone Conversation',
                                                                       'Email Marked Spam',
                                                                         'Unreachable',
                                                                         'Unsubscribed',
                                                                         'Email Bounced',                                                                    
                                                                       'Resubscribed to emails',
                                                                       'View in browser link Clicked',
                                                                       'Approached upfront', 
                                                                       'Form Submitted on Website', 
                                                                       'Email Received'],'Other_Notable_activity')


# In[73]:


#visualizing count of Variable based on Converted value

plt.figure(figsize = (14,5))
ax1=sns.countplot(x = "Last Notable Activity", hue = "Converted", data = leads)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
plt.show()


# In[74]:


#checking value counts for variable

leads['Last Notable Activity'].value_counts()


# In[75]:


#list of columns to be dropped
cols_to_drop


# In[76]:


#dropping columns
leads = leads.drop(cols_to_drop,1)
leads.info()


# ## Numerical Attributes Analysis:

# In[77]:


#Check the % of Data that has Converted Values = 1:

Converted = (sum(leads['Converted'])/len(leads['Converted'].index))*100
Converted


# In[78]:


#Checking correlations of numeric values
# figure size
plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(leads.corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[79]:


#Total Visits
#visualizing spread of variable

plt.figure(figsize=(6,4))
sns.boxplot(y=leads['TotalVisits'])
plt.show()


# We can see presence of outliers here

# In[80]:


#checking percentile values for "Total Visits"

leads['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[81]:


#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values

Q3 = leads.TotalVisits.quantile(0.99)
leads = leads[(leads.TotalVisits <= Q3)]
Q1 = leads.TotalVisits.quantile(0.01)
leads = leads[(leads.TotalVisits >= Q1)]
sns.boxplot(y=leads['TotalVisits'])
plt.show()


# In[82]:


leads.shape


# Check for the Next Numerical Column:

# In[83]:


#checking percentiles for "Total Time Spent on Website"

leads['Total Time Spent on Website'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[84]:


#visualizing spread of numeric variable

plt.figure(figsize=(6,4))
sns.boxplot(y=leads['Total Time Spent on Website'])
plt.show()


# Since there are no major Outliers for the above variable we don't do any Outlier Treatment for this above Column

# Check for Page Views Per Visit:

# In[85]:


#checking spread of "Page Views Per Visit"

leads['Page Views Per Visit'].describe()


# In[86]:


#visualizing spread of numeric variable

plt.figure(figsize=(6,4))
sns.boxplot(y=leads['Page Views Per Visit'])
plt.show()


# In[87]:


#Outlier Treatment: Remove top & bottom 1% 

Q3 = leads['Page Views Per Visit'].quantile(0.99)
leads = leads[leads['Page Views Per Visit'] <= Q3]
Q1 = leads['Page Views Per Visit'].quantile(0.01)
leads = leads[leads['Page Views Per Visit'] >= Q1]
sns.boxplot(y=leads['Page Views Per Visit'])
plt.show()


# In[88]:


leads.shape


# In[89]:


#checking Spread of "Total Visits" vs Converted variable
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = leads)
plt.show()


# Inference
# - Median for converted and not converted leads are the close.
# - Nothng conclusive can be said on the basis of Total Visits

# In[90]:


#checking Spread of "Total Time Spent on Website" vs Converted variable

sns.boxplot(x=leads.Converted, y=leads['Total Time Spent on Website'])
plt.show()


# Inference
# - Leads spending more time on the website are more likely to be converted.
# - Website should be made more engaging to make leads spend more time.

# In[91]:


#checking Spread of "Page Views Per Visit" vs Converted variable

sns.boxplot(x=leads.Converted,y=leads['Page Views Per Visit'])
plt.show()


# Inference
# - Median for converted and unconverted leads is the same.
# - Nothing can be said specifically for lead conversion from Page Views Per Visit

# In[92]:


#checking missing values in leftover columns/

round(100*(leads.isnull().sum()/len(leads.index)),2)


# There are no missing values in the columns to be analyzed further

#     ## Dummy Variable Creation:

# In[93]:


#getting a list of categorical columns

cat_cols= leads.select_dtypes(include=['object']).columns
cat_cols


# In[94]:


# List of variables to map

varlist =  ['A free copy of Mastering The Interview','Do Not Email']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
leads[varlist] = leads[varlist].apply(binary_map)


# In[95]:


#getting dummies and dropping the first column and adding the results to the master dataframe
dummy = pd.get_dummies(leads[['Lead Origin','What is your current occupation',
                             'City']], drop_first=True)

leads = pd.concat([leads,dummy],1)


# In[96]:


dummy = pd.get_dummies(leads['Specialization'], prefix  = 'Specialization')
dummy = dummy.drop(['Specialization_Not Specified'], 1)
leads = pd.concat([leads, dummy], axis = 1)


# In[97]:


dummy = pd.get_dummies(leads['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
leads = pd.concat([leads, dummy], axis = 1)


# In[98]:


dummy = pd.get_dummies(leads['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
leads = pd.concat([leads, dummy], axis = 1)


# In[99]:


dummy = pd.get_dummies(leads['Last Notable Activity'], prefix  = 'Last Notable Activity')
dummy = dummy.drop(['Last Notable Activity_Other_Notable_activity'], 1)
leads = pd.concat([leads, dummy], axis = 1)


# In[100]:


dummy = pd.get_dummies(leads['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
leads = pd.concat([leads, dummy], axis = 1)


# In[101]:


#dropping the original columns after dummy variable creation

leads.drop(cat_cols,1,inplace = True)


# In[102]:


leads.head()


# ## Train-Test Split & Logistic Regression Model Building:

# In[103]:


from sklearn.model_selection import train_test_split

# Putting response variable to y
y = leads['Converted']

y.head()

X=leads.drop('Converted', axis=1)


# In[104]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[105]:


X_train.info()


# ### Scaling of Data:

# In[106]:


#scaling numeric columns

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

X_train.head()


# ### Model Building using Stats Model & RFE:

# In[107]:


import statsmodels.api as sm


# In[108]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[109]:


rfe.support_


# In[110]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[111]:


#list of RFE supported columns
col = X_train.columns[rfe.support_]
col


# In[112]:


X_train.columns[~rfe.support_]


# In[113]:


#BUILDING MODEL #1

X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# p-value of variable Lead Source_Referral Sites is high, so we can drop it.

# In[114]:


#dropping column with high p-value

col = col.drop('Lead Source_Referral Sites',1)


# In[115]:


#BUILDING MODEL #2

X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# Since 'All' the p-values are less we can check the Variance Inflation Factor to see if there is any correlation between the variables

# In[116]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[117]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# There is a high correlation between two variables so we drop the variable with the higher valued VIF value

# In[118]:


#dropping variable with high VIF

col = col.drop('Last Notable Activity_SMS Sent',1)


# In[119]:


#BUILDING MODEL #3
X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[120]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# So the Values all seem to be in order so now, Moving on to derive the Probabilities, Lead Score, Predictions on Train Data:

# In[121]:


# Getting the Predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[122]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[123]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[124]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[125]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[126]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[127]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[128]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[129]:


# Let us calculate specificity
TN / float(TN+FP)


# In[130]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[131]:


# positive predictive value 
print (TP / float(TP+FP))


# In[132]:


# Negative predictive value
print (TN / float(TN+ FN))


# ### PLOTTING ROC CURVE

# In[133]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[134]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )


# In[135]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# The ROC Curve should be a value close to 1. We are getting a good value of 0.97 indicating a good predictive model.

# ### Finding Optimal Cutoff Point

# Above we had chosen an arbitrary cut-off value of 0.5. We need to determine the best cut-off value and the below section deals with that: 

# In[136]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[137]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[138]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[139]:


#### From the curve above, 0.3 is the optimum point to take it as a cutoff probability.

y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)

y_train_pred_final.head()


# In[140]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))

y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[141]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[142]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[143]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[144]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[145]:


# Let us calculate specificity
TN / float(TN+FP)


# ### Observation:
# So as we can see above the model seems to be performing well. The ROC curve has a value of 0.97, which is very good. We have the following values for the Train Data:
# - Accuracy : 92.29%
# - Sensitivity : 91.70%
# - Specificity : 92.66%

# Some of the other Stats are derived below, indicating the False Positive Rate, Positive Predictive Value,Negative Predictive Values, Precision & Recall. 

# In[146]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[147]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[148]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[149]:


#Looking at the confusion matrix again

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[150]:


##### Precision
TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[151]:


##### Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[152]:


from sklearn.metrics import precision_score, recall_score


# In[153]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[154]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[155]:


from sklearn.metrics import precision_recall_curve


# In[156]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[157]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[158]:


#scaling test set

num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[159]:


X_test = X_test[col]
X_test.head()


# In[160]:


X_test_sm = sm.add_constant(X_test)


# ### PREDICTIONS ON TEST SET

# In[161]:


y_test_pred = res.predict(X_test_sm)


# In[162]:


y_test_pred[:10]


# In[163]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[164]:


# Let's see the head
y_pred_1.head()


# In[165]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[166]:


# Putting CustID to index
y_test_df['Prospect ID'] = y_test_df.index


# In[167]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[168]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[169]:


y_pred_final.head()


# In[170]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[171]:


y_pred_final.head()


# In[172]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[173]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[174]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[175]:


y_pred_final.head()


# In[176]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[177]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[178]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[179]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[180]:


# Let us calculate specificity
TN / float(TN+FP)


# In[181]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[182]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# ### Observation:
# After running the model on the Test Data these are the figures we obtain:
# - Accuracy : 92.78%
# - Sensitivity : 91.98%
# - Specificity : 93.26%

# ## Final Observation:
# 
# Let us compare the values obtained for Train & Test:
# 
# ### <u> Train Data: </u>
# - Accuracy : 92.29%
# - Sensitivity : 91.70%
# - Specificity : 92.66%
# 
# ### <u> Test Data: </u>
# - Accuracy : 92.78%
# - Sensitivity : 91.98%
# - Specificity : 93.26%

# The Model seems to predict the Conversion Rate very well and we should be able to give the CEO confidence in making good calls based on this model

# In[ ]:




