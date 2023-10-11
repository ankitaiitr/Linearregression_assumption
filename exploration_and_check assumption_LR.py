# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:32:35 2020

@author: Ankita
"""

#data exploration and checking if linear regression assumptions hold true

#https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python

#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#bring in the six packs
df_train = pd.read_csv('../input/train.csv')

#check the decoration
df_train.columns

#descriptive statistics summary
df_train['SalePrice'].describe()

#histogram
sns.distplot(df_train['SalePrice']);

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


#correlation matrix-heatmap style

corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


#saleprice correlation matrix  -zoomed heatmap style
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()



#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();


#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)   


#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max() #just checking that there's no missing data missing...



#outliers
#univariate analysis

#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#bivariate analysis

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)


#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

'''According to Hair et al. (2013), four assumptions should be tested:

Normality - When we talk about normality what we mean is that the data should 
look like a normal distribution. This is important because several statistic 
tests rely on this (e.g. t-statistics). In this exercise we'll just check univariate 
normality for 'SalePrice' (which is a limited approach). Remember that univariate 
normality doesn't ensure multivariate normality (which is what we would like to 
have), but it helps. Another detail to take into account is that in big samples 
    (>200 observations) normality is not such an issue. However, if we solve 
    normality, we avoid a lot of other problems (e.g. heteroscedacity) so that's
    the main reason why we are doing this analysis.

Homoscedasticity - I just hope I wrote it right. Homoscedasticity refers to the
 'assumption that dependent variable(s) exhibit equal levels of variance across
 the range of predictor variable(s)' (Hair et al., 2013). Homoscedasticity is 
 desirable because we want the error term to be the same across all values of 
 the independent variables.

Linearity- The most common way to assess linearity is to examine scatter plots 
and search for linear patterns. If patterns are not linear, it would be worthwhile
 to explore data transformations. However, we'll not get into this because most
 of the scatter plots we've seen appear to have linear relationships.

Absence of correlated errors - Correlated errors, like the definition suggests,
 happen when one error is correlated to another. For instance, if one positive 
 error makes a negative error systematically, it means that there's a relationship
 between these variables. This occurs often in time series, where some patterns
 are time related. We'll also not get into this. However, if you detect something,
 try to add a variable that can explain the effect you're getting. That's the most
 common solution for correlated errors.
 '''
 #normality
 '''
 The point here is to test 'SalePrice' in a very lean way. We'll do this paying attention to:

Histogram - Kurtosis and skewness.
Normal probability plot - Data distribution should closely follow the 
diagonal that represents the normal distribution.'''
 
 #histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)


#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])



#homoscedasticity check

#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);

#That's the power of normality! Just by ensuring normality in some variables,
# we solved the homoscedasticity problem.

#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);


#convert categorical variable into dummy
df_train = pd.get_dummies(df_train)