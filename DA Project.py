#!/usr/bin/env python
# coding: utf-8

# ### Problem 1 [3 points]
# Load the data provided in universities.csv from the moodle course page. The data is about economic and
# demographic characteristics of various US universities.
# Load the data from tuition_income.csv from the same page.
# Select from the universities dataset data for all universities located in the following states: New Mexico,
# Utah, Montana, Kentucky, South Carolina, Alabama, Idaho, West Virginia, Arkansas, Mississipi. Use this
# restricted data set as basis for the entire exam (and modify accordingly in the following questions).

# In[1]:


import pandas as pd
import plotly.express as px
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
pd.set_option('max_columns', None)
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_curve,auc,plot_confusion_matrix


# In[2]:


# Taking University dataset
uni= pd.read_csv("universities.csv")
uni


# In[3]:


# Taking Tuition Income datatset
dftution= pd.read_csv("tuition_income.csv")


# ### 1. How many universities are in the resulting dataset? [1] 

# In[4]:


df = uni
df = df.loc[df['state'].isin(["New Mexico", 
                            "Utah", 
                            "Montana", 
                            "Kentucky", 
                            "South Carolina", 
                            "Alabama", 
                            "Idaho", 
                            "West Virginia", 
                            "Arkansas", 
                            "Mississipi"])]


df = df[df["name"].str.contains('University')]
df


# Solution: There in total 99 Universities in the resulting Dataset.

# ### 2. Graduates of how many universities have an estimated early career pay greater than 50000 dollars per year? [1]

# In[5]:


# ECP is estimated early career pay
ECP = df[df['early_career_pay']>50000]
ECP


# Solution: 15 Universities Graduates are having an estimated early career pay greater than 50000 dollars
# per year?

# ### 3. How many universities are not public? [1]

# In[6]:


# Not public = Private + For Profit
pvt = df[df['type']!= 'Public']
print(pvt.type.value_counts())
pvt


# Solution: 38 Universities are not public .

# ### Problem 2: Making sense of your data [15 points]

# In[7]:


df.dtypes


# ### 1. Take a quick look at the data structure. Describe the dataset in a few sentences. [2]
# 
# Most of the columns in our University data are of type "int64". This means that they are 64 bit integers. But type "float64" column is a floating point value which means it contains decimals. Some datatypes are of "object", which means it contain numeric or strings or both vaues.

# ### 2. Which facts or principles do you need to know about US university system to make a good sense of this dataset? [2]
# 
# According to my evaluation, name of the university, state, total enrollment is important as it will give us an idea of the total number of students getting admission each year. Henceforth, the state code, type of university is also informative to know whether the university is public, private or for profit. Walking down the road, degree length gives the duration of the course plus, the private in_state_tuition, out_of_state_tuition and private in_state_total, out_of_state_total accounts for same data, so maybe we can merge it into one column to reduce number of columns. Early career pay, mid career pay gives a solid idea of how much they are going to earn during and after getting a degree. Make_world_better_percent also importantly contribute to the dataset as it gives an idea as how much we are contributing towars world with our knowledge.

# ### 3. Which things about the data you have at hand you do not know for sure and have to assume? [2]
# 
# Some of the data's I have less idea about thier existance are:
# 
# n_asian                        
# n_black                        
# n_hispanic                     
# n_pacific  
# n_total_minority   
# n_multiracial  
# n_unknown             
# n_white          
# room_and_board        
# stem_percent                 

# ### 4. In this project, you will have to predict out-of-state tuition fee. Do students choose university solely based on the cost? Which other factors might be important? [2]
# 
# No, student have not selected university only on the basis of cost. If we will compare total_enrollment of row no. 70 and 84, the out_of_state tuition fee of row number 70 is 15848 and total_enrollment = 12002, also out_of_state tuition fee of row number 84 is 12870 and total_enrollment = 3128. As we can see the university in row 70 costs higher and also have higher enrollment of student than university in row number 84, which is cheaper, it can deduce that students are not choosing university on the basis of cost.
# 
# Other factors which might be important other than the University are, type of university, early_career_pay and mid_career_pay.

# ### 5. To whom is this cost variable more important than the other three? Explain. [2]
# 
# 

# Other than those three the cost variable are important room_and_board. As we can see that the total cost whether it be in_state_tuition or out_of_state_tuition, the room_and_board are adding up in the total cost of both of these.

# ### 6. Formulate a reasonable business goal, execution of which would require analysis of out-of-state tuition fee in this dataset. [1]

# 

# ### 7. Which variable would you have to optimize for this goal? What variables would you have to constrain for it to be reasonable? [1]

# The variable which I will optimize will be total_enrollment.  I will put constrain on mid_career_pay.

# ### 8. Which data manipulations would you have to perform for an analysis or an ML system for this goal? What would you have to predict? Would classification or regression be more suitable, and why? [3]

# 

# ### Problem 3: EDA and Data preprocessing [10 points]

# In[8]:


# Number of time different states occurs in dataset.
print(df.state.value_counts())


# In[9]:


# Number of time different states_codes occurs in dataset. 
print(df.state_code.value_counts())


# In[10]:


# Different types of University
print(df.type.value_counts())


# In[11]:


# Different length period
print(df.degree_length.value_counts())


# In[12]:


# Counting all categorical variables in dataset
columns = ['state', 'state_code', 'type', 'degree_length']
for cols in columns:
    samples = df[cols].nunique()
    print("Total number of categories in", cols , " is ", samples)


# ### 1. Check how many categorical features exist and for each categorical feature see how many samples belong to each category. [2]
# 
# There are four categorical features which exist, those are: state, state_code, type and degree.length.
# 
# No. of samples belong to each category are:
# 
# state = 9;
# state_code = 9;
# type = 3;
# degree.length = 2;

# ### 2. Visualize the distributions of all features in the data set and summarize your findings. [6]

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df.plot.hist(subplots= True,bins=100,figsize=(15,30),edgecolor='black');


# ### Summary
# 
# Unnamed: We don't see much distriution as it is equal to the Serial number <br><br>
# total_enrollment: There we experience a right skewed plot. Which means number of students getting enrolled are as high as 5000, hence declintion is observed until 35000.<br><br>
# n_native: Few thousand native students are only enrolled in university <br><br>
# n_asian: This also contrubute as same as n_native, concluding that few students from asian countries went to USA for studying. <br><br>
# n_black: This shows a minor right skewed graph which means that we will find these category student on a average, <br><br>
# n_hispanic: It shows distribution almost equal to what we have seen in n_asian. Contributing only few thousands of students <br><br>
# n_pacific: It shows distribution almost equal to what wehave seen in n_asian.  Contributing only few thousands of students<br><br>
# n_nonresident: It shows distribution almost equal to what wehave seen in n_asian.  Contributing only few thousands of students<br><br>
# n_total_minority: Here we see a right skewed, indicating high frequescy of students coming from minorities. <br><br>
# n_multiracial: Multiracial shows excalty same distribution as n_pacific. Contributing few thousands of students. <br><br>
# n_unknown: This growth is similar to n_hispanic distribution. <br><br>
# n_white: This shows a right skewed showing high number of student in each intervals upto 45000 approx. <br><br>
# n_women: This is also a right skewed which show high number during 5000 thousand and then there is a certain drop gofing forward. <br><br>
# room_and_board: This histogram shows a rise and then fall of frequency before and after 10000's<br><br>
# in_state_tuition: Leading at approx 8000, in state tution shows quite a distribution of mean value in each frequency level.<br><br>
# in_state_total: The distribution shows a gradual incease and gradual decreas in the frequency, shooting up again at approx 32000, extending the mean upto 63000 approximate.<br><br>
# out_of_state_tuition: Show a little co-relation with in_state_total where the average frequency is below 5.<br><br>
# out_of_state_total: This represent a left skewed graph. Extending upto 62000 with highest frequency at 35000.<br><br>
# early_career_pay: Shows an average frequency distribution of below 10.<br><br>
# mid_career_pay: Shows a right skewed, with frequency of 1.0 approaching to 80000. Starting its mean value at 65000 mean approx and extending above 100000. <br><br>
# make_world_better_percent: This also contrubute as same as n_native. <br><br>
# stem_percent: This also contrubute as same as n_native

# ### 3. Split the data set into a training (70%) and a test (30%) set with the help of stratified sampling based on the degree length. Report mean and standard deviation for out_of_state_tuition in train and test data set. [2]

# In[14]:


# Splitting data into test and train dataset

from sklearn.model_selection import train_test_split
X = df.loc[:, df.columns != 'out_of_state_tuition']
y = df['out_of_state_tuition']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=df['degree_length'], 
                                                    test_size=0.3)


# In[15]:


y_train.describe()


# In[16]:


y_test.describe()


# ### Problem 4: Data Visualization [16 points]

# In[17]:


from scipy import stats, optimize, interpolate
columns = ['n_native', 'n_asian', 'n_black',
       'n_hispanic', 'n_pacific', 'n_nonresident', 'n_total_minority',
       'n_multiracial', 'n_unknown', 'n_white', 'n_women']
for col in columns:
    ax = sns.scatterplot(data =df[(np.abs(stats.zscore(df[col])) < 3)] , x=col, y='total_enrollment', hue='out_of_state_tuition', s=100)
    
    plt.style.use('fivethirtyeight')
    plt.title(col +' vs total_enrollment')
    plt.xlabel(col)
    plt.ylabel('total_enrollment')
    plt.legend(title = 'out_of_state_tuition' )
    plt.gcf().set_size_inches(7, 7)
    plt.show()


# ### 1. Describe what can be seen and interpret it. [5]
# 

# The scatterplot between n_native and enrollment shows an association that is positive, linear, and appears to be somewhat strong with a few outliers. We can also find good number of native students with the total enrollment of 32000 and few around 350 native students with enrollment approx 60000. <br><br>
# 
# The scatterplot between n_asian and enrollment shows an association that is positive, linear, and appears to be somewhat strong with a medium outliers. We can also find high density of asian students upto total_enrollment of 5000. <br><br>
# 
# The scatterplot between n_black and enrollment shows an association that is positive, non-linear, and appears to be somewhat strongly scatters. We can also find high density of black students during first few total enrollment of 5000. But this graph also depicts there are quite a number of students scattered around each interval of total_enrollment.<br><br>
# 
# The scatterplot between n_hispanic and enrollment shows an association that is positive, linear, and appears to be somewhat strong with a few outliers. The number of hispanic students are approx 2500 in the total enrollment of 30000 and below, and only few around 4000 with total enrollment of 60000.<br><br>
# 
# 
# The scatterplot between n_pacific and enrollment shows an association that is positive, linear, and appears to be somewhat strong with few outliers. Maximum of approx 38 pacific students can be observed for the enrollment upto 30000 and few students with medium out of state tuition are represented in the outliers.<br><br>
# 
# The scatterplot between n_resident and enrollment shows an association that is positive, curve, and appears to be somewhat strong with few outliers. A gap can br observed in nonresident students between around 800 to 1600. However, we can also find few hundered students in the total enrollment of around 60000.<br><br>
# 
# The scatterplot between n_minority and enrollment shows an association that is positive, non-linear, and appears to be somewhat strong with a medium outliers. Maximum number of students are getting chance in university with high enrollment of above 2000. <br><br>
# 
# The scatterplot between n_multiracial and enrollment shows an association that is positive, linear, and appears to be somewhat strong with a few outliers. The linear growth represent with the increase in total enrollment the number of student in multiracial are also getting increased.<br><br>
# 
# The scatterplot between n_unknown and enrollment shows an association that is positive, non-linear, and appears to be somewhat strong with a medium outliers. We can find good number of unknown students with the total enrollment of 30000 and few around 1700 unknown students with enrollment approx 60000.<br><br>
# 
# The scatterplot between n_white and enrollment shows an association that is positive, linear, and appears to be somewhat light with a few outliers. We can find a good linear growth of white students with high number of enrolled students.<br><br>
# 
# The scatterplot between n_women and enrollment shows an association that is positive, linear with no outliers. We can find a directly proportionality of women students with the number of enrolled students.
# 
# 

# ### 2. Which demographic characteristics are more pronounced for more expensive universities? [3]

# There is a high competition in almost all demographics for more expensive university. The most pronounced one I will consider all of them as we can see from the graph that on an average out_of_state_tuition=30000 dollar for all demogrpahics. Moving forward we can also observe high out_of_state_tuition= 40000 for few demographics such as n_asian ad n_white.

# ### 3. Write down your assumptions about why universities with lower tuition fees tend to attract certain groups more than more expensive universities. [3]

# It is very natural that the University with low tuition fees tends to attract certian group of people because moving to anothter countries for education  not only add up tuition fee on a student, but also adds up staying and fooding price. Hence it is also see that the native students are also a part of low tuition fees and reason could be that the native student belongs to that country and hence has to pay low, whereas, which is not in case with non native or students.

# ### 4. Which important considerations might these visualization conceal, rather than reveal? Produce visualizations that illustrate the findings that might be unobvious from scatterplots above. [5]

# Problem 4 part 4

# ### Problem 5: Correlation [15 points]
# Create and look at the correlation plots between the variables.

# ### 1. Which feature has the strongest correlation to the target feature? Plot the correlation between them exclusively. [2]

# In[18]:


def corr(dataframe,i):
    plt.figure(figsize=(6, 6))
    heatmap = sns.heatmap(pd.DataFrame(dataframe[dataframe.columns[1:]].corr()[i][:]).sort_values(ascending=False,by=i), annot=True)
    heatmap.set_title('Pearson Correlation Heatmap', fontdict={'fontsize':10}, pad=5);
    
corr(df,'out_of_state_tuition')


# The strongest correlation with respect to out_of_state_tuition is out_of_state_total.

# ### 2. Describe and interpret what you can see in this correlation plot. Any unusual occurrences? [3]

# In the pearson correlation heatmap, we can observe the most correlated or variables which are highly proportional to out_of_state_tuition are out_of_state_total, in_state_total, in_state_tuition, room_and_board which are relted upto 60%. The reason could be, as these corresponds to the dollars which are costing to the students, and it do also calls for university profit of tuiton fees and room and board.<br> Further going there a drop of atleast 25% in comaparision to mid_career_pay. We can also observe n_nonresident and early_career_pay corresponds equal that is 30% correlation with out_of_state_tuition. <br>Henceforth, there is a sharp drop of values from 27% to 9.5% when comparision hits different type of student in the dataframe.<br> We can also see negative correlation of values to -1.3% to -21%. Which means that out_of_state_tuition are not much related with values lesser than 0.

# ### 3. Which three features correlate the most with make_world_better_percent, a percentage of alumni who make the world a better place? [1]

# In[19]:


def corr1(dataframe,i):
    plt.figure(figsize=(6, 6))
    heatmap = sns.heatmap(pd.DataFrame(dataframe[dataframe.columns[1:]].corr()[i][:]).sort_values(ascending=False,by=i), annot=True)
    heatmap.set_title('Pearson Correlation Heatmap', fontdict={'fontsize':10}, pad=5);
    
corr1(df,'make_world_better_percent')


# Three features correlate the most with make_world_better_percent are out_of_state_total, room_and_board and n_white. These three features are inversely correlated with make_world_better_place.

# ### 4. Choose the strongest of these three correlations and propose four hypotheses about the nature of the link between these variables. [4]

# 4 Hypothesis are:<br><br>
# Hypothesis 1 - Lower is the cost of room_and_board, will have a positive impact on make_world_better_percent.<br><br>
# Hypothesis 2 - More students belonging from different ethenicity, less will be the discrimination and it will also give a postive impact on make_world_better_percent. As there will young talent from all around the world. <br><br>
# Hypothesis 3 - Lower the overall total cost of studying in University, more student will take addmission and probability of educated youth will increase which will make a positive impact on make_world_better_percent.<br><br>
# Hypothesis 4 - More students means more research will be done by University which ultimately impact on make_world_better_percent.<br><br>

# ### 5. Which hypothesis do you find the most plausible? Which sources are there supporting it? [2]

# The most plausible hypothesis I found is Hypothesis 4. The sources which support it are the number of students from different Ethenicity contributes high percentage of make_world_better_percent.

# ### 6. Which features do you lack in this dataset, which would have helped you determine whether your hypothesis is likely true? [1]

# The feature which is lacking in dataset which would have determine my hypothesis to be true is "Research dataset". The number of research done by each University would have greatly contribute to prove my hypothesis true.

# ### 7. Explain the difference between Pearson and Spearman correlation coefficients. Which of them have you just used in this problem? Which of them would be more feasible for the analysis you are doing?[2]

# In Pearson correlation the variables are directly proportional to each other, or we can say there is a linear relationship between two values. As the one change in one variable occurs, proportional change in other variable also occurs.<br><br>
# 
# In Spearman, there is monotonic relationship between two variables. Variable change togeather but not necessatily at constant rate.<br><br>
# 
# I have used Pearson correlation.<br><br>
# 
# Spearman correlation may have a better analysis.

# ### Problem 6: Correlation 2 [16 points]
# 
# Create new attributes by combining different features with each other.

# In[20]:


dfethnicity = df.n_native+df.n_asian+df.n_black+df.n_hispanic+df.n_pacific+df.n_nonresident+df.n_total_minority+df.n_multiracial+df.n_unknown+df.n_white+df.n_women
dfethnicity


# In[21]:


dfmid = pd.merge(df, dfethnicity.to_frame(),left_index=True, right_index=True)
dfmid = dfmid.drop(["n_native","n_asian","n_black","Unnamed: 0","n_pacific","n_nonresident","n_total_minority","n_multiracial","stem_percent","n_unknown","n_white","n_women","state_code","room_and_board","in_state_tuition","n_hispanic"], axis=1)
dfmid.columns.values[11] = "ethnicity"

dfmid


# ### 1. Explain why you think a combination will be useful before you create them. [2]

# The newly created combination "dfmid" is useful as I have combined all the different ethnicity into one which reduces the size and complexity of dataset drastically. Moving forward I removed more features as there were already combined into one. This combination now give a proper visualization of datas which may postively impact the dataset.

# ### 2. Check their correlation with the out_of_state_tuition in comparison with the other features from before. Show the correlations with the target feature in a descending order. [2]

# In[22]:


# Heat map of newly created dataset with respect to out_of_state_tuition
def make_corr(dataframe,i):
    plt.figure(figsize=(6, 6))
    heatmap = sns.heatmap(pd.DataFrame(dataframe[dataframe.columns[1:]].corr()[i][:]).sort_values(ascending=False,by=i), annot=True)
    heatmap.set_title('Pearson Correlation Heatmap', fontdict={'fontsize':10}, pad=5);
    
make_corr(dfmid,'out_of_state_tuition')


# ### 3. Do your newly created features have higher correlation with the target feature? [1]

# Yes, as represented from the above heatmap, I find these features to have higher correlation with out_of_state_tuition.

# ### 4. Take a look at the data from tuition_income dataset and decide which features or combinations of features you think will also be beneficial for prediction. Repeat steps 1-3 for them. Note that you may need to make some extra transformations to add them. [4]

# In[23]:


dftution.head()
dfmerge = pd.merge(dfmid,dftution,how="inner",on = "name")
dftuiton_filter = dftution[dftution["name"].isin(dfmerge.name.unique().tolist())]
dftuiton_filter = pd.concat([dftuiton_filter,pd.get_dummies(dftuiton_filter.income_lvl)],axis = 1).drop("income_lvl",axis=1)
dftuiton_filter_merge = pd.DataFrame()
low = []
mid = []
high = []
veryhigh = []
highest = []
total_price = []
year = []
campus = []
net_cost=[]
income_lvl = []

for i in dftuiton_filter.name.unique():
    temp = dftuiton_filter[dftuiton_filter["name"]==i]
    total_price.append(temp["total_price"].median())
    year.append(temp['year'].mode(dropna = True).iloc[0])
    campus.append(temp['campus'].mode(dropna = True).iloc[0])
    net_cost.append(temp['net_cost'].median())
    low.append(temp["0 to 30,000"].sum())
    mid.append(temp["30,001 to 48,000"].sum())
    high.append(temp["48_001 to 75,000"].sum())
    veryhigh.append(temp["75,001 to 110,000"].sum())
    highest.append(temp['Over 110,000'].sum())
    

dftuiton_filter_merge["name"] = dftuiton_filter.name.unique().tolist()
dftuiton_filter_merge["year"] = year
dftuiton_filter_merge["total_price"] = total_price
dftuiton_filter_merge["net_cost"] = net_cost
dftuiton_filter_merge["0 to 30,000"] = low
dftuiton_filter_merge["30,001 to 48,000"] = mid
dftuiton_filter_merge["48,001 to 75,000"] = high
dftuiton_filter_merge["75,001 to 110,000"] = veryhigh
dftuiton_filter_merge["Over 110,000"] = highest
dftuiton_filter_merge["campus"] = campus


# In[24]:


dfuni_tuition_combined = pd.merge(dfmid,dftuiton_filter_merge,how="inner",on = "name")


# In[25]:


# Displaying merge dataset of university and tuition income.
dfuni_tuition_combined


# The combination is useful because it narrow down data to more specific dataset which contain all the useful information needed.

# In[26]:


# Correlation heatmap of combine datatset with out_of_state_tuition
def corr(dataframe,i):
    plt.figure(figsize=(6, 6))
    heatmap = sns.heatmap(pd.DataFrame(dataframe[dataframe.columns[1:]].corr()[i][:]).sort_values(ascending=False,by=i), annot=True)
    heatmap.set_title('Pearson Correlation Heatmap', fontdict={'fontsize':10}, pad=5);
    
corr1(dfuni_tuition_combined,'out_of_state_tuition')


# Yes, my newly created features are having higher correlation with the target feature that is out_of_state_tuition.

# ### 8. Do any of your new features shine the new light on possible determinants of what makes students feel they are making the world the better place? Explain your insights. [2]
# 
# If we compare both the heat map that is <b>Correlation heatmap of combine datatset with out_of_state_tuition</b> and <b> Heat map of newly created dataset with respect to out_of_state_tuition</b>, we will not see much difference on make_world_better_percent. In Correlation heatmap of combine datatset with out_of_state_tuition the value is -0.26 and in Heat map of newly created dataset with respect to out_of_state_tuition is -0.21 i.e.; there is only a growth of -0.05 units value.

# ### Problem 7: Data cleaning [10 points]

# ### 1. Find out which variables contain missing values. If there are any, how many values are missing? [2]

# In[27]:


dfuni_tuition_combined.isnull().sum() * 100 / len(dfuni_tuition_combined)


# ### 2. Which approaches exist for handling missing data? Describe two approaches of how to handle them. Write one advantage and one disadvantage for each of those methods. [4]

# Approaches exist for handling missing data are: Multiple Imputatuion and K Nearest Neighbors(KNN)<br><br>
# Multiple Imputation: It is mainly for large dataset. In this uses subsituting data in place of  missing data.<br>
# Advantage: Results are readily interpreted.<br>
# Disadvantage:It assumes a random data in place of missing data.<br><br>
# KNN: This method uses Eucliden distance between neighbouring data cordinates to know how similar data is.<br>
# Advantage: It do not assumes data.<br>
# Disadvamtage: Since it stores all training data, hence it requires large memory capacity.
# 

# ### 3. Handle the missing data by methods you find the most suitable. Explain how you chose the method for each column. [3]

# In[28]:


# Handling missing datas.
dfuni_tuition_combined_imputation = dfuni_tuition_combined.fillna(dfuni_tuition_combined.mean())


# In[29]:


dfuni_tuition_combined_imputation.head()


# While cleaning the data we observed 3 features that is are early_career_pay, mid_career_pay, make_world_better_percent denotes missing values. So I implemented Multiple Imputation technique to fill the missing values as thats is the best technique for large dataset.

# In[30]:


# Splitting values into train and test set with 70% versus 30% ratio, respectively.
dfuni_tuition_combined_imputation.drop(["name"],inplace=True,axis=1) 
dfuni_tuition_combined_imputation["year"] = dfuni_tuition_combined_imputation.year.apply(str)
X = dfuni_tuition_combined_imputation.loc[:, dfuni_tuition_combined_imputation.columns != 'out_of_state_tuition']
y = dfuni_tuition_combined_imputation['out_of_state_tuition']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=dfuni_tuition_combined_imputation['degree_length'], 
                                                    test_size=0.3,random_state = 131)


# ### 4. Handle the categorical data with one-hot-encoding. [1]

# In[31]:


X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


# ### Problem 8: Feature scaling [9 points]
# 
# ### 1. What feature scaling methods exist? Name five methods and write a short description for three of them. Pick one to use for this data set. [6]

# Feature Scaling is the process by which we transform data into a better version. It is use to normalize the features in the dataset into a finite range.<br><br>
# Five Feature Scaling methods are:<br><br>
# <b>Absolute Maximum Scaling:</b> In this scaling, datas are scaled into maximum value and the results varies approximately within the range of -1 to 1<br><br>
# <b>Min-Max Scaling:</b> It is feature pre-processing technique in which data are scaled in fixed range of 0 to 1.<br><br>
# <b>Normalization:</b> It is the feature by which numeric columns in dataset changes to common scale.<br><br>
# <b>Standardization:</b> In this method the datas are converted into uniform format which is further used to different operations.<br><br>
# <b>Robust Scaling:</b> In this the algorithm scale features that are robust to outliers.

# In[32]:


dfuni_tuition_combined_imputation.head()


# In[33]:


# Performing Standardization method for my dataset.

from sklearn.preprocessing import StandardScaler
def scal_independent(DataFrame):
    scale = StandardScaler()
    cols = ['total_enrollment','in_state_total', 'out_of_state_total','early_career_pay', 'mid_career_pay', 
            'make_world_better_percent','total_price', 'net_cost','ethnicity']
    DataFrame[cols] = scale.fit_transform(DataFrame[cols])
    
    return DataFrame

def scal_dependent(DataFrame):
    dataframe = scale(DataFrame)
    return dataframe

X_train = scal_independent(X_train)
X_test = scal_independent(X_test)
y_train = scal_dependent(y_train)
y_test = scal_dependent(y_test)


# In[34]:


# Dropping non-numeric features from imputed dataset.

X_train.drop(list(set(X_train.columns.tolist())- set(X_test.columns.tolist())),axis=1,inplace=True)


# In[35]:


X_test.drop('state_Montana',inplace = True, axis = 1)


# ### 2. Find the feature importance with a quick Random Forest and show them in a plot. What insights do you get out of it? [3]

# In[36]:


from sklearn.datasets import make_regression
X, y = make_regression(n_samples=165, n_features=52, n_informative=5, random_state=1)
mdl = RandomForestRegressor()
mdl.fit(X_train, y_train)
feature_scores = pd.Series(mdl.feature_importances_, index=X_train.columns).sort_values(ascending=False)
f, ax = plt.subplots(figsize=(30, 24))
ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=df)
ax.set_title("Visualize feature scores of the features")
ax.set_yticklabels(feature_scores.index)
ax.set_xlabel("Feature importance score")
ax.set_ylabel("Features")
plt.show()


# I can observe that not much feature shows up with the Random Forest feature scaling method.

# ### Problem 9: Test data [6 points]
# 
# Perform the same data cleaning steps from questions 7 and 8 for the test data set. (Replace missing
# values, handle the categorical data, add the new features, and scale the features) [6]
# 
# #### Already perform above in question 7 & 8.

# ### Problem 10: Regression [19 points]
# 
# Select and train the following regression models on the training set. Linear model, support vector
# regression, and random forest. 

# ### 1. Evaluate the three regression models on the test set. Which model performs best? [9]

# In[37]:


# Performing Linear Regression.

r =LinearRegression()
model = r.fit(X_train, y_train)
y_pred = r.predict(X_test)
residuals = y_test - y_pred
mae = round(metrics.mean_absolute_error(y_test, y_pred),2)
mse = round(metrics.mean_squared_error(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
mape = round(np.mean(abs(residuals/y_test)),2)
accuracy = str(100-(round(np.mean(abs(residuals/y_test)),4)*100)) +' %'
nRMSE = round((rmse/70.127909),2)
aError = round(np.mean(abs(residuals)),2)
error_df = pd.DataFrame(data = [[mae, mse, rmse,mape, accuracy, nRMSE,aError]], 
                        columns = ['MAE', 'MSE', 'RMSE','MAPE','Model Accuracy','Normalized RMSE','Average Error'], 
                        index = ['Linear Regression'])
print(error_df)


# In[38]:


# Performing Random Forest Regression
r = RandomForestRegressor()
model = r.fit(X_train, y_train)
y_pred = r.predict(X_test)
residuals = y_test - y_pred
mae = round(metrics.mean_absolute_error(y_test, y_pred),2)
mse = round(metrics.mean_squared_error(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
mape = round(np.mean(abs(residuals/y_test)),2)
accuracy = str(100-(round(np.mean(abs(residuals/y_test)),4)*100)) +' %'
nRMSE = round((rmse/70.127909),2)
aError = round(np.mean(abs(residuals)),2)
error_df = pd.DataFrame(data = [[mae, mse, rmse,mape, accuracy, nRMSE,aError]], 
                        columns = ['MAE', 'MSE', 'RMSE','MAPE','Model Accuracy','Normalized RMSE','Average Error'], 
                        index = ['Random Forest Regression'])
print(error_df)


# In[39]:


# Performing Support Vector Regression
r =SVR ()
model = r.fit(X_train, y_train)
y_pred = r.predict(X_test)
residuals = y_test - y_pred
mae = round(metrics.mean_absolute_error(y_test, y_pred),2)
mse = round(metrics.mean_squared_error(y_test, y_pred),2)
rmse = round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2)
mape = round(np.mean(abs(residuals/y_test)),2)
accuracy = str(100-(round(np.mean(abs(residuals/y_test)),4)*100)) +' %'
nRMSE = round((rmse/70.127909),2)
aError = round(np.mean(abs(residuals)),2)
error_df = pd.DataFrame(data = [[mae, mse, rmse,mape, accuracy, nRMSE,aError]], 
                        columns = ['MAE', 'MSE', 'RMSE','MAPE','Model Accuracy','Normalized RMSE','Average Error'], 
                        index = ['SVR'])
print(error_df)


# #####  Solution: 
# Random forest model performs the best.

# ### 2. Explain one approach about how the models can be further optimized. [3] 

# One approach by which a model can be further be optimize is through isotonic regression. Isotonic Regression is a technique by which a free-form line is fitted to a sequence of observation in such a way that fitted line is neither non-decreasing or non-increasing everywhere rather should be close to the observation as much as possible.

# ### 3. Based on the data you have at hand and your background knowledge, which problems might you encounter when trying use your best model to predict out-of-state tuition fee for the next year? [2] 

# 

# ### 4. Name and explain two accuracy metrics for regression other than MSE or RMSE. Are any of them better suited for evaluating your models, given the goal you have formulated earlier? [2]

# Two accuracy model except MSE and MAPE are: <br><br> 
# MAE (Mean Absolute Error): It is the average measure of errors in a set of models without taking vector into account.<br><br>
# MAPE (Mean Absolute Percentage Error): It is the percentage of the average difference between forecasted and true values.<br><br>
# Yes they both sucessfully suited for the evaluation of my model.
# 

# ### 5. What has to change in real life to completely invalidate the results you get and accuracy of your model? I.e., what laws, natural events, societal changes have to happen to make conclusions based on this dataset inadmissible for further decision-making? [3]

# If out_of_state_tuition and in_state_total feature gets invalidate, the results and accuracy of the whole model will become inaccurate. If such a law is passed that no fees will be taken by the students the dataset will become inadmissible for further decision making. 

# ### Problem 11: Classification [21 points]
# 
# Categorize the target variable (out_of_state_tuition) into five categories and build a classification model
# for the above pre-processed data.

# ### 1. Train the following classification models on the training set for classification and evaluate the models on the test set: SVM, k-NN, and Random Forest. [9]

# In[40]:


# Function for classification.
def classification(Xtrain, Xtest, ytrain, ytest, classifier):
    cls = classifier
    cls.fit(Xtrain, ytrain)
    ypred = cls.predict(Xtest)
    print(f"Accuracy of the classifier is: {accuracy_score(ytest, ypred)}")
    print(f"Precision Score of the classifier is: {precision_score(ytest, ypred,average='macro')}")
    print(f"Recall Score of the classifier is: {recall_score(ytest, ypred,average='macro')}")
    print(f"F1 Score of the classifier is: {f1_score(ytest, ypred,average='macro')}")
    plot_confusion_matrix(classifier, Xtest, ytest)


# In[41]:


# Creating bins for y_train and y_test
y_train_class = pd.cut(y_train, bins = [-3,-1,0,1,3,4], include_lowest = True, labels=[1,2,3,4,5])
y_test_class = pd.cut(y_test, bins = [-3,-1,0,1,3,4], include_lowest = True, labels=[1,2,3,4,5])


# In[42]:


# Plotting Confusion matrix for model k-NN 
classification(X_train, X_test, y_train_class, y_test_class, KNeighborsClassifier())


# In[43]:


# Plotting Confusion matrix for model SVM 
classification(X_train, X_test, y_train_class, y_test_class, SVC())


# In[44]:


# Plotting Confusion matrix for model Random Forest
classification(X_train, X_test, y_train_class, y_test_class, RandomForestClassifier())


# ### 3. Which model performs best based on which evaluation method? [2]

# Random Forest perform best on our model. The can be seen by the accuracy of the classifier which is 88.4615% approx

# ### 4. Explain the evaluation method you used. [2]

# The evaluation method I have used here are: <br><br>
# <b>SVM:</b> SVM is a supervised learning method. These are used in high dimenation spaces.<br><br><b>k-NN:</b> This method does not make assumption on underlying datas. It perform action at the time of classifcation, hence it is also known as lazy lerner algorithm.<br><br>and,<br><br> <b>Random Forest:</b> It uses ensemble learning that is it combines classifiers to perform solution to complex problem.

# ### 5. For which applications would classification into these artificial categories be more useful than regression? Name at least two. [2]

# Application where classification will be more useful than regression is when there will discrete value. In these artificial category we can opt for "type" and "degree_length".

# ### 6. For which applications would regression on original values be more useful? Name at least two. [2]

# Regression is used in contunues values. Here we can use in "out_of_state_total" and "total_enrollment"

# ### Part 2
# 
# ### Problem 12: Text Mining [40 points]
# 
# The dataset GoodReads.csv contains book descriptions. Load this dataset and draw random sample of size 5000 from it, setting the seed to 45.

# In[45]:


Goodreads = pd.read_csv('GoodReads.csv')
Goodreads.head()


# In[46]:


# dfgr is the name of the data frame of Good Reads book

dfgr = Goodreads.sample(n=5000, random_state=45)
dfgr.head()


# ### 1. Plot and describe the distribution of average ratings of the books.[2]

# In[47]:


fig = px.histogram(x = dfgr['rating'], width=800, marginal='box', labels={'col':'col'})
fig.update_layout(title = 'Average Ratings of books')
fig.show()


# ### 2. Clean the data in your dataframe. Explain your reasoninng for selecting the cleaning methods you decided to use.[3]

# In[48]:


dfgr = dfgr.drop(['img', 'isbn', 'isbn13', 'link'], axis = 1)
dfgr.head()


# I decided to drop image, link and isbn, because, img and link are the web-links for the picture and the PDF file of the book, I also ommitted the isbn, which will not be using.

# ### 3. Categorize the rating variable into several categories. Explain how many categories you decided to have, where you drew the line, and why you decided to do it this way[5]

# In[49]:


category = pd.cut(dfgr['rating'], bins = [0, 1, 2, 3, 4, 5.1], include_lowest = True, labels=['Very Poor', 
                                                                                          'Poor', 
                                                                                          'Neutral',
                                                                                          'Good',
                                                                                          'Very Good'])
dfgr['Rating Category'] = category
dfgr.head()


# ### 4. Get a feel about the distribution of text lengths of the book descriptions by adding a new feature for the length of each message. Check the statistical values. [2]

# In[50]:


msgLen = dfgr['desc'].str.len()
dfgr['Description Length'] = msgLen
dfgr.head()


# ### 5. Visualize the distribution of text lengths with a histogram, where the colouring is according to the rating category. [2]

# In[51]:


x1 = dfgr.loc[dfgr['Rating Category'] == 'Very Poor', 'Description Length']
x2 = dfgr.loc[dfgr['Rating Category'] == 'Poor', 'Description Length']
x3 = dfgr.loc[dfgr['Rating Category'] == 'Neutral', 'Description Length']
x4 = dfgr.loc[dfgr['Rating Category'] == 'Good', 'Description Length']
x5 = dfgr.loc[dfgr['Rating Category'] == 'Very Good', 'Description Length']

kwargs = dict(alpha = 0.5, bins = 80)

plt.figure(figsize=(20,9))
plt.hist(x1, **kwargs, color='#465362', label='Very Poor')
plt.hist(x2, **kwargs, color='#011936', label='Poor')
plt.hist(x3, **kwargs, color='#C2EABD', label='Neutral')
plt.hist(x4, **kwargs, color='#F9DC5C', label='Good')
plt.hist(x5, **kwargs, color='#ED254E', label='Very Good')
plt.gca().set(title = 'Frequency Histogram of Text Length',  ylabel = 'Frequency', xlabel = 'Text Length')
plt.legend(loc = 'upper right', prop = {'size': 20});


# ### 6. Create a random stratified training and test split (70/30 split). Verify the correct proportions of the splitted data sets by creating proportion table. [1]

# In[52]:


X = dfgr.loc[:, dfgr.columns != 'Description Length']
y = dfgr['Description Length']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=dfgr['Rating Category'],test_size=0.3)

grStr = {'X': [len(X_train)/(len(X_train)+len(X_test)),len(X_test)/(len(X_train)+len(X_test))],
                     'Y': [len(y_train)/(len(y_train)+len(y_test)),len(y_test)/(len(y_train)+len(y_test))]}

grProportion = pd.DataFrame(grStr)
grProportion.head()


# In[53]:


# Train dataset

Xtrain_tab = pd.crosstab(index = X_train['Rating Category'],  
                         columns = 'Count') 
Xtrain_tab


# In[54]:


# Test dataset

Xtest_tab = pd.crosstab(index = X_test['Rating Category'],  
                        columns = 'Count')

Xtest_tab


# ### 7. Tokenize the descriptions with help of the quanteda package and illustrate the effect of each action by showing the content of some of the reviews in the training data set:
# 
# Remove the numbers, punctuations, symbols, and hyphens. Turn the texts in the reviews into lower case. Remove the stopwords in the modified training set (the tokens) with the predefined stopword list of quanteda. Perform stemming on the tokens. [5]

# In[55]:


import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# In[56]:


# Getting rid of NaN in the description column, to prepare it for tokenization

dfgr['desc']=dfgr['desc'].fillna('')
dfgr.head(40)


# In[57]:


# Creating the new string with lowercase desctiption

dfgr['Lower Case'] = dfgr['desc'].str.lower() 
dfgr.head(25)


# In[58]:


# Getting rid of the punctuations in the Lower Case variable

dfgr['Lower Case'] = dfgr['Lower Case'].str.replace('[^\w\s]', '')
dfgr.head()


# In[59]:


# Deleting stopwords

stop = stopwords.words('english')
dfgr['Without Stopwords'] = dfgr['Lower Case'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
dfgr.head()


# In[60]:


# Create a new column with tokenized lowercase column

dfgr['Tokenized Description'] = dfgr.apply(lambda row: nltk.word_tokenize(row['Without Stopwords']), axis = 1)
dfgr.head()


# In[61]:


# Stemming

stemmer = SnowballStemmer('english')
dfgr['Stemmed Token'] = dfgr['Tokenized Description'].apply(lambda x: [stemmer.stem(y) for y in x])
dfgr.head()


# ### 8. Create a bag-of-words model and add bi-grams to the normal feature matrix. [2]

# In[62]:


vocab = []

for i in dfgr['Stemmed Token']:
    for x in i:
            vocab.append(x)
        
        
vocab


# In[63]:


# Getting rid of duplicated values

vocabDupl = list(dict.fromkeys(vocab))
vocabDupl


# In[64]:


def convert(org_list, separator = ' '):
    
    return separator.join(org_list)

sentences = []

for j in dfgr['Stemmed Token']:
    sentence = ''
    sentence = convert(j)
    sentences.append(sentence)
    
sentences


# In[65]:


sentence100 = sentences[:100]
print(sentence100)


# In[66]:


# Transforming given text into vectors

from sklearn.feature_extraction.text import CountVectorizer
import itertools
count_vec = CountVectorizer()
cdf = count_vec.fit_transform(sentence100)
bag = pd.DataFrame(cdf.toarray(), columns = count_vec.get_feature_names())
bag


# In[67]:


# Bi-Gram for 100 sentences

bi_gram = [(x, i.split()[j + 1]) for i in sentence100
       for j, x in enumerate(i.split()) if j < len(i.split()) - 1]

print ("The BiGram are :\n\n " + str(bi_gram))


# ### 9. Build a function for relative term frequency (TF) and another one to calculate the inverse document frequency (IDF).

# In[68]:


# Calculating TF

def tf(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict


# In[69]:


# Calculating IDF

def idf(docList):
    import math
    idfDict = dict()
    n = len(docList)
    
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val == 0:
                idfDict[word] += 1.0
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(n / float(val))
        
    return idfDict


# In[70]:


import __future__ 
from __future__ import division

# Calculating TFIDF
def computeTFIDF(tfBow,idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val * idfs[word]
    return tfidf


# In[71]:


# Create a dictionary, to count the words in a certain second

wordDict1 = dict.fromkeys(vocabDupl, 0)
wordDict2 = dict.fromkeys(vocabDupl, 0)

# Counting words

for word in dfgr['Stemmed Token'].values[0]:
    wordDict1[word] += 1

for word in dfgr['Stemmed Token'].values[1]:
    wordDict1[word] += 1

wordDict1


# In[72]:


#  Convert into Data Frame

pd.DataFrame([wordDict1])


# In[74]:


tfBow1 = tf(wordDict1, dfgr['Stemmed Token'].values[0])
tfBow2 = tf(wordDict1, dfgr['Stemmed Token'].values[1])
tfBow1


# In[75]:


idfs = idf([wordDict1, wordDict2])


# In[76]:


tfidfBow1 = computeTFIDF(tfBow1, idfs)
tfidfBow2 = computeTFIDF(tfBow2, idfs)
pd.DataFrame([tfidfBow1, tfidfBow2])

