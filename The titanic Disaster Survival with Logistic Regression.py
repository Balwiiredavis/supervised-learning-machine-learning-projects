#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Packages / libraries
import os 
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from math import sqrt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


raw_data = pd.read_csv('Titanic-Dataset.csv')


# In[3]:


raw_data.shape


# In[4]:


#runs the first 5 rows
raw_data.head()


# In[5]:


# Checking for null values
raw_data.isnull().sum()


# In[6]:


# Visualize the NULL observations

raw_data[raw_data['Cabin'].isnull()]


# In[7]:


# Deleting the NULL values
raw_data = raw_data.dropna(subset = ['Cabin'])

# Visualize the NULL observations
raw_data.isnull().sum()


# In[8]:


# Deleting the NULL values
raw_data = raw_data.dropna(subset = ['Age'])

# Visualize the NULL observations
raw_data.isnull().sum()


# In[9]:


# Deleting the NULL values
raw_data = raw_data.dropna(subset = ['Embarked'])

# Visualize the NULL observations
raw_data.isnull().sum()


# In[10]:


raw_data.shape


# In[11]:


# Investigate all the elements whithin each Feature 
for column in raw_data:
    unique_vals = np.unique(raw_data[column])
    nr_values = len(unique_vals)
    if nr_values < 10:
        print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
    else:
        print('The number of values for feature {} :{}'.format(column, nr_values))


# In[12]:


# Visualize the data using seaborn Pairplots
g = sns.pairplot(raw_data)


# In[13]:


raw_data.columns


# In[14]:


p = sns.pairplot(raw_data,hue = 'Survived')


# In[15]:


sns.countplot(x = 'Survived', data = raw_data, palette = 'Set3')


# In[16]:


raw_data.columns


# In[17]:


# Looping through all the features by our y variable - see if there is relationship

features = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

for f in features:
    sns.countplot(x = f, data = raw_data, palette = 'Set3',hue = 'Survived')
    plt.show()


# In[29]:


# Selecting usefull columns only
new_raw_data = raw_data[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp',
       'Parch', 'Fare']]

#visualize the raw data
new_raw_data.head()


# In[ ]:





# In[30]:


# Feature selection and splitting the data
X = new_raw_data.drop('Survived', axis=1).values
y = new_raw_data['Survived']


y = (y == 'M').astype(int)

print(X.shape)
print(y.shape)


# In[31]:


# Run a Tree-based estimators (i.e. decision trees & random forests)
dt = DecisionTreeClassifier(random_state=15, criterion = 'entropy', max_depth = 10)
dt.fit(X,y)


# In[32]:


# Running Feature Importance

fi_col = []
fi = []

for i,column in enumerate(new_raw_data.drop('Survived', axis=1)):
    print('The feature importance for {} is : {}'.format(column, dt.feature_importances_[i]))
    
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])


# In[33]:


fi_col
fi

fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns = ['Feature','Feature Importance'])
fi_df


# In[34]:


# Ordering the data
fi_df = fi_df.sort_values('Feature Importance', ascending = False).reset_index()


# In[35]:


# Creating columns to keep
columns_to_keep = fi_df['Feature'][0:4]

fi_df


# In[36]:


# Print the shapes

print(new_raw_data.shape)
print(new_raw_data[columns_to_keep].shape)


# In[38]:


# Split the data into X & y

X = new_raw_data[columns_to_keep].values
X

y = new_raw_data['Survived']
y = y.astype(int)
y

print(X.shape)
print(y.shape)


# In[39]:


# Hold-out validation

# first one
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=15)

# Second one
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size = 0.9, test_size=0.1, random_state=15)

print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)

print(y_train.shape)
print(y_test.shape)
print(y_valid.shape)


# In[40]:


# Investigating the distr of all ys

ax = sns.countplot(x = y_valid, palette = "Set3")


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


# Training my model

log_reg = LogisticRegression(random_state=10, solver = 'lbfgs')

log_reg.fit(X_train, y_train)


# In[45]:


log_reg.predict(X_train)
y_pred = log_reg.predict(X_train)


# In[46]:


# predict_proba - Probability estimates
pred_proba = log_reg.predict_proba(X_train)


# In[47]:


log_reg.coef_


# In[49]:


from sklearn.metrics import classification_report


# In[50]:


# Evaluating the Model

# Accuracy on Train
print("The Training Accuracy is: ", log_reg.score(X_train, y_train))

# Accuracy on Test
print("The Testing Accuracy is: ", log_reg.score(X_test, y_test))


# Classification Report
print(classification_report(y_train, y_pred))


# In[53]:


# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[55]:


from sklearn.metrics import confusion_matrix


# In[56]:


# Visualizing cm

cm = confusion_matrix(y_train, y_pred)

cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

plot_confusion_matrix(cm_norm, classes = log_reg.classes_, title='Confusion matrix')


# In[57]:


log_reg.classes_


# In[58]:


cm.sum(axis=1)
cm_norm


# In[59]:


cm


# In[60]:


cm.sum(axis=0)


# In[61]:


np.diag(cm)


# In[62]:


# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
print("The True Positive Rate is:", TPR)

# Precision or positive predictive value
PPV = TP / (TP + FP)
print("The Precision is:", PPV)

# False positive rate or False alarm rate
FPR = FP / (FP + TN)
print("The False positive rate is:", FPR)

# False negative rate or Miss Rate
FNR = FN / (FN + TP)
print("The False Negative Rate is: ", FNR)

#Total averages :
print("")
print("The average TPR is:", TPR.sum()/2)
print("The average Precision is:", PPV.sum()/2)
print("The average False positive rate is:", FPR.sum()/2)
print("The average False Negative Rate is:", FNR.sum()/2)


# In[65]:


from sklearn.metrics import log_loss


# In[66]:


# Logarithmic loss - or Log Loss - or cross-entropy loss

# Running Log loss on training
print("The Log Loss on Training is: ", log_loss(y_train, pred_proba))

# Running Log loss on testing
pred_proba_t = log_reg.predict_proba(X_test)
print("The Log Loss on Testing Dataset is: ", log_loss(y_test, pred_proba_t))


# In[67]:


# Hyper Parameter Tuning
np.geomspace(1e-5, 1e5, num=20)


# In[68]:


# Creating a range for C values
np.geomspace(1e-5, 1e5, num=20)

# ploting it
plt.plot(np.geomspace(1e-5, 1e5, num=20)) 
plt.plot(np.linspace(1e-5, 1e5, num=20)) 
# plt.plot(np.logspace(np.log10(1e-5) , np.log10(1e5) , num=20)) # same as geomspace



# In[69]:


# Looping over the parameters

C_List = np.geomspace(1e-5, 1e5, num=20)
CA = []
Logarithmic_Loss = []

for c in C_List:
    log_reg2 = LogisticRegression(random_state=10, solver = 'lbfgs', C=c)
    log_reg2.fit(X_train, y_train)
    score = log_reg2.score(X_test, y_test)
    CA.append(score)
    print("The CA of C parameter {} is {}:".format(c, score))
    pred_proba_t = log_reg2.predict_proba(X_test)
    log_loss2 = log_loss(y_test, pred_proba_t)
    Logarithmic_Loss.append(log_loss2)
    print("The Logg Loss of C parameter {} is {}:".format(c, log_loss2))
    print("")


# In[71]:


# putting the outcomes in a Table

# reshaping
CA2 = np.array(CA).reshape(20,)
Logarithmic_Loss2 = np.array(Logarithmic_Loss).reshape(20,)

# zip
outcomes = zip(C_List, CA2, Logarithmic_Loss2)

#df
df_outcomes = pd.DataFrame(outcomes, columns = ["C_List", 'CA2','Logarithmic_Loss2'])

#print
df_outcomes

# Ordering the data (sort_values)
df_outcomes.sort_values("Logarithmic_Loss2", ascending = True).reset_index()


# In[72]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
kf = KFold(n_splits=3, random_state=0, shuffle=True)

# Logistic Reg CV
Log_reg3 = LogisticRegressionCV(random_state=15, Cs = C_List, solver ='lbfgs')
Log_reg3.fit(X_train, y_train)
print("The CA is:", Log_reg3.score(X_test, y_test))
pred_proba_t = Log_reg3.predict_proba(X_test)
log_loss3 = log_loss(y_test, pred_proba_t)
print("The Logistic Loss is: ", log_loss3)

print("The optimal C parameter is: ", Log_reg3.C_)


# In[73]:


# Training a Dummy Classifier

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
score = dummy_clf.score(X_test, y_test)

pred_proba_t = dummy_clf.predict_proba(X_test)
log_loss2 = log_loss(y_test, pred_proba_t)

print("Testing Acc:", score)
print("Log Loss:", log_loss2)


# In[74]:


# Final Model 

log_reg3 = LogisticRegression(random_state=10, solver = 'lbfgs', C=0.014384)
log_reg3.fit(X_train, y_train)
score = log_reg3.score(X_valid, y_valid)

pred_proba_t = log_reg3.predict_proba(X_valid)
log_loss2 = log_loss(y_valid, pred_proba_t)

print("Testing Acc:", score)
print("Log Loss:", log_loss2)


# In[ ]:




