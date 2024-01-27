#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


breast_cancer = pd.read_csv("Breast_cancer_data.csv")


# In[4]:


print(breast_cancer)


# In[5]:


breast_cancer.head(5)


# In[7]:


#Data preprocessing
breast_cancer.isnull().sum()


# In[8]:


# Investigate all the elements whithin each Feature 

for column in breast_cancer:
    unique_values = np.unique(breast_cancer[column])
    nr_values = len(unique_values)
    if nr_values <= 10:
        print("The number of values for feature {} is: {} -- {}".format(column, nr_values, unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column, nr_values))


# In[52]:


# Deleting the outlier

breast_cancer = breast_cancer[breast_cancer['mean_area'] < 1800]

breast_cancer.shape


# In[53]:


# Visualize the data using seaborn Pairplots

g = sns.pairplot(breast_cancer)


# In[54]:


features_to_visualize = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']

# Melt the dataframe to long format
melted_df = pd.melt(breast_cancer, id_vars='diagnosis', value_vars=features_to_visualize)

# Visualize boxplot for multiple features
plt.figure(figsize=(12, 6))
sns.boxplot(x='variable', y='value', hue='diagnosis', data=melted_df)
plt.xticks(rotation=45)
plt.show()


# In[55]:


p = sns.pairplot(breast_cancer,hue = 'diagnosis')


# In[56]:


sns.countplot(x = 'diagnosis', data = breast_cancer, palette = 'Set3')


# In[15]:


breast_cancer.columns


# In[57]:


# Looping through all the features by our y variable - see if there is relationship

features = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
       'mean_smoothness']

for f in features:
    sns.countplot(x = f, data = breast_cancer, palette = 'Set3',hue = 'diagnosis')
    plt.show()


# In[47]:


breast_cancer.head()


# In[58]:


# Feature selection and splitting the data
X = breast_cancer.drop('diagnosis', axis=1).values
y = breast_cancer['diagnosis']


y = (y == 'M').astype(int)

print(X.shape)
print(y.shape)


# In[35]:


from sklearn.ensemble import RandomForestClassifier


# In[61]:


# Run a Tree-based estimators (i.e. decision trees & random forests)


dt = DecisionTreeClassifier(random_state=15, criterion = 'entropy', max_depth = 10)
dt.fit(X,y)


# In[63]:


# Running Feature Importance

fi_col = []
fi = []

for i,column in enumerate(breast_cancer.drop('diagnosis', axis=1)):
    print('The feature importance for {} is : {}'.format(column, dt.feature_importances_[i]))
    
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])


# In[51]:


fi_col
fi

fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns = ['Feature','Feature Importance'])
fi_df


# In[34]:


# Ordering the data
fi_df = fi_df.sort_values('Feature Importance', ascending = False).reset_index()


# In[64]:


# Creating columns to keep
columns_to_keep = fi_df['Feature'][0:4]

fi_df


# In[67]:


# Print the shapes

print(breast_cancer.shape)
print(breast_cancer[columns_to_keep].shape)


# In[68]:


# Split the data into X & y

X = breast_cancer[columns_to_keep].values
X

y = breast_cancer['diagnosis']
y = y.astype(int)
y

print(X.shape)
print(y.shape)


# In[69]:


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


# In[70]:


# Investigating the distr of all ys

ax = sns.countplot(x = y_valid, palette = "Set3")


# In[72]:


# Training my model

log_reg = LogisticRegression(random_state=10, solver = 'lbfgs')

log_reg.fit(X_train, y_train)


# In[75]:


log_reg.predict(X_train)
y_pred = log_reg.predict(X_train)


# In[76]:


# predict_proba - Probability estimates
pred_proba = log_reg.predict_proba(X_train)


# In[77]:


log_reg.coef_


# In[78]:


# Evaluating the Model

# Accuracy on Train
print("The Training Accuracy is: ", log_reg.score(X_train, y_train))

# Accuracy on Test
print("The Testing Accuracy is: ", log_reg.score(X_test, y_test))


# Classification Report
print(classification_report(y_train, y_pred))


# In[79]:


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


# In[82]:


# Visualizing cm

cm = confusion_matrix(y_train, y_pred)

cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

plot_confusion_matrix(cm_norm, classes = log_reg.classes_, title='Confusion matrix')


# In[83]:


log_reg.classes_


# In[84]:


cm.sum(axis=1)
cm_norm


# In[85]:


cm


# In[86]:


cm.sum(axis=0)


# In[87]:


np.diag(cm)


# In[88]:


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


# In[89]:


# Logarithmic loss - or Log Loss - or cross-entropy loss

# Running Log loss on training
print("The Log Loss on Training is: ", log_loss(y_train, pred_proba))

# Running Log loss on testing
pred_proba_t = log_reg.predict_proba(X_test)
print("The Log Loss on Testing Dataset is: ", log_loss(y_test, pred_proba_t))


# In[90]:


# Hyper Parameter Tuning
np.geomspace(1e-5, 1e5, num=20)


# In[91]:


# Creating a range for C values
np.geomspace(1e-5, 1e5, num=20)

# ploting it
plt.plot(np.geomspace(1e-5, 1e5, num=20)) 
plt.plot(np.linspace(1e-5, 1e5, num=20)) 
# plt.plot(np.logspace(np.log10(1e-5) , np.log10(1e5) , num=20)) # same as geomspace



# In[92]:


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


# In[93]:


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


# In[94]:


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


# In[95]:


# Training a Dummy Classifier

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
score = dummy_clf.score(X_test, y_test)

pred_proba_t = dummy_clf.predict_proba(X_test)
log_loss2 = log_loss(y_test, pred_proba_t)

print("Testing Acc:", score)
print("Log Loss:", log_loss2)


# In[97]:


# Final Model 

log_reg3 = LogisticRegression(random_state=10, solver = 'lbfgs', C=100000.000000)
log_reg3.fit(X_train, y_train)
score = log_reg3.score(X_valid, y_valid)

pred_proba_t = log_reg3.predict_proba(X_valid)
log_loss2 = log_loss(y_valid, pred_proba_t)

print("Testing Acc:", score)
print("Log Loss:", log_loss2)


# In[ ]:




