import numpy as np
import pandas as pd
import tensorflow as tf
import math
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualisations
%matplotlib inline
sns.set_style('white')

# Importing the dataset
dataset = pd.read_csv('data.csv')
dataset.head()
dataset.shape
dataset.columns
dataset.info()
dataset.describe()
dataset.isnull().sum()
dataset['outcome'].value_counts()


X = dataset.drop(columns = 'outcome')
y = dataset[['outcome']]

# Handling missing values in the independent variables
si = SimpleImputer(missing_values = np.nan, strategy = 'mean')

columns = X.select_dtypes(include = 'float64').columns
columns

si.fit(X[columns])
X[columns] = si.transform(X[columns])
X.info()

# Handling missing values in the dependent variable
si = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
si.fit(y)

y = si.transform(y)
y = pd.DataFrame(y, columns = ['outcome'], dtype = 'int64')
y



# Plot the outcome distribution
new_dataframe = pd.concat([X, y], axis = 1)
new_dataframe.shape

new_dataframe.groupby(by = 'group').describe().round().T

fig, ax = plt.subplots()
patches, text, autotexts = ax.pie(new_dataframe['outcome'].value_counts(),
                                  labels = ['Alive', 'Dead'],
                                  explode = (0.1, 0),
                                  autopct = '%1.1f%%',
                                  textprops = {'fontsize': 14, 'weight': 'bold'}
                                  )
plt.setp(autotexts, size = 14, color = 'white', weight = 'bold')
plt.title('Outcome Distribution', fontsize = 18, weight = 'bold')
plt.show()


# Plot correlation heat map
correlation_map = new_dataframe[new_dataframe.columns].corr()
correlation_map
# mask = np.array(correlation_map)
# mask[np.tril_indices_from(mask)] = False
plt.figure(figsize = (30, 30))
# sns.heatmap(correlation_map, mask = mask, vmax = 0.7, square = True, annot = True)
sns.heatmap(correlation_map, vmax = 0.7, square = True, annot = True)
plt.title('Correlation Map', fontsize = 20, weight = 'bold')


# Getting the independent & dependent variables
X = new_dataframe.iloc[:, 2:-1].values
y = new_dataframe.iloc[:, -1].values

# Splitting the dataset into Training set & Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0, stratify = y)