import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from dataprep import X, y, X_train, X_test, y_train, y_test, new_dataframe


# Handle over sampling of the minority class
smote = SMOTE(sampling_strategy = 'minority')

X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the resampled dataset into Training set & Test set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2,
                                                    random_state = 0, stratify = y_resampled)


# Feature Scaling
rc = RobustScaler()
X_train = rc.fit_transform(X_train)
X_test = rc.transform(X_test)

# Building the model
classifier = Sequential()

# Add the first layer
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 196, activation = 'relu'))
classifier.add(Dense(units = 196, activation = 'relu'))

classifier.add(BatchNormalization())

# Add the output layer
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)

classifier.summary()

# Predicting the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print(classification_report(
    y_test, 
    y_pred))
cm = confusion_matrix(y_test, y_pred)
cm

print("Accuracy is {}%".format(accuracy_score(y_pred = y_pred, y_true = y_test)))

# Plot the confusion_matrix
plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, fmt = '.1f', linewidths = 1.5)
plt.ylabel('Actual', fontsize = 16, weight = 'bold')
plt.xlabel('Predicted', fontsize = 16, weight = 'bold')
plt.title('Confusion Matrix', fontsize = 24, weight = 'bold')
plt.show()
