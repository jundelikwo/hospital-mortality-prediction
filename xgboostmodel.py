import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import cross_val_score, train_test_split
from dataprep import X, y, X_train, X_test, y_train, y_test, new_dataframe
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, plot_tree, plot_importance

# Train the model
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the test set
y_pred = classifier.predict(X_test)

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

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Plot ROC curve
plot_roc_curve(classifier, X_test, y_test)
plt.plot([0, 1], [0,1], color = 'red', ls = '-')
plt.title('ROC curve')

# Plot precision recall curve
plot_precision_recall_curve(classifier, X_test, y_test)
plt.title('Precision recall curve')

# Plot the model tree
plt.figure(figsize = (10, 10))
plot_tree(classifier)



# REFINING THE MODEL

# Handle over sampling of the minority class
smote = SMOTE(sampling_strategy = 'minority')

X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting the resampled dataset into Training set & Test set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2,
                                                    random_state = 0, stratify = y_resampled)

# Train the model on the resampled dataset
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the test set
y_pred = classifier.predict(X_test)

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


# Plot ROC curve
plot_roc_curve(classifier, X_test, y_test)
plt.plot([0, 1], [0,1], color = 'red', ls = '-')
plt.title('ROC curve')

# Plot precision recall curve
plot_precision_recall_curve(classifier, X_test, y_test)
plt.title('Precision recall curve')

# Plot the model tree
plt.figure(figsize = (30, 30))
plot_tree(classifier)

importantFeatures = classifier.feature_importances_
importantFeatures

fi = pd.Series(importantFeatures, index = new_dataframe.iloc[:, 2:-1].columns)
fi.sort_values(ascending = False)

plt.figure(figsize = (30, 30))
fi.nlargest(n = 20).plot(kind = 'barh')
