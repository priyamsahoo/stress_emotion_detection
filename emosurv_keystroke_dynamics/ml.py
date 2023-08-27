import pandas as pd
import numpy as np              
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier    # For Random Forest model
from sklearn.tree import DecisionTreeClassifier        # For Decision Tree model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from mlxtend.preprocessing import minmax_scaling
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn import metrics, model_selection

import shap


from utils import compute_metrics, plot_confusion_matrix


import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Dataset/emosurv/Fixed_Text_Typing_Dataset_mod.csv')
# print(data.head())
# print(data.info())

# # shape of the dataset
# print(data.shape)

# # To show statistical summary of the columns of our data
# print(data.describe(include="all"))

# # checking for null values in the dataframe
# print(data.isnull().sum())

# # display number of samples on each class
# print(data['emotionIndex'].value_counts())

data = data.drop(columns=data.columns[0], axis=1)
data = data.drop('sentence', axis=1)
data = data.drop('leftFreq', axis=1)

# shift column 'emotionIndex' to the last position
last_column = data.pop('emotionIndex')
data.insert(len(data.columns), 'emotionIndex', last_column)

print(data.info())

# convert categorical features into numerical features
data.emotionIndex = data.emotionIndex.map(
    {'N': 0, 'H': 1, 'C': 2, 'S': 3, 'A': 4})
data.gender = data.gender.map({'Female': 1, 'Male': 2})
data.ageRange = data.ageRange.map(
    {'16-19': 1, '20-29': 2, '30-39': 3, '>=50': 4})
data.typeWith = data.typeWith.map({'1 hand': 1, '2 hands': 2})

# # Correlation Analysis: Correlation matrix heatmap
# plt.figure(figsize=(10, 8))
# correlation_matrix = data.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
# plt.title("Correlation Matrix Heatmap")
# plt.savefig('Experiment_Code/emosurv_keystroke_dynamics/correlation.png')

# Split the data into features (X) and the target variable (y)
X = data.drop('emotionIndex', axis=1)
y = data['emotionIndex']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# # Display the shapes of the training and testing sets
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)

# identifying important features
# Create an instance of the RandomForestClassifier with hyperparameters
forest = RandomForestClassifier(n_estimators=500, random_state=1)

# Train the RandomForestClassifier on the training data
forest.fit(X_train, y_train.values.ravel())

# Get the feature importances from the trained RandomForestClassifier
importances = forest.feature_importances_

# Loop over each feature and its importance
for i in range(X_train.shape[1]):
    # Print the feature number, name, and importance score
    print("%2d) %-*s %f" % (i + 1, 30, X_train.columns[i], importances[i]))

# # Plotting the feature importances as a bar chart
# plt.figure(figsize=(10, 6))
# plt.bar(range(X_train.shape[1]), importances, align='center')
# plt.title('Feature Importance')
# plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
# plt.xlabel('Features')
# plt.ylabel('Importance Score')
# plt.tight_layout()
# plt.savefig('Experiment_Code/emosurv_keystroke_dynamics/feature_importance_2.png')

# Data processing and ML

# define classification models
classifiers_name = ['LogReg', 'RF', 'XGB', 'SVM', 'MLP']

classifiers = [
    LogisticRegression(multi_class='auto', max_iter=500, solver='newton-cg',
                       class_weight={0: 0.1, 1: 1, 2: 1, 3: 1, 4: 1}),
    RandomForestClassifier(n_estimators=200, max_depth=5, class_weight={
                           0: 0.1, 1: 1, 2: 1, 3: 1, 4: 1}),
    xgb.XGBClassifier(objective='multi:softmax',
                      eval_metric='mlogloss', use_label_encoder=False),
    # , class_weight={0:0.1,1:1,2:1,3:1,4:1}),
    SVC(kernel='rbf', decision_function_shape='ovr', probability=True),
    MLPClassifier(alpha=1, max_iter=500)]

best_model = 'XGB'

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = classifiers[classifiers_name.index(best_model)]
model.fit(X_train, y_train)

performance_metrics = compute_metrics(model, X_test, y_test, show=True)
plot_confusion_matrix(performance_metrics["Confusion Matrix"])

# SHAP value

# columns = X.columns
# print(columns)
# DF, based on which importance is checked
X_importance = pd.DataFrame(X_test, columns=X.columns)

# Explain model predictions using shap library:
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_importance)

# Plot summary_plot
plt.clf()
shap.summary_plot(shap_values, X_importance, plot_size="auto", show=False)
plt.savefig('Experiment_Code/emosurv_keystroke_dynamics/shap_summary.png')
