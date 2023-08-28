import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from mlxtend.preprocessing import minmax_scaling
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier


import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Dataset/health_and_sleep/SaYoPillow.csv')

# Renaming the columns of the DataFrame for better readability and understanding
data.columns = ['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen',
                'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']

data.drop(columns=['limb_movement'], axis=1, inplace=True)

# print(data.head())

# Correlation Analysis: Correlation matrix heatmap
plt.figure(figsize=(14, 12))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.savefig('Experiment_Code/sleep_pattern/correlation.png')
plt.clf()


# Splitting dataset
# Split the data into features (X) and the target variable (y)
X = data.drop(['stress_level'], axis=1)
y = data['stress_level']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Identifying important features
# Create an instance of the RandomForestClassifier with hyperparameters
forest = RandomForestClassifier(n_estimators=500, random_state=1)

# Train the RandomForestClassifier on the training data
forest.fit(X_train, y_train.values.ravel())

# Get the feature importance from the trained RandomForestClassifier
importance = forest.feature_importances_

# Loop over each feature and its importance
for i in range(X_train.shape[1]):
    # Print the feature number, name, and importance score
    print("%2d) %-*s %f" % (i + 1, 30, data.columns[i], importance[i]))

# Plotting the feature importances as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(X_train.shape[1]), importance, align='center')
plt.title('Feature Importance')
plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('Experiment_Code/sleep_pattern/feature_importance.png')
plt.clf()


# Feature-wise plotting

# # sleeping hours and stress
# fig, ax = plt.subplots(figsize=(7,7))
# data.groupby(data["stress_level"])["sleeping_hours"].mean().plot(kind='bar', rot=0, color='#84a3cf')
# plt.title("Stress Levels Measured by Sleeping Hours")
# plt.xlabel("Stress Levels")
# plt.ylabel("Number of Hours Slept")
# plt.savefig('Experiment_Code/sleep_pattern/sleeping_hrs.png')

# # heart rate and stress
# fig, ax = plt.subplots(figsize=(7,7))
# data.groupby(data["stress_level"])["heart_rate"].mean().plot(kind='bar', rot=0, color='#c789a4')
# plt.title("Stress Levels Measured by Heart Rate")
# plt.xlabel("Stress Levels")
# plt.ylabel("Heart Rate")
# plt.savefig('Experiment_Code/sleep_pattern/heart_rate.png')

# # snoring rate and stress
# fig, ax = plt.subplots(figsize=(7,7))
# data.groupby(data["stress_level"])["snoring_rate"].mean().plot(kind='bar', rot=0, color='#c789a4')
# plt.title("Stress Levels Measured by Snoring Rate")
# plt.xlabel("Stress Levels")
# plt.ylabel("Snoring Rate")
# plt.savefig('Experiment_Code/sleep_pattern/snoring_rate.png')


# Data preprocessing and ML

# data-preprocessing
X = data.drop('stress_level', axis=1)
y = pd.DataFrame(data['stress_level'])
X_scaled = minmax_scaling(X, columns=X.columns)

mi = pd.DataFrame(mutual_info_regression(X_scaled, y), columns=[
                  'MI Scores'], index=X_scaled.columns)
corr = pd.DataFrame(X_scaled[X_scaled.columns].corrwith(
    y['stress_level']), columns=['Correlation'])
s_corr = pd.DataFrame(X_scaled[X_scaled.columns].corrwith(y['stress_level'], method='spearman'),
                      columns=['Spearman_Correlation'])

relation = mi.join(corr)
relation = relation.join(s_corr)
relation.sort_values(by='MI Scores', ascending=False)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.8, test_size=0.2, random_state=42,
                                                    stratify=y, shuffle=True)

dtc = DecisionTreeClassifier()
lr = LogisticRegression()
gnb = GaussianNB()
lsvc = LinearSVC()
svc = SVC()
rfc = RandomForestClassifier()
knn = KNeighborsClassifier()
sgdc = SGDClassifier()
gbc = GradientBoostingClassifier()

models = [dtc, lr, gnb, lsvc, svc, rfc,  knn, sgdc, gbc]
model_name = ['Decision Tree', 'Logistic Regression', 'Gaussian Naive Bayes', 'Linear SVC', 'SVC', 'Random Forest',
              'KNN or k-Nearest Neighbors', 'Stochastic Gradient Descent', 'Gradient Boosting']

acc_scores = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_model = round(accuracy_score(y_pred, y_test) * 100, 2)
    acc_scores.append(acc_model)

models_acc = pd.DataFrame(
    {'Model name': model_name, 'Accuracy scores': acc_scores})
models_acc.sort_values(by='Accuracy scores', ascending=False)

print(models_acc)


# Random forest classifier
print(rfc.score(X_test, y_test))
y_predict = rfc.predict(X_test)
matrix = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(matrix)
report = classification_report(y_test, y_predict)
print("Classification Report:")
print(report)


# Random forest classifier
print(svc.score(X_test, y_test))
y_predict = svc.predict(X_test)
matrix = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(matrix)
report = classification_report(y_test, y_predict)
print("Classification Report:")
print(report)

# regression plot between Stress Level and rest of the columns
plt.figure(figsize=(15, 10))
for i, column in enumerate(data.columns, 1):
    plt.subplot(3, 3, i)
    sns.regplot(x="stress_level", y=data[column], data=data, scatter_kws={
                "color": "green"}, line_kws={"color": "blue"})
plt.show()

plt.savefig('Experiment_Code/sleep_pattern/all_regression_plot.png')
