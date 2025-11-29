import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from matplotlib.colors import ListedColormap

#Loading dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#Check the balance of classes in the target variable (Purchased)
print("\nClass distribution:")
print(dataset['Purchased'].value_counts())

# PREPARING VARIABLES
# Predictor features : Age & Estimated Salary
X = dataset[['Age', 'EstimatedSalary']].values

# Target : Purchased (0/1)
y = dataset['Purchased'].values

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
#random_state=0 ensures that the division will be the same each time

#Feature Scaling
#Standardizing the features 
#(the difference between fit_transform and transform is that fit_transform computes the parameters on the training set and applies them, while transform uses the already computed parameters)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Logistic Regression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("\nPredictions :", y_pred)
print("Accuracy :", accuracy_score(y_test, y_pred))

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix :\n", cm)

#ROC and AUC curves 
# Probabilities for class 1
y_prob = classifier.predict_proba(X_test)[:, 1]

#ROC Computation
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("\nAUC :", roc_auc)

# PLotting ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

#Train set/test
def plot_decision_boundary(X_set, y_set, title):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )
    plt.figure(figsize=(8, 6))
    plt.contourf(
        X1, X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(('#c0f6fb', '#c7fbc0'))
    )
    plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set,
                cmap=ListedColormap(('purple', 'pink')))
    plt.title(title)
    plt.xlabel('Age (scaled)')
    plt.ylabel('Estimated Salary (scaled)')
    plt.show()
plot_decision_boundary(X_train, y_train, "Logistic Regression (Training set)")
plot_decision_boundary(X_test, y_test, "Logistic Regression (Test set)")

#Model coefficients (writing the equation)
b0 = classifier.intercept_[0]
b1, b2 = classifier.coef_[0]

print("\n--- Equation of Logistic Regression ---")
print(f"Logit(p) = {b0:.4f} + {b1:.4f}*Age + {b2:.4f}*EstimatedSalary") #printing logistic regression equation

#Cross Validation ( underfitting/overfitting detection)
scores = cross_val_score(classifier, X_train, y_train, cv=10)
print("\nCross-validation accuracy scores :", scores)
print("Mean accuracy :", scores.mean())
print("Standard deviation :", scores.std())
