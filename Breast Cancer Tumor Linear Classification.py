# Author: Akash Das
# Copywrite: Akash Das
# Data from UC Irvine Machine Learning Repository - Diagnostic Breast Cancer Wisconsin
# Citation: Wolberg, William, Mangasarian, Olvi, Street, Nick, and Street, W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

# -- Step 1: Import Libraries --
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection as ms

# -- Step 2: Find key features with high correlation, define desired dataset features --
cancer_data = datasets.load_breast_cancer()

X = cancer_data.data
feature_names = cancer_data.feature_names
correlation_matrix = np.corrcoef(X, rowvar=False)
plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=cancer_data.feature_names, yticklabels=cancer_data.feature_names)
plt.title('Correlation Matrix of Breast Cancer Dataset Features')
plt.xlabel(cancer_data.feature_names)
plt.ylabel(cancer_data.feature_names)
#plt.show()

y = cancer_data.target # y in set [0, 1] s.t. 0 = cancerous, 1 = benign
X = preprocessing.scale(X)

# -- Step 3: Plot the inital data to see if linearly seperable --
sns.scatterplot(x = X[:, 3], y = X[:, 7], hue = y) # Here, we plot attribute 3 (Mean Area) and attribute 2 (Mean Concave Points) in the cancer_data dataset. 
                                                   # Hue represents the labels of the dataset (0 or 1).
plt.xlabel("Mean Area")
plt.ylabel("Mean Concave Points")
plt.show()

# -- Step 4: Define a set of alphas and find accuracy via Cross Validation --
alphas = np.arange(1e-15, 1, 0.005)
validation_score = np.zeros((len(alphas), 1)) # Create zero-array for rows = # alphas, 1 column

for i in range(len(alphas)):
    model = linear_model.SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = alphas[i])
    accuracy = ms.cross_val_score(model, X, y, cv = 5)
    validation_score[i] = accuracy.mean()

# -- Step 5: Plot all alphas --
#plt.plot(alphas, validation_score)
plt.xlim(0,1)
plt.ylim(0.94, 0.989999)
plt.xlabel("Alpha")
plt.ylabel("Mean Cross Validation Score")
#plt.show()

# -- Step 6: Find alpha_star --
index_max = np.argmax(validation_score)
alpha_star = alphas[index_max]

# -- Step 7: Plot alpha_star relative to other alphas --
#plt.plot(alphas, validation_score)
plt.xlim(0,1)
plt.ylim(0.94, 0.989999)
#plt.plot(np.ones(10) * alpha_star, np.arange(0.1, 1.1, 0.1), '--r')
plt.xlabel("Alpha")
plt.ylabel("Mean Cross Validation Score")
#plt.show()

# -- Step 8: Train the model using alpha_star and show accuracy --
model_alpha_star = linear_model.SGDClassifier(loss = "hinge", penalty = 'l2', alpha = alpha_star)
trained_model = model_alpha_star.fit(X, y)
print(f"Accuracy of model {trained_model.score(X, y)}")

# -- Step 9: Show orignal plot with decision boundary --
db_slope = trained_model.coef_[0,1]/-trained_model.coef_[0,0]
x_intercepts_db = np.arange(-20, 20, 0.05)
y_values_db = db_slope * x_intercepts_db
sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = y)
plt.plot(x_intercepts_db, y_values_db, '--r')
plt.xlim(-4,4)
plt.ylim(-6,6)
plt.xlabel('Mean Area')
plt.ylabel('Mean Concave Points')
plt.show()