import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


data = 'data_banknote_authentication.txt'
df = pd.read_csv(data, header=None)

# Assigning column names
df.columns = ['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Class']

# Visualize feature relationships (pairplot)
sns.pairplot(df, hue='Class')
plt.suptitle("Feature Relationships by Class", y=1.02)
plt.show()



# Splitting the data
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We may try and visualize different decision tree structures by changing 'max_depth' and 'min_samples_split'
model = DecisionTreeClassifier(max_depth=4, min_samples_split=10, criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
disp.ax_.set_title("Confusion Matrix")
plt.show()

# Visualizing the trained decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['Fake', 'Authentic'])
plt.title("Decision Tree")
plt.show()

# Deeper trees are harder to interpret. Depth of a value '4' is readable.


# Feature importance
importances = model.feature_importances_
feature_names = X.columns

# Plotting feature importances
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()




