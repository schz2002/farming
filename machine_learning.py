import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Train a random forest classifier on the Iris dataset
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(iris.data, iris.target)

# Save the trained model to a file
joblib.dump(model, 'model.joblib')
