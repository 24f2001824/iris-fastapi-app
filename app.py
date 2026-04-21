from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

class_names = iris.target_names


@app.get("/")
def home():
    return {"message": "Iris Classifier API is running 🌸"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(sl: float, sw: float, pl: float, pw: float):
    prediction = model.predict([[sl, sw, pl, pw]])[0]
    return {
        "prediction": int(prediction),
        "class_name": class_names[prediction]
    }
