import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# TODO 1: Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TODO 2: Elegir y entrenar un algoritmo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# TODO 3: Evaluar el modelo
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# TODO 4: Guardar el modelo
joblib.dump(model, r'C:\Users\PC\Desktop\P3\PRACTICA\model.pkl')
print("¡Modelo guardado correctamente!")