import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap


data = pd.read_csv("outputs/dataset.csv")

print("Dataset size:", data.shape)


features = [
    "elevation",
    "slope",
    "roughness",
    "curvature",
    "tpi",
    "tri"
]

X = data[features]
y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\nSTATIC MODEL PERFORMANCE:")


print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)


print("\nConfusion Matrix")

cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Safe","Moderate","Danger"],
            yticklabels=["Safe","Moderate","Danger"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png")
plt.show()


print("\nFeature Importance")

for name,val in zip(features,model.feature_importances_):
    print(f"{name}: {val:.4f}")


joblib.dump(model,"outputs/static_model.pkl")
joblib.dump(scaler,"outputs/static_scaler.pkl")


from preprocessing.feature_extraction import extract_features

features_full = extract_features("data/dem.tif")

dem = features_full["dem"]

rows, cols = dem.shape

print("DEM size:", rows, cols)

elevation = dem.flatten()
slope = features_full["slope"].flatten()
roughness = features_full["roughness"].flatten()
curvature = features_full["curvature"].flatten()
tpi = features_full["tpi"].flatten()
tri = features_full["tri"].flatten()

total_pixels = rows * cols

print("Total pixels:", total_pixels)

risk_map = np.zeros(total_pixels, dtype=np.uint8)

chunk = 500000  


for start in range(0, total_pixels, chunk):

    end = min(start + chunk, total_pixels)

    X_chunk = pd.DataFrame({
        "elevation": elevation[start:end],
        "slope": slope[start:end],
        "roughness": roughness[start:end],
        "curvature": curvature[start:end],
        "tpi": tpi[start:end],
        "tri": tri[start:end]
    })

    X_chunk = scaler.transform(X_chunk)

    preds = model.predict(X_chunk)

    risk_map[start:end] = preds

    print(f"Processed {end}/{total_pixels}")

risk_map = risk_map.reshape(rows, cols)


np.save("outputs/static_risk_map.npy", risk_map)


cmap = ListedColormap(["green","yellow","red"])

plt.figure(figsize=(10,6))
plt.imshow(risk_map, cmap=cmap)
plt.title("Static Lunar Risk Map")
plt.colorbar()
plt.savefig("outputs/static_risk_map.png")
