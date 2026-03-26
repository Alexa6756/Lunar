import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TERRAIN_FEATURES = ["elevation", "slope", "roughness", "curvature", "tpi", "tri"]
THERMAL_FEATURES = ["temp_day", "temp_night", "temp_variation", "temp_gradient"]
FEATURES         = TERRAIN_FEATURES + THERMAL_FEATURES
LABEL_NAMES      = ["Safe", "Moderate", "Danger"]
CMAP             = ListedColormap(["#2ECC71", "#F1C40F", "#E74C3C"])


class DynamicHazardModel:

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self.model         = None
        self.scaler        = None
        self.feature_names = FEATURES
        self._trained      = False

    def train(self, dataset_path: str = "dataset/dataset.csv") -> dict:

        data = pd.read_csv(dataset_path)

        missing_thermal = [f for f in THERMAL_FEATURES if f not in data.columns]
        if missing_thermal:
            raise ValueError(
                f"Dataset is missing thermal columns: {missing_thermal}. "
                "Re-run dataset.py to include thermal features."
            )

        X = data[FEATURES]
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_s   = self.scaler.fit_transform(X_train)
        X_test_s    = self.scaler.transform(X_test)

        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train_s, y_train)
        self._trained = True

        train_pred = self.model.predict(X_train_s)
        test_pred  = self.model.predict(X_test_s)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc  = accuracy_score(y_test,  test_pred)

        print(f"Train accuracy : {train_acc:.4f}")
        print(f"Test  accuracy : {test_acc:.4f}")
        print("\nClassification Report")
        print(classification_report(y_test, test_pred, target_names=LABEL_NAMES))

        print("\nFeature Importances")
        importances = sorted(
            zip(FEATURES, self.model.feature_importances_),
            key=lambda x: -x[1],
        )
        for name, val in importances:
            print(f"  {name:<16}: {val:.4f}")

        return {
            "train_acc":   train_acc,
            "test_acc":    test_acc,
            "importances": dict(importances),
            "cm":          confusion_matrix(y_test, test_pred).tolist(),
            "report":      classification_report(
                               y_test, test_pred,
                               target_names=LABEL_NAMES,
                               output_dict=True,
                           ),
        }


    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        return self.model.predict_proba(self.scaler.transform(X))



    def save(self, output_dir: str = "outputs") -> None:
        self._check_trained()
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model,  os.path.join(output_dir, "dynamic_model.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "dynamic_scaler.pkl"))
        

    @classmethod
    def load(cls, output_dir: str = "outputs") -> "DynamicHazardModel":
        instance = cls()
        instance.model  = joblib.load(os.path.join(output_dir, "dynamic_model.pkl"))
        instance.scaler = joblib.load(os.path.join(output_dir, "dynamic_scaler.pkl"))
        instance._trained = True
        return instance

    @staticmethod
    def load_thermal(
        temp_max_path: str,
        temp_min_path: str,
        target_rows:   int,
        target_cols:   int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        with rasterio.open(temp_max_path) as src:
            day = src.read(1).astype(np.float32)
        with rasterio.open(temp_min_path) as src:
            night = src.read(1).astype(np.float32)

        scale_y = target_rows / day.shape[0]
        scale_x = target_cols / day.shape[1]
        day   = zoom(day,   (scale_y, scale_x), order=1)
        night = zoom(night, (scale_y, scale_x), order=1)

        temp_variation = day - night
        dTy, dTx       = np.gradient(day)
        temp_gradient  = np.sqrt(dTx**2 + dTy**2)

        return day, night, temp_variation, temp_gradient


    def generate_risk_map(
        self,
        dem_path:      str = "data/dem.tif",
        temp_max_path: str = "data/temp_max.tif",
        temp_min_path: str = "data/temp_min.tif",
        output_path:   str = "outputs/dynamic_risk_map.npy",
        chunk_size:    int = 500_000,
    ) -> np.ndarray:

        self._check_trained()

        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from preprocessing.feature_extraction import extract_features

        feats      = extract_features(dem_path)
        dem        = feats["dem"]
        rows, cols = dem.shape

        day, night, temp_variation, temp_gradient = self.load_thermal(
            temp_max_path, temp_min_path, rows, cols
        )

        elevation      = dem.flatten()
        slope          = feats["slope"].flatten()
        roughness      = feats["roughness"].flatten()
        curvature      = feats["curvature"].flatten()
        tpi            = feats["tpi"].flatten()
        tri            = feats["tri"].flatten()
        temp_day_f     = day.flatten()
        temp_night_f   = night.flatten()
        temp_var_f     = temp_variation.flatten()
        temp_grad_f    = temp_gradient.flatten()

        total     = rows * cols
        risk_flat = np.zeros(total, dtype=np.uint8)

        
        for start in range(0, total, chunk_size):
            end     = min(start + chunk_size, total)
            X_chunk = pd.DataFrame({
                "elevation":      elevation[start:end],
                "slope":          slope[start:end],
                "roughness":      roughness[start:end],
                "curvature":      curvature[start:end],
                "tpi":            tpi[start:end],
                "tri":            tri[start:end],
                "temp_day":       temp_day_f[start:end],
                "temp_night":     temp_night_f[start:end],
                "temp_variation": temp_var_f[start:end],
                "temp_gradient":  temp_grad_f[start:end],
            })
            risk_flat[start:end] = self.predict(X_chunk)
            

        risk_map = risk_flat.reshape(rows, cols)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        np.save(output_path, risk_map)
        
        return risk_map



    def plot_confusion_matrix(
        self,
        save_path: str = "outputs/confusion_matrix_dynamic.png",
    ) -> None:
        data = pd.read_csv("dataset/dataset.csv")
        X    = data[FEATURES]
        y    = data["label"]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        preds = self.predict(X_test)
        cm    = confusion_matrix(y_test, preds)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Dynamic Model — Confusion Matrix")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    

    def plot_feature_importance(
        self,
        save_path: str = "outputs/feature_importance_dynamic.png",
    ) -> None:
        self._check_trained()
        importances = sorted(
            zip(FEATURES, self.model.feature_importances_),
            key=lambda x: x[1],
        )
        names, vals = zip(*importances)
        plt.figure(figsize=(8, 5))
        plt.barh(names, vals, color="#E67E22")
        plt.xlabel("Importance")
        plt.title("Dynamic Model — Feature Importances")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_risk_map(
        self,
        risk_map:   np.ndarray,
        save_path:  str = "outputs/dynamic_risk_map.png",
        downsample: int = 20,
    ) -> None:
        plt.figure(figsize=(12, 7))
        plt.imshow(risk_map[::downsample, ::downsample], cmap=CMAP, vmin=0, vmax=2)
        cbar = plt.colorbar(ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(LABEL_NAMES)
        plt.title("Dynamic Lunar Hazard Map (terrain + thermal)")
        plt.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
       

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError(
                "Model is not trained"
             )
if __name__ == "__main__":
    model = DynamicHazardModel()
    model.train("dataset/dataset.csv")
    model.save("outputs")
    model.plot_confusion_matrix()
    model.plot_feature_importance()
    risk_map = model.generate_risk_map(
    "data/dem.tif", "data/temp_max.tif", "data/temp_min.tif"
    )
    model.plot_risk_map(risk_map)
   
