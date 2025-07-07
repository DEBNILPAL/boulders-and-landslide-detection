# --- File: src/8_model_training.py ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_landslide_classifier(features, labels, model_path=None):
    """
    Train a random forest classifier and optionally save the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    if model_path:
        joblib.dump(clf, model_path)
        print(f"Model saved to: {model_path}")

    return clf

def load_trained_model(model_path):
    """
    Load a pre-trained model from a file.
    """
    return joblib.load(model_path)