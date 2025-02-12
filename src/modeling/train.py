import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import mlflow

def prepare_data(dataset_path, target_col):
    # Load processed data
    df = pd.read_csv(dataset_path)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data (stratified due to class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale features (if needed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Example for creditcard.csv
X_train, X_test, y_train, y_test = prepare_data(
    "data/processed/cleaned_creditcard.csv", "Class"
)
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
    }
    return metrics

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(),
}

# Train and evaluate
results = {}
for name, model in models.items():
    metrics = train_evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = metrics
    print(f"{name}: {metrics}")

mlflow.set_experiment("Fraud_Detection_CreditCard")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        metrics = train_evaluate_model(model, X_train, X_test, y_train, y_test)
        # Log parameters and metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        # Log model
        mlflow.sklearn.log_model(model, name)