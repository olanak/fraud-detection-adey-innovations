from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Usage in train.py
X_train_res, y_train_res = apply_smote(X_train, y_train)
model.fit(X_train_res, y_train_res)
