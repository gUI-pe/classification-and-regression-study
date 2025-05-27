import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
features = ['P_sist', 'P_dist', 'qPA', 'Pulse', 'BreathFreq']

data['Class'] = data['Class'].astype('category').cat.codes

X = data[features]
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    max_depth=None,
    min_samples_split=2,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
oob = rf.oob_score_

print(f"OOB estimate: {oob:.2f}")
print(f"Acurácia: {accuracy:.2f}")