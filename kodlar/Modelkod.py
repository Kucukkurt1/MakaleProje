# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import librosa
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# 1. Veri Yükleme ve Hazırlık
# -------------------------------


data_dir = "dataset/"  # RAVDESS 
emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']  # örnek sınıflar

def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=None)
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    # Chroma
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    
    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)
    
    # Tüm özellikleri birleştir
    features = np.hstack([mfccs_mean, chroma_mean, zcr_mean])
    return features


X = []
y = []

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            
            # Dosya ismine göre sınıf etiketi
            for idx, emotion in enumerate(emotions):
                if emotion in file:
                    label = emotion
                    break
            else:
                label = 'unknown'
            
            X.append(extract_features(file_path))
            y.append(label)

X = np.array(X)
y = np.array(y)

# -------------------------------
# 2. Veri Ön İşleme
# -------------------------------


le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Özellikleri standardize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. Modeller ve RFE ile Özellik Seçimi
# -------------------------------

models = {
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(random_state=42)
}

# RFE ile özellik seçimi (SVM tabanlı)
svm_rfe = SVC(kernel='linear', random_state=42)
rfe_selector = RFE(estimator=svm_rfe, n_features_to_select=30, step=1)  # en iyi 30 özellik
X_rfe = rfe_selector.fit_transform(X_scaled, y_encoded)

# -------------------------------
# 4. K-Fold Çapraz Doğrulama
# -------------------------------

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for model_name, model in models.items():
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for train_index, test_index in kf.split(X_rfe, y_encoded):
        X_train, X_test = X_rfe[train_index], X_rfe[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy_list.append(accuracy_score(y_test, y_pred))
        precision_list.append(precision_score(y_test, y_pred, average='weighted'))
        recall_list.append(recall_score(y_test, y_pred, average='weighted'))
        f1_list.append(f1_score(y_test, y_pred, average='weighted'))
    
    results[model_name] = {
        'Accuracy': (np.mean(accuracy_list), np.std(accuracy_list)),
        'Precision': (np.mean(precision_list), np.std(precision_list)),
        'Recall': (np.mean(recall_list), np.std(recall_list)),
        'F1-Score': (np.mean(f1_list), np.std(f1_list))
    }

# -------------------------------
# 5. Sonuçların Yazdırılması
# -------------------------------

for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    for metric_name, (mean_val, std_val) in metrics.items():
        print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
