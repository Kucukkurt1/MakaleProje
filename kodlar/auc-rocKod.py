import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Masaüstü yolu
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
os.makedirs(desktop_path, exist_ok=True)

# Modeller ve sınıflar
models = ['SVM', 'Random Forest', 'KNN', 'Decision Tree']
classes = ['Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Surprise', 'Disgust']
n_classes = len(classes)

# y_true ve y_score sözlükleri model adıyla eşleşmeli
# y_true[model]: (num_samples,) gerçek etiketler (0..6)
# y_score[model]: (num_samples, n_classes) tahmin olasılıkları
# Örnek:
# y_true = {'SVM': np.array([...]), ...}
# y_score = {'SVM': np.array([...]), ...}

for model in models:
    y_true_model = y_true[model]
    y_score_model = y_score[model]
    
    # One-hot encode
    y_true_bin = label_binarize(y_true_model, classes=range(n_classes))
    
    plt.figure(figsize=(8,6))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_model[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model} ROC Eğrileri')
    plt.legend(loc="lower right")
    
    # PNG olarak masaüstüne kaydet
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, f'roc_{model.lower().replace(" ", "_")}.png'), dpi=300)
    plt.close()
