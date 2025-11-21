import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Model isimleri ve sınıf etiketleri
models = ["SVM", "Random Forest", "KNN", "Decision Tree"]
labels = ["Happy", "Sad", "Angry", "Neutral"]

# Simüle edilmiş Confusion Matrix oluşturma
conf_matrices = {}
for model in models:
    # 20-30 civarında rastgele değerler (4x4) ile matris oluşturuyoruz
    mat = np.random.randint(18, 28, size=(4,4))
    conf_matrices[model] = mat

# Confusion Matrix görselleştirme
for model, matrix in conf_matrices.items():
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model} Confusion Matrix")
    plt.show()

# Simüle edilmiş ROC eğrileri
for model in models:
    fpr = np.linspace(0,1,6)
    tpr = np.sort(np.random.rand(6))  # 0-1 arasında artan TPR değerleri
    plt.plot(fpr, tpr, marker='o', label=model)

plt.plot([0,1], [0,1], 'k--')  # Rastgele ROC referans çizgisi
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Eğrileri (Simüle Edilmiş)")
plt.legend()
plt.show()
