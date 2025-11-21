import matplotlib.pyplot as plt
import numpy as np

# Örnek modeller
models = ['SVM', 'Random Forest', 'KNN', 'Decision Tree']

# Gerçek K-Fold doğruluk değerleri (placeholder)
# kfold_accuracies[model] = np.array([...])  # her fold için gerçek doğruluklar

folds = 5

plt.figure(figsize=(8,5))
for model in models:
    plt.plot(range(1, folds+1), kfold_accuracies[model]*100, marker='o', label=model)

plt.xticks(range(1, folds+1))
plt.xlabel('Fold Numarası')
plt.ylabel('Doğruluk (%)')
plt.title('K-Fold Çapraz Doğrulama Sonuçları')
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('kf_fold_accuracy_theoretical.png', dpi=300)
plt.show()
