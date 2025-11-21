import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Modellerinizi ve test setinizi tanımlayın
# Örnek: modeller = {'SVM': svm_model, 'Random Forest': rf_model, ...}
# X_test, y_test test verileri

models = ['SVM', 'Random Forest', 'KNN', 'Decision Tree']
accuracy = []

# Her model için Accuracy hesapla
for model in models:
    y_pred = model.predict(X_test)  # Gerçek model tahmini
    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc * 100)  # % cinsinden

# Çubuk grafik çizimi
plt.figure(figsize=(8,5))
bars = plt.bar(models, accuracy, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

# Yükseklikleri göster
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}%', ha='center', va='bottom')

plt.ylabel('Doğruluk (%)')
plt.title('Modellerin Kilitli Nihai Test Seti Üzerindeki Doğruluk Performansı')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# PNG olarak kaydet
plt.tight_layout()
plt.savefig('accuracy_bar.png', dpi=300)
plt.show()
