import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
df = pd.read_csv("meteorite-landings.csv")
df1 = df.copy()
df1.head()
df1.describe().T
df1.isnull().values.any()
df1.dropna( inplace= True)
df1.isnull().values.any()

data = pd.get_dummies(df1, columns=['recclass'])

data.head()
label_encoder = LabelEncoder()
data['fall_encoded'] = label_encoder.fit_transform(df1['fall'])

data.head()

# Gerekli sütunları seç
X = data.drop(['name', 'nametype' , 'fall', 'fall_encoded' , 'GeoLocation'  ], axis=1)  # Hedef değişken hariç diğer sütunları alın
y = data['fall_encoded']  # Tahmin edilecek sütun (fall) olarak ayarlayın
X.head()

# Veri setini eğitim ve test olarak bölme
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Destek Vektör Makinesi (SVM) sınıflandırıcısı
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_preds)
svm_precision = precision_score(y_test, svm_preds)
svm_recall = recall_score(y_test, svm_preds)
svm_f1_score = f1_score(y_test, svm_preds)
svm_probs = svm_model.predict_proba(X_test)[:, 1]
svm_roc_auc = roc_auc_score(y_test, svm_probs)
svm_confusion_matrix = confusion_matrix(y_test, svm_preds)

# K-En Yakın Komşu (KNN) sınıflandırıcısı
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_preds)
knn_precision = precision_score(y_test, knn_preds)
knn_recall = recall_score(y_test, knn_preds)
knn_f1_score = f1_score(y_test, knn_preds)
knn_probs = knn_model.predict_proba(X_test)[:, 1]
knn_roc_auc = roc_auc_score(y_test, knn_probs)
knn_confusion_matrix = confusion_matrix(y_test, knn_preds)

# Karar Ağacı sınıflandırıcısı
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_probs = dt_model.predict_proba(X_test)[:, 1]
dt_accuracy = accuracy_score(y_test, dt_preds)
dt_precision = precision_score(y_test, dt_preds)
dt_recall = recall_score(y_test, dt_preds)
dt_f1_score = f1_score(y_test, dt_preds)
dt_roc_auc = roc_auc_score(y_test, dt_probs)
dt_confusion_matrix = confusion_matrix(y_test, dt_preds)

# Performans metriklerini DataFrame'e kaydet
performance_data = pd.DataFrame({
    'Model': ['SVM', 'KNN', 'Decision Tree'],
    'Accuracy': [svm_accuracy, knn_accuracy, dt_accuracy],
    'Precision': [svm_precision, knn_precision, dt_precision],
    'Recall': [svm_recall, knn_recall, dt_recall],
    'F1 Score': [svm_f1_score, knn_f1_score, dt_f1_score],
    'ROC AUC Score': [svm_roc_auc, knn_roc_auc, dt_roc_auc]
})

# Grafikleri çizme
plt.figure(figsize=(5, 5))

# Accuracy
plt.subplot(1, 1, 1)
ax = sns.barplot(x='Model', y='Accuracy', data=performance_data)
plt.title('Accuracy')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.show()
plt.figure(figsize=(5, 5))

# Precision
plt.subplot(1, 1, 1)
ax = sns.barplot(x='Model', y='Precision', data=performance_data)
plt.title('Precision')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.show()
plt.figure(figsize=(5, 5))

# Recall
plt.subplot(1, 1, 1)
ax = sns.barplot(x='Model', y='Recall', data=performance_data)
plt.title('Recall')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.show()
plt.figure(figsize=(5, 5))

# F1 Score
plt.subplot(1, 1, 1)
ax = sns.barplot(x='Model', y='F1 Score', data=performance_data)
plt.title('F1 Score')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.show()
plt.figure(figsize=(5, 5))

# ROC AUC Score
plt.subplot(1, 1, 1)
ax = sns.barplot(x='Model', y='ROC AUC Score', data=performance_data)
plt.title('ROC AUC Score')
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
plt.show()
plt.figure(figsize=(5, 5))

# ROC Curve
plt.subplot(1, 1, 1)
fpr, tpr, _ = roc_curve(y_test, svm_probs)
plt.plot(fpr, tpr, label='SVM')
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
plt.plot(knn_fpr, knn_tpr, label='KNN')
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
plt.plot(dt_fpr, dt_tpr, label='Decision Tree')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.tight_layout()
plt.show()

# SVM Karmaşıklık Matrisi
plt.figure(figsize=(6, 4))
sns.heatmap(svm_confusion_matrix, annot=True, fmt=".0f", cmap='Blues')
plt.title("SVM Karmaşıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# KNN Karmaşıklık Matrisi
plt.figure(figsize=(6, 4))
sns.heatmap(knn_confusion_matrix, annot=True, fmt=".0f", cmap='Blues')
plt.title("KNN Karmaşıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# Decision Tree Karmaşıklık Matrisi
plt.figure(figsize=(6, 4))
sns.heatmap(dt_confusion_matrix, annot=True, fmt=".0f", cmap='Blues')
plt.title("Decision Tree Karmaşıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()