import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics import mean_absolute_error, r2_score

# 1) Veri Yükleme
data = Dataset.load_builtin('ml-100k')
# Suprise kütüphanesine gömülü olan ama ayrıca internette bulunan grup lensin data seti
# Örnek şartını sağlıyor.
# Hazır MovieLens 100K


# 2) Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 3) Model oluştur ve eğit
svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
svd_model.fit(trainset)

# 4) Test seti tahmini
predictions = svd_model.test(testset)

# 5) Performans Metrikleri
# Surprise MAE
mae_surprise = accuracy.mae(predictions, verbose=True)

# y_true, y_pred listelerini ayır
y_true = [pred.r_ui for pred in predictions]
y_pred = [pred.est for pred in predictions]

# sklearn MAE
mae_sklearn = mean_absolute_error(y_true, y_pred)

# R^2
r2 = r2_score(y_true, y_pred)

print(f"MAE : {mae_surprise:.4f}")
print(f"R^2: {r2:.4f}")
