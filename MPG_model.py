#ライブラリのインポート
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold,GridSearchCV
from sklearn.inspection import permutation_importance
import pickle

#データセットの準備
train_df = pd.read_csv(r'C:\Users\Junji Akiyama\MPG_Project\Dataset\MPG_train_data.csv')
test_df = pd.read_csv(r'C:\Users\Junji Akiyama\MPG_Project\Dataset\MPG_test_data.csv')

#入力変数を定義 ※モデル作成に不要なカラムを削除
train_df = train_df.drop(['id','car name','acceleration','origin','horsepower'],axis=1)
test_x = test_df.drop(['id','car name','acceleration','origin','horsepower'],axis=1)

#目的変数[mpg]と[displacement]の外れ値を削除
train_df = train_df.drop(train_df[(train_df['mpg'].between(37,40))&(train_df['displacement'].between(250,280))].index)

#説明変数と出力変数に分割
train_x = train_df.drop(['mpg'],axis=1)
train_t = train_df['mpg']

#入力変数の標準化
scaler = StandardScaler()
scaled_train_x = scaler.fit_transform(train_x)
scaled_test_x = scaler.fit_transform(test_x)

#学習
model = SVR(
    gamma='scale',
    C=22,
    epsilon=1.6,
    kernel='rbf'
    )
model.fit(scaled_train_x,train_t)
y_pred = model.predict(scaled_train_x)
# RMSEスコアを計算するためのカスタムスコアリング関数
def rmse(train_t, y_pred):
    return np.sqrt(mean_squared_error(train_t, y_pred))
# RMSE用のスコアラーを作成
rmse_scorer = make_scorer(rmse, greater_is_better=False)
# KFoldクロスバリデーションの設定
kf = KFold(n_splits=5, shuffle=True, random_state=0)
# クロスバリデーションを実施しRMSEを計算 ※best_model
best_model_rmse_scores = cross_val_score(model, scaled_train_x, train_t, cv=kf, scoring=rmse_scorer)
# クロスバリデーションのRMSEの平均と標準偏差を出力
print(f"平均RMSE: {-np.mean(best_model_rmse_scores):.4f}")
print(f"RMSEの標準偏差: {np.std(best_model_rmse_scores):.4f}")

#各特徴量ごとの重要度を確認 ※学習データに対して
# パーミュテーション重要度を計算
results = permutation_importance(model, scaled_train_x, train_t, scoring=rmse_scorer)

# 特徴量の重要度を取得
importances = results.importances_mean
std = results.importances_std
# 特徴量の名前を取得（必要に応じて）
feature_names = train_x.columns
# 特徴量の重要度をプロット
plt.figure(figsize=(10, 6))
indices = np.argsort(importances)
plt.barh(range(len(indices)), importances[indices], xerr=std[indices], align='center')
plt.yticks(range(len(indices)), np.array(feature_names)[indices])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance')
plt.show()

#モデルの保存
with open('mpg_model','wb')as file:
    pickle.dump(model,file)
print("モデルが保存されました")


