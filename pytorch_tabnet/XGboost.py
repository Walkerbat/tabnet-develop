import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# 加载数据集
df = pd.read_csv("insurance_data.csv")

# 预处理数据
df.fillna(0, inplace=True)

# 对分类变量进行独热编码
categorical_columns = ['Insured.sex', 'Marital', 'Car.use', 'Region', 'Territory']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('AMT_Claim', axis=1), df['AMT_Claim'], test_size=0.2, random_state=42
)

# 训练 XGBoost 模型
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = xgb_model.predict(X_test)
y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_xgb > 0.5).astype(int)

# 计算 F1 分数和 AUC
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_xgb)

# 打印 F1 分数和 AUC
print(f"F1 分数：{f1}")
print(f"AUC：{auc}")
