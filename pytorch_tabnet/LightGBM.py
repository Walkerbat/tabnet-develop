import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score

# 加载数据集
df = pd.read_csv("../csv/insurance_data.csv")

# 预处理数据
df.fillna(0, inplace=True)

# 对分类变量进行独热编码
categorical_columns = ['Insured.sex', 'Marital', 'Car.use', 'Region', 'Territory']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('AMT_Claim', axis=1), df['AMT_Claim'], test_size=0.2, random_state=42
)

# Train LGBMRegressor model
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1)
lgbm_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_gbm = lgbm_model.predict(X_test)

# 计算 F1 分数和 AUC
f1 = f1_score(y_test, y_pred_gbm)
auc = roc_auc_score(y_test, y_pred_gbm)

print(f"F1 分数：{f1}")
print(f"AUC：{auc}")

