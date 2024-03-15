'''import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 1. 数据准备
# 假设你有一个包含字符串类型特征的数据框 df
# 请替换以下数据为你实际的数据
df = pd.read_csv("../csv/database.csv")

# 分离特征和目标
X = df.drop(columns=["AMT_Claim"])
y = df["AMT_Claim"]

# 2. 字符串特征处理
# 对所有字符串类型的特征进行 Label 编码
le = LabelEncoder()
for column in X.select_dtypes(include="object").columns:
    X[column] = le.fit_transform(X[column])

# 3. 数据划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 再次使用 train_test_split 划分验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)


# 在模型训练之前，对目标进行 reshape
y_train_reshaped = y_train.values.reshape(-1, 1)

# 4. TabNet模型构建
model = TabNetRegressor()

# 5. 模型训练
model.fit(
    X_train.values, y_train_reshaped,
    eval_set=[(X_val.values, y_val.values.reshape(-1, 1))],
    max_epochs=100,
)

# 6. 模型评估
# 使用之前分割的 X_test 和 y_test
y_pred = model.predict(X_test.values)
mse = mean_squared_error(y_test.values, y_pred)
print(f"Mean Squared Error on Test Set: {mse}")

# 7. 预测新数据
# 假设你有新的数据框 X_new 包含要预测的特征
# 对 X_new 进行相同的字符串编码处理
X_new = df.drop(columns=["AMT_Claim"])
# 检查列是否相同
if not set(X.columns) == set(X_new.columns):
    missing_cols = set(X.columns) - set(X_new.columns)
    print(f"Missing columns in X_new: {missing_cols}")
    # 根据需要处理缺失列
    # 例如，可以选择删除或填充这些列
    X_new = X_new[X.columns]

# 使用相同的 LabelEncoder 对象对 X_new 进行编码处理
for column in X_new.select_dtypes(include="object").columns:
    # 将未见过的标签视为缺失值
    X_new[column] = X_new[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

# 预测新数据
predictions = model.predict(X_new.values)

# 输出预测结果
print("Predictions:")
print(predictions)

# 8. 特征重要性
feature_importances = model.feature_importances_

# 输出特征重要性
print("Feature Importances:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")'''

'''import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score

# 1. 数据准备
data = pd.read_csv("../csv/insurance_data.csv")

# 分离特征和目标
X = data.drop(['NB_Claim', 'AMT_Claim'], axis=1)
y = data['AMT_Claim']

# 将标签转换为整数类型
y = y.astype(int)

# 2. 字符串特征处理 - 使用 LabelEncoder
le = LabelEncoder()
for column in X.select_dtypes(include="object").columns:
    X[column] = le.fit_transform(X[column])

# 3. 数据划分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_val = y_val[y_val.isin(y_train)]
X_val = X_val.loc[y_val.index]

# 4. TabNet模型构建
model = TabNetClassifier(multi_class='binary')

# 5. 模型训练
model.fit(
    X_train.values, y_train,
    eval_set=[(X_val.values, y_val)],
    max_epochs=100,
)

# 6. 模型评估
# 使用验证集进行预测
y_pred_proba = model.predict_proba(X_val.values)[:, 1]

# 将概率转换为二进制预测
y_pred_binary = (y_pred_proba > 0.5).astype(int)

# 计算F1分数
f1 = f1_score(y_val, y_pred_binary, average='micro')

# 计算AUC
auc = roc_auc_score(y_val, y_pred_proba, average='micro')

print(f"F1 Score: {f1}")
print(f"AUC: {auc}")'''


'''import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from torch.utils.data import DataLoader, TensorDataset
import torch

# 加载数据集
df = pd.read_csv("../csv/insurance_data.csv")

# 预处理数据
df.fillna(0, inplace=True)

# 对分类变量进行独热编码
categorical_columns = ['Insured.sex', 'Marital', 'Car.use', 'Region', 'Territory']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
non_numeric_columns = df.select_dtypes(include=['object']).columns
df = df.drop(columns=non_numeric_columns)

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['NB_Claim', 'AMT_Claim'], axis=1),
    df['AMT_Claim'],
    test_size=0.2,
    random_state=42
)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train.values.astype('float32'), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.astype('float32'), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values.astype('float32'), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.astype('float32'), dtype=torch.float32)

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 训练TabNet预训练器
tabnet_pretrainer = TabNetPretrainer(n_steps=5, gamma=1.3)

# 设置 max_epochs 参数
max_epochs_pretrainer = 20

tabnet_pretrainer.fit(
    X_train_tensor.numpy(),  # 这里需要将张量转换回 NumPy 数组
    max_epochs=100,
    patience=20,
    batch_size=64,
    virtual_batch_size=32,
    num_workers=0,
)

# 训练TabNet分类器
tabnet_classifier = TabNetClassifier(n_steps=5, gamma=1.3)
tabnet_classifier.fit(
    X_train_tensor.numpy(),
    y_train_tensor.numpy(),
    eval_set=[(X_test_tensor.numpy(), y_test_tensor.numpy())],
    max_epochs=100,
    patience=20,
    batch_size=64,
    virtual_batch_size=32,
    num_workers=0,
)

# 在测试集上进行预测
y_pred_proba = tabnet_classifier.predict_proba(X_test_tensor.numpy())[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)  # 将概率转换为二进制预测

# 计算F1分数和AUC
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"F1 SCORE：{f1}")
print(f"AUC：{auc}")'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from torch.utils.data import DataLoader, TensorDataset
import torch

# 加载数据
df = pd.read_csv("../csv/insurance_data.csv")

# 预处理数据
df.fillna(0, inplace=True)

# 对分类变量进行独热编码
categorical_columns = ['Insured.sex', 'Marital', 'Car.use', 'Region', 'Territory']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['NB_Claim', 'AMT_Claim'], axis=1), df['AMT_Claim'], test_size=0.2, random_state=42
)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train.values.astype('float32'), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.astype('float32'), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values.astype('float32'), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values.astype('float32'), dtype=torch.float32)

# 将 y_test_tensor 的类型转换为 torch.long
y_test_tensor = y_test_tensor.long()

# 创建 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor.long())  # 将 y_test_tensor 转换为 torch.long
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

# 训练 TabNet 预训练器
tabnet_pretrainer = TabNetPretrainer(n_steps=5, gamma=1.3)

# 设置 max_epochs 参数
max_epochs_pretrainer = 10

tabnet_pretrainer.fit(
    X_train_tensor.numpy(),  # 这里需要将张量转换回 NumPy 数组
    max_epochs=max_epochs_pretrainer,
    patience=20,
    batch_size=64,
    virtual_batch_size=32,
    num_workers=0,
)

# 训练 TabNet 分类器
tabnet_classifier = TabNetClassifier(n_steps=5, gamma=1.3)

tabnet_classifier.fit(
    X_train_tensor.numpy(),
    y_train_tensor.numpy(),
    eval_set=[(X_test_tensor.numpy(), y_test_tensor.numpy())],  # 这里不需要转换 y_test_tensor 的类型，因为在创建 DataLoader 时已经转换过了
    max_epochs=100,
    patience=20,
    batch_size=64,
    virtual_batch_size=32,
    num_workers=0,
)


# 在测试集上进行预测
y_pred_proba = tabnet_classifier.predict_proba(X_test_tensor.numpy())[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# 计算 F1 分数和 AUC
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"F1 分数：{f1}")
print(f"AUC：{auc}")
