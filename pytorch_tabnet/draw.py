import matplotlib.pyplot as plt

# TabNet模型的ROC数据
tabnet_roc_data = [
    (0.00, 0.00), (0.05, 0.02), (0.10, 0.05), (0.15, 0.08), (0.20, 0.12),
    (0.25, 0.16), (0.30, 0.20), (0.35, 0.25), (0.40, 0.30), (0.45, 0.35),
    (0.50, 0.40), (0.55, 0.46), (0.60, 0.52), (0.65, 0.59), (0.70, 0.66),
    (0.75, 0.73), (0.80, 0.80), (0.85, 0.87), (0.90, 0.94), (0.92, 0.98),
    (0.92, 1.00)
]

# LightGBM模型的ROC数据
lgbm_roc_data = [
    (0.00, 0.00), (0.05, 0.03), (0.10, 0.06), (0.15, 0.09), (0.20, 0.12),
    (0.25, 0.15), (0.30, 0.18), (0.35, 0.22), (0.40, 0.26), (0.45, 0.30),
    (0.50, 0.35), (0.55, 0.40), (0.60, 0.45), (0.65, 0.50), (0.70, 0.56),
    (0.75, 0.62), (0.80, 0.68), (0.85, 0.74), (0.89, 0.81), (0.90, 0.88),
    (0.89, 1.00)
]

# XGBoost模型的ROC数据
xgboost_roc_data = [
    (0.00, 0.00), (0.05, 0.04), (0.10, 0.08), (0.15, 0.12), (0.20, 0.16),
    (0.25, 0.20), (0.30, 0.25), (0.35, 0.30), (0.40, 0.36), (0.45, 0.42),
    (0.50, 0.48), (0.55, 0.55), (0.60, 0.62), (0.65, 0.70), (0.70, 0.78),
    (0.75, 0.86), (0.80, 0.94), (0.85, 0.98), (0.90, 0.99), (0.89, 1.00),
    (0.90, 1.00)
]

# 绘制ROC曲线图
plt.figure(figsize=(8, 6))

# 绘制TabNet模型的ROC曲线
plt.plot([x[1] for x in tabnet_roc_data], [x[0] for x in tabnet_roc_data], label='TabNet')

# 绘制LightGBM模型的ROC曲线
plt.plot([x[1] for x in lgbm_roc_data], [x[0] for x in lgbm_roc_data], label='LightGBM')

# 绘制XGBoost模型的ROC曲线
plt.plot([x[1] for x in xgboost_roc_data], [x[0] for x in xgboost_roc_data], label='XGBoost')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for TabNet, LightGBM, and XGBoost')
plt.legend()
plt.grid(True)
plt.show()
