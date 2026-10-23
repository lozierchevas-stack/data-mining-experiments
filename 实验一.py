import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# ---------------------- 1. 数据加载 ----------------------
# 加载训练集和测试集（假设文件在当前目录）
train_data = pd.read_csv("train.txt", sep="\t")  # 若分隔符是逗号则用","
test_data = pd.read_csv("test.txt", sep="\t")
# 提取特征和目标变量
X_train = train_data[["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
                      "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                      "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29",
                      "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37"]]
y_train = train_data["target"]
X_test = test_data[["V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
                    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "V29",
                    "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37"]]
# ---------------------- 2. 数据处理 ----------------------
# 2.1 属性过滤（这里假设保留所有特征，可根据实际需求筛选）
# 2.2 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 2.3 异常值检测（用3σ原则）
def detect_outliers(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    outliers = (np.abs(data - mean) > 3 * std)
    return ~outliers  # 返回非异常值的掩码
# 过滤训练集异常值
valid_mask = detect_outliers(X_train_scaled)
X_train_clean = X_train_scaled[valid_mask.all(axis=1)]
y_train_clean = y_train[valid_mask.all(axis=1)]

# 2.4 设置角色（已完成特征/目标划分）
# 2.5 数据拆分（训练集再拆分验证集）
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_clean, y_train_clean, test_size=0.2, random_state=42
)
# ---------------------- 3. 模型训练（回归预测） ----------------------
model = LinearRegression()
model.fit(X_train_split, y_train_split)
# ---------------------- 4. 模型评估与可视化 ----------------------
# 验证集预测
y_val_pred = model.predict(X_val_split)
mse = mean_squared_error(y_val_split, y_val_pred)
print(f"验证集均方根误差（RMSE）：{np.sqrt(mse):.4f}")
# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_val_split, y_val_pred, alpha=0.6)
plt.plot([y_val_split.min(), y_val_split.max()], [y_val_split.min(), y_val_split.max()], "r--")
plt.xlabel("实际蒸汽量")
plt.ylabel("预测蒸汽量")
plt.title("实际值 vs 预测值")
plt.show()
# ---------------------- 5. 测试集预测（工程化部署用） ----------------------
y_test_pred = model.predict(X_test_scaled)
# 保存预测结果
test_data["pred_target"] = y_test_pred
test_data.to_csv("test_pred_result.txt", sep="\t", index=False)
print("测试集预测结果已保存至 test_pred_result.txt")