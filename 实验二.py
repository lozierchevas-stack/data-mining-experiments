import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# 配置Matplotlib使用支持中文的字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定中文字体
plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示异常问题

# ---------------------- 1. 数据读取与预处理 ----------------------
# 读取数据
data = pd.read_csv("公募基金购买.csv")

# 生成“周转率”特征
data["周转率"] = data["股票交易量"] / data["近半年日均资产"]

# 选择客户细分的特征（开户年限、年龄、近半年日均资产、半年内股票持仓比例、股票佣金、周转率）
cluster_features = ["开户年限", "年龄", "近半年日均资产", "半年内股票持仓比例", "股票佣金", "周转率"]
X_cluster = data[cluster_features]

# 数据标准化（KMeans对数据尺度敏感）
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)


# ---------------------- 2. 客户细分（KMeans聚类） ----------------------
# 构建KMeans模型（假设聚为3类）
kmeans = KMeans(n_clusters=3, random_state=42)
data["cluster_label"] = kmeans.fit_predict(X_cluster_scaled)

# 还原聚类中心到原始特征尺度
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_center_df = pd.DataFrame(cluster_centers, columns=cluster_features)
print("=== 聚类中心（原始特征）===")
print(cluster_center_df)

# 可视化任意两个特征的聚类结果（以“近半年日均资产”和“周转率”为例）
plt.figure(figsize=(10, 6))
sns.scatterplot(x="近半年日均资产", y="周转率", hue="cluster_label", data=data, palette="viridis", s=60)
plt.scatter(cluster_center_df["近半年日均资产"], cluster_center_df["周转率"],
            c="red", marker="X", s=200, label="聚类中心")
plt.title("客户细分聚类结果（近半年日均资产 vs 周转率）")
plt.legend()
plt.show()


# ---------------------- 3. 潜客预测（梯度提升决策树） ----------------------
# 选择预测特征（风险等级、开户年限、年龄、总资产、近一年最大资产、近半年日均资产、股票交易次数、近两年最后买公募基金天数、近半年理财持有比例）
predict_features = ["风险等级", "开户年限", "年龄", "总资产", "近一年最大资产",
                   "近半年日均资产", "股票交易次数", "近两年最后买公募基金天数", "近半年理财持有比例"]
X_predict = data[predict_features]
y_predict = data["flag"]  # 假设数据中存在“是否购买”标签

# 划分训练集和测试集（70%训练，30%测试）
X_train, X_test, y_train, y_test = train_test_split(X_predict, y_predict, test_size=0.3, random_state=42)

# 构建梯度提升决策树模型
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# 预测与评估
y_pred = gbc.predict(X_test)
print("\n=== 潜客预测模型评估 ===")
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))
print("\n分类报告：")
print(classification_report(y_test, y_pred))


# ---------------------- 4. 输出结果 ----------------------
# 输出高概率购买的潜客名单
data["购买概率"] = gbc.predict_proba(X_predict)[:, 1]  # 预测购买的概率
potential_customers = data[data["购买概率"] >= 0.7]  # 筛选概率≥0.7的客户
potential_customers.to_csv("公募基金潜客名单.csv", index=False)
print("\n高概率购买潜客名单已保存至“公募基金潜客名单.csv”")