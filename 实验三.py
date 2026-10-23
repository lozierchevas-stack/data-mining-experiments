# -*- coding: utf-8 -*-
"""
垃圾短信识别系统 - 独立可视化图片版本
每个可视化图表保存为单独的图片文件
"""
import numpy as np
import pandas as pd
import jieba
import random
import warnings

warnings.filterwarnings("ignore")

# 可视化库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 加载和预处理数据 ====================
print("=" * 60)
print("步骤1: 数据加载与预处理")
print("=" * 60)


# 创建更真实的模拟数据
def create_realistic_sms_data(n_samples=5000):
    """创建更真实的短信数据"""

    spam_templates = [
        "{}新春第一批限量{}到店啦，{}，{}，{}，欢迎选购！电话：{}",
        "尊敬的{}用户，恭喜您获得{}元{}，点击链接{}领取，回复TD退订",
        "{}店庆大促销，{}折起，{}，{}，地址：{}，电话：{}",
    ]

    normal_templates = [
        "{}，晚上一起吃饭吧，顺便讨论下{}项目",
        "{}老师，{}同学的{}成绩出来了，请查收",
        "{}会议安排在{}，地点：{}，请准时参加",
    ]

    brands = ["阿玛施", "金汕教育", "韩式烧烤", "全金瓷", "红棉百货"]
    products = ["春装", "小西装", "公主衫", "淑女裙", "长半裙"]
    actions = ["到店", "限量", "特价", "优惠", "满减"]

    data = []

    for i in range(n_samples):
        msg_id = i + 1
        label = np.random.choice([0, 1], p=[0.85, 0.15])

        if label == 1:  # 垃圾短信
            template = random.choice(spam_templates)
            phone = f"1{random.randint(3, 9)}{''.join([str(random.randint(0, 9)) for _ in range(9)])}"
            content = template.format(
                random.choice(brands),
                random.choice(products),
                random.choice(actions),
                random.choice(actions),
                random.choice(["金城路", "南京路", "北京路"]),
                phone
            )
        else:  # 正常短信
            template = random.choice(normal_templates)
            content = template.format(
                random.choice(["家长您好", "同事", "王先生"]),
                random.choice(["工作", "学习", "旅游"]),
                random.choice(["语文", "数学", "英语"]),
            )

        data.append([msg_id, label, content])

    return pd.DataFrame(data, columns=['id', 'label', 'message'])


# 生成数据
df = create_realistic_sms_data(5000)
print(f"数据形状: {df.shape}")
print(f"类别分布:\n{df['label'].value_counts()}")

# 去重
df = df.drop_duplicates(subset=['message'], keep='first')
print(f"去重后数据量: {len(df)}")

# 保存数据信息
data_info = {
    "total_samples": len(df),
    "normal_sms": df['label'].value_counts().get(0, 0),
    "spam_sms": df['label'].value_counts().get(1, 0),
    "normal_ratio": df['label'].value_counts().get(0, 0) / len(df),
    "spam_ratio": df['label'].value_counts().get(1, 0) / len(df)
}

# ==================== 2. 生成独立的数据可视化图片 ====================
print("\n" + "=" * 60)
print("步骤2: 生成独立的数据可视化图片")
print("=" * 60)

# 图片1: 类别分布图
print("生成图片1: 类别分布图...")
plt.figure(figsize=(8, 6))
category_counts = df['label'].value_counts()
colors = ['#4CAF50', '#FF5722']
bars = plt.bar(['正常短信', '垃圾短信'], category_counts.values,
               color=colors, edgecolor='black', alpha=0.8)
plt.title('短信类别分布', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('短信类别', fontsize=12)
plt.ylabel('数量', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# 在柱状图上添加数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
             f'{int(height)}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('01_类别分布图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 01_类别分布图.png")

# 图片2: 短信长度分布图
print("生成图片2: 短信长度分布图...")
plt.figure(figsize=(10, 6))
df['msg_length'] = df['message'].apply(len)
normal_lengths = df[df['label'] == 0]['msg_length']
spam_lengths = df[df['label'] == 1]['msg_length']

plt.hist(normal_lengths, bins=20, alpha=0.7, label='正常短信',
         color='#4CAF50', edgecolor='black')
plt.hist(spam_lengths, bins=20, alpha=0.7, label='垃圾短信',
         color='#FF5722', edgecolor='black')
plt.title('短信长度分布', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('字符数', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_短信长度分布图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 02_短信长度分布图.png")

# 图片3: 高频词统计
print("生成图片3: 高频词统计图...")
from collections import Counter

all_text = ' '.join(df['message'].tolist())
words = jieba.lcut(all_text)
word_counts = Counter(words)
word_counts = {k: v for k, v in word_counts.items() if len(k) > 1}
top_words = dict(sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:15])

plt.figure(figsize=(12, 8))
bars = plt.barh(list(top_words.keys()), list(top_words.values()),
                color='#2196F3', edgecolor='black', alpha=0.8)
plt.title('Top 15 高频词', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('出现次数', fontsize=12)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

# 在条形图上添加数值
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2.,
             f'{int(width)}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('03_高频词统计图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 03_高频词统计图.png")

# 图片4: 正常短信高频词
print("生成图片4: 正常短信高频词图...")
normal_text = ' '.join(df[df['label'] == 0]['message'].tolist())
if normal_text:
    normal_words = jieba.lcut(normal_text)
    normal_word_counts = Counter(normal_words)
    normal_word_counts = {k: v for k, v in normal_word_counts.items() if len(k) > 1}
    top_normal_words = dict(sorted(normal_word_counts.items(), key=lambda x: x[1], reverse=True)[:12])

    plt.figure(figsize=(10, 6))
    bars = plt.barh(list(top_normal_words.keys()), list(top_normal_words.values()),
                    color='#4CAF50', edgecolor='black', alpha=0.8)
    plt.title('正常短信 Top 12 高频词', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('出现次数', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2.,
                 f'{int(width)}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('04_正常短信高频词图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 已保存: 04_正常短信高频词图.png")

# 图片5: 垃圾短信高频词
print("生成图片5: 垃圾短信高频词图...")
spam_text = ' '.join(df[df['label'] == 1]['message'].tolist())
if spam_text:
    spam_words = jieba.lcut(spam_text)
    spam_word_counts = Counter(spam_words)
    spam_word_counts = {k: v for k, v in spam_word_counts.items() if len(k) > 1}
    top_spam_words = dict(sorted(spam_word_counts.items(), key=lambda x: x[1], reverse=True)[:12])

    plt.figure(figsize=(10, 6))
    bars = plt.barh(list(top_spam_words.keys()), list(top_spam_words.values()),
                    color='#FF5722', edgecolor='black', alpha=0.8)
    plt.title('垃圾短信 Top 12 高频词', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('出现次数', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2.,
                 f'{int(width)}', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('05_垃圾短信高频词图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 已保存: 05_垃圾短信高频词图.png")

# 图片6: 短信长度箱线图对比
print("生成图片6: 短信长度箱线图...")
plt.figure(figsize=(8, 6))
data_to_plot = [normal_lengths, spam_lengths]
box = plt.boxplot(data_to_plot, labels=['正常短信', '垃圾短信'],
                  patch_artist=True, widths=0.6)
colors = ['#4CAF50', '#FF5722']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.title('短信长度箱线图对比', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('字符数', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# 添加统计信息
for i, data in enumerate(data_to_plot, 1):
    plt.text(i, data.median() + 1, f'中位数: {int(data.median())}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('06_短信长度箱线图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 06_短信长度箱线图.png")

# ==================== 3. 文本预处理 ====================
print("\n" + "=" * 60)
print("步骤3: 文本预处理")
print("=" * 60)

# 定义停用词
stopwords = set([
    '的', '了', '在', '是', '我', '有', '和', '就',
    '不', '人', '都', '一', '个', '上', '也', '很',
])


def preprocess_text(text):
    """文本预处理：分词 + 停用词过滤"""
    words = jieba.lcut(text)
    filtered_words = [w for w in words if w not in stopwords and len(w.strip()) > 1]
    return ' '.join(filtered_words)


print("正在进行文本预处理（分词 + 停用词过滤）...")
df['processed'] = df['message'].apply(preprocess_text)
print("预处理完成！")

# ==================== 4. 特征提取 ====================
print("\n" + "=" * 60)
print("步骤4: 特征提取 (TF-IDF)")
print("=" * 60)

vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=3,
    max_df=0.8,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['processed'])
y = df['label']

print(f"特征矩阵形状: {X.shape}")
print(f"特征数: {len(vectorizer.get_feature_names_out())}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]} 条")
print(f"测试集: {X_test.shape[0]} 条")

# ==================== 5. 多模型训练与比较 ====================
print("\n" + "=" * 60)
print("步骤5: 多模型训练与比较")
print("=" * 60)

models = {
    '朴素贝叶斯': MultinomialNB(),
    '逻辑回归': LogisticRegression(max_iter=1000, random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    print(f"训练 {name}...")
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': conf_mat,
        'report': report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

    print(f"  {name} - 准确率: {accuracy:.4f}")

# 确定最佳模型
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_result = results[best_model_name]
print(f"\n最佳模型: {best_model_name} (准确率: {best_result['accuracy']:.4f})")

# ==================== 6. 生成独立的模型性能可视化图片 ====================
print("\n" + "=" * 60)
print("步骤6: 生成独立的模型性能可视化图片")
print("=" * 60)

# 图片7: 模型准确率对比图
print("生成图片7: 模型准确率对比图...")
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
cv_means = [results[name]['cv_mean'] for name in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = plt.bar(x - width / 2, accuracies, width, label='测试集准确率',
                color='#4CAF50', edgecolor='black', alpha=0.8)
bars2 = plt.bar(x + width / 2, cv_means, width, label='交叉验证平均',
                color='#2196F3', edgecolor='black', alpha=0.8)

plt.xlabel('模型', fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.title('模型准确率对比', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, model_names)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0, 1.0])

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('07_模型准确率对比图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 07_模型准确率对比图.png")

# 图片8: 最佳模型混淆矩阵
print("生成图片8: 最佳模型混淆矩阵图...")
plt.figure(figsize=(8, 6))
conf_mat = best_result['confusion_matrix']
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测正常', '预测垃圾'],
            yticklabels=['真实正常', '真实垃圾'],
            cbar_kws={'label': '样本数'})
plt.title(f'{best_model_name} - 混淆矩阵', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.tight_layout()
plt.savefig('08_最佳模型混淆矩阵图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 08_最佳模型混淆矩阵图.png")

# 图片9: ROC曲线
print("生成图片9: ROC曲线图...")
plt.figure(figsize=(8, 6))
for name, result in results.items():
    if result['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率', fontsize=12)
plt.ylabel('真正率', fontsize=12)
plt.title('ROC曲线', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('09_ROC曲线图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 09_ROC曲线图.png")

# 图片10: 逻辑回归特征重要性
print("生成图片10: 逻辑回归特征重要性图...")
if '逻辑回归' in results and hasattr(results['逻辑回归']['model'], 'coef_'):
    feature_names = vectorizer.get_feature_names_out()
    coefs = results['逻辑回归']['model'].coef_[0]

    top_n = 12
    top_indices = np.argsort(np.abs(coefs))[-top_n:]
    top_features = feature_names[top_indices]
    top_coefs = coefs[top_indices]

    plt.figure(figsize=(10, 6))
    colors = ['#FF5722' if c > 0 else '#4CAF50' for c in top_coefs]
    bars = plt.barh(range(len(top_features)), top_coefs, color=colors,
                    edgecolor='black', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('系数值', fontsize=12)
    plt.title('逻辑回归特征重要性 (Top 12)', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    plt.gca().invert_yaxis()

    # 添加系数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (0.01 if width >= 0 else -0.01),
                 bar.get_y() + bar.get_height() / 2.,
                 f'{width:.3f}',
                 ha='left' if width >= 0 else 'right',
                 va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('10_逻辑回归特征重要性图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 已保存: 10_逻辑回归特征重要性图.png")

# 图片11: 分类报告热力图
print("生成图片11: 分类报告热力图...")
report = best_result['report']
if '0' in report and '1' in report:
    metrics_df = pd.DataFrame({
        '正常短信': [report['0']['precision'], report['0']['recall'], report['0']['f1-score']],
        '垃圾短信': [report['1']['precision'], report['1']['recall'], report['1']['f1-score']]
    }, index=['精确率', '召回率', 'F1分数'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': '分数'}, square=True)
    plt.title(f'{best_model_name} - 分类指标', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('11_分类报告热力图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 已保存: 11_分类报告热力图.png")

# 图片12: 预测置信度分布
print("生成图片12: 预测置信度分布图...")
if best_result['y_pred_proba'] is not None:
    correct_mask = (best_result['y_pred'] == y_test.values)
    wrong_mask = ~correct_mask

    plt.figure(figsize=(10, 6))
    n_bins = 20
    plt.hist(best_result['y_pred_proba'][correct_mask], bins=n_bins,
             alpha=0.7, label='正确预测', color='#4CAF50', edgecolor='black')
    plt.hist(best_result['y_pred_proba'][wrong_mask], bins=n_bins,
             alpha=0.7, label='错误预测', color='#FF5722', edgecolor='black')

    plt.xlabel('预测为垃圾短信的置信度', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('预测置信度分布', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    correct_mean = np.mean(best_result['y_pred_proba'][correct_mask])
    wrong_mean = np.mean(best_result['y_pred_proba'][wrong_mask])
    plt.axvline(x=correct_mean, color='#4CAF50', linestyle='--', alpha=0.8,
                label=f'正确预测均值: {correct_mean:.3f}')
    plt.axvline(x=wrong_mean, color='#FF5722', linestyle='--', alpha=0.8,
                label=f'错误预测均值: {wrong_mean:.3f}')
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('12_预测置信度分布图.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 已保存: 12_预测置信度分布图.png")

# ==================== 7. 生成模型性能摘要 ====================
print("\n" + "=" * 60)
print("步骤7: 生成模型性能摘要")
print("=" * 60)

# 图片13: 模型性能摘要
print("生成图片13: 模型性能摘要图...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

# 准备表格数据 - 确保有14行
table_data = [
    ["数据集信息", ""],
    ["总样本数", f"{data_info['total_samples']}"],
    ["正常短信数", f"{data_info['normal_sms']} ({data_info['normal_ratio']:.1%})"],
    ["垃圾短信数", f"{data_info['spam_sms']} ({data_info['spam_ratio']:.1%})"],
    ["", ""],
    ["特征工程", ""],
    ["特征数量", f"{X.shape[1]}"],
    ["训练集大小", f"{X_train.shape[0]}"],
    ["测试集大小", f"{X_test.shape[0]}"],
    ["", ""],
    ["模型性能", ""],
    ["最佳模型", f"{best_model_name}"],
    ["测试准确率", f"{best_result['accuracy']:.4f}"],
    ["交叉验证平均", f"{best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})"],
]

# 检查行数
print(f"表格行数: {len(table_data)}")  # 应该是14

# 创建颜色配置 - 确保有14行，每行2个颜色
rows = len(table_data)  # 14
cols = 2

# 创建14行×2列的颜色矩阵
cell_colors = []
for i in range(rows):
    if i in [0, 4, 9, 11]:  # 标题行
        cell_colors.append(['#E3F2FD', '#E3F2FD'])
    elif i == 2:  # 正常短信行
        cell_colors.append(['#F5F5F5', '#E8F5E8'])
    elif i == 3:  # 垃圾短信行
        cell_colors.append(['#FFFFFF', '#FCE4EC'])
    else:  # 其他行
        cell_colors.append(['#F5F5F5', '#FFFFFF'])

# 验证颜色矩阵的维度
print(f"颜色矩阵行数: {len(cell_colors)}")
print(f"颜色矩阵每行列数: {len(cell_colors[0]) if cell_colors else 0}")

# 创建表格 - 第546行
table = ax.table(cellText=table_data,
                 cellLoc='left',
                 colWidths=[0.4, 0.6],
                 loc='center',
                 cellColours=cell_colors)  # 使用修正后的颜色列表

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# 设置标题单元格样式
for (row, col), cell in table.get_celld().items():
    if row in [0, 4, 9, 11]:  # 标题行
        cell.set_text_props(weight='bold', size=12)
    if col == 0 and row not in [0, 4, 9, 11]:  # 第一列（非标题行）
        cell.set_text_props(weight='bold')

plt.title('垃圾短信识别模型性能摘要', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('13_模型性能摘要图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 13_模型性能摘要图.png")

# 图片14: 预测结果可视化
print("\n生成图片14: 预测结果示例图...")
print("\n生成图片14: 预测结果示例图...")

# 确保predictions已经被定义
if 'predictions' not in locals() and 'predictions' not in globals():
    print("⚠️ 警告: predictions变量未定义，正在重新生成...")

    # 如果test_messages也没有定义，先定义它
    if 'test_messages' not in locals() and 'test_messages' not in globals():
        test_messages = [
            "家长您好：新学期数学提高班开课啦，报名电话13800138000，金汕教育",
            "晚上一起吃饭吧，顺便讨论下周末的旅游项目",
            "恭喜您获得免费抽奖机会，点击链接领取大奖！回复TD退订",
            "商业秘密的保护对于企业至关重要，是维持竞争优势的前提",
            "阿玛施新春第一批限量春装到店啦，欢迎选购！",
            "您的快递已发货，快递单号：SF123456789，预计明天送达",
        ]

    # 使用最佳模型进行预测
    best_model = best_result['model']
    predictions = []

    for i, msg in enumerate(test_messages, 1):
        processed = preprocess_text(msg)
        vectorized = vectorizer.transform([processed])

        prediction = best_model.predict(vectorized)[0]

        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(vectorized)[0]
            normal_prob = proba[0]
            spam_prob = proba[1]
        else:
            normal_prob = 0.5
            spam_prob = 0.5

        predictions.append({
            "短信": msg[:30] + "...",
            "预测结果": "正常短信" if prediction == 0 else "垃圾短信",
            "正常概率": f"{normal_prob:.1%}",
            "垃圾概率": f"{spam_prob:.1%}",
            "置信度": "高" if max(normal_prob, spam_prob) > 0.7 else "中"
        })
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# 准备表格数据
table_data = [["序号", "短信内容", "预测结果", "正常概率", "垃圾概率", "置信度"]]
for i, pred in enumerate(predictions, 1):
    table_data.append([
        str(i),
        pred["短信"],
        pred["预测结果"],
        pred["正常概率"],
        pred["垃圾概率"],
        pred["置信度"]
    ])

# 创建表格
table = ax.table(cellText=table_data,
                 cellLoc='center',
                 colWidths=[0.08, 0.4, 0.15, 0.12, 0.12, 0.13],
                 loc='center',
                 cellColours=[['#E3F2FD'] * 6] +
                             [['#FFFFFF', '#F5F5F5', '#E8F5E8' if cell == "正常短信" else '#FCE4EC',
                               '#E8F5E8', '#FCE4EC', '#FFF3E0'] for cell in [row[2] for row in table_data[1:]]])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# 设置标题行样式
for j in range(len(table_data[0])):
    table[(0, j)].set_text_props(weight='bold', size=11)
    table[(0, j)].set_facecolor('#1976D2')
    table[(0, j)].set_text_props(color='white')

plt.title('新短信预测结果示例', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('14_预测结果示例图.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ 已保存: 14_预测结果示例图.png")

# ==================== 9. 输出总结 ====================
print("\n" + "=" * 60)
print("垃圾短信识别系统 - 独立可视化图片生成完成！")
print("=" * 60)
print(f"\n🎉 共生成 {14} 个独立的可视化图片文件：")
print("\n【数据可视化】")
print("  01_类别分布图.png           - 显示正常和垃圾短信的数量分布")
print("  02_短信长度分布图.png       - 显示两类短信的字符数分布")
print("  03_高频词统计图.png         - 显示所有短信中的高频词汇")
print("  04_正常短信高频词图.png     - 显示正常短信特有的高频词")
print("  05_垃圾短信高频词图.png     - 显示垃圾短信特有的高频词")
print("  06_短信长度箱线图.png       - 对比两类短信的长度统计特征")
print("\n【模型性能】")
print("  07_模型准确率对比图.png     - 比较不同模型的准确率")
print("  08_最佳模型混淆矩阵图.png   - 显示最佳模型的预测结果矩阵")
print("  09_ROC曲线图.png           - 显示各模型的ROC曲线和AUC值")
print("  10_逻辑回归特征重要性图.png - 显示最重要的特征词")
print("  11_分类报告热力图.png       - 以热力图形式展示分类指标")
print("  12_预测置信度分布图.png     - 显示模型预测的置信度分布")
print("\n【结果摘要】")
print("  13_模型性能摘要图.png       - 汇总所有关键性能指标")
print("  14_预测结果示例图.png       - 展示对新短信的预测结果")
print("\n📁 所有图片已保存在当前目录，您可以使用图片查看器逐个查看。")
