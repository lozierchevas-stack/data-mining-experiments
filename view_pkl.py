# view_pkl.py
import joblib
import json
import pprint


def view_model_summary():
    """查看模型摘要"""
    print("=" * 60)
    print("1. 查看模型性能摘要 (model_summary.json)")
    print("=" * 60)

    try:
        with open('model_summary.json', 'r', encoding='utf-8') as f:
            summary = json.load(f)

        print("最佳模型:", summary.get('best_model', '未知'))
        print("测试准确率:", summary.get('test_accuracy', 0))
        print("交叉验证平均准确率:", summary.get('cv_mean_accuracy', 0))
        print("交叉验证标准差:", summary.get('cv_std_accuracy', 0))
        print("特征数量:", summary.get('feature_count', 0))
        print("数据量:", summary.get('data_size', 0))

        print("\n混淆矩阵:")
        conf_matrix = summary.get('confusion_matrix', [])
        if conf_matrix:
            print(f"            预测:正常  预测:垃圾")
            print(f"真实:正常  {conf_matrix[0][0]:>8}  {conf_matrix[0][1]:>8}")
            print(f"真实:垃圾  {conf_matrix[1][0]:>8}  {conf_matrix[1][1]:>8}")
    except Exception as e:
        print(f"读取model_summary.json失败: {e}")


def view_best_model():
    """查看最佳模型信息"""
    print("\n" + "=" * 60)
    print("2. 查看最佳模型 (best_sms_classifier.pkl)")
    print("=" * 60)

    try:
        model = joblib.load('best_sms_classifier.pkl')
        print(f"模型类型: {type(model).__name__}")

        # 获取模型参数
        if hasattr(model, 'get_params'):
            params = model.get_params()
            print("\n模型参数:")
            for key, value in list(params.items())[:10]:  # 只显示前10个参数
                print(f"  {key}: {value}")

        # 对于逻辑回归模型，显示特征重要性
        if hasattr(model, 'coef_'):
            print(f"\n模型权重维度: {model.coef_.shape}")
            if hasattr(model, 'intercept_'):
                print(f"截距项: {model.intercept_}")

        # 对于朴素贝叶斯模型
        elif hasattr(model, 'class_count_'):
            print(f"\n类别计数: {model.class_count_}")
            if hasattr(model, 'feature_log_prob_'):
                print(f"特征对数概率维度: {model.feature_log_prob_.shape}")

        # 对于随机森林模型
        elif hasattr(model, 'estimators_'):
            print(f"\n树的数量: {len(model.estimators_)}")
            if hasattr(model, 'feature_importances_'):
                print(f"特征重要性维度: {model.feature_importances_.shape}")

    except Exception as e:
        print(f"加载模型失败: {e}")


def view_vectorizer():
    """查看向量化器信息"""
    print("\n" + "=" * 60)
    print("3. 查看TF-IDF向量化器 (tfidf_vectorizer.pkl)")
    print("=" * 60)

    try:
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print(f"向量化器类型: {type(vectorizer).__name__}")

        if hasattr(vectorizer, 'get_feature_names_out'):
            features = vectorizer.get_feature_names_out()
            print(f"特征数量: {len(features)}")
            print(f"特征示例 (前20个):")
            for i, feature in enumerate(features[:20]):
                print(f"  {i + 1:2d}. {feature}")

            print(f"\nTF-IDF参数:")
            print(f"  max_features: {vectorizer.max_features}")
            print(f"  min_df: {vectorizer.min_df}")
            print(f"  max_df: {vectorizer.max_df}")
            print(f"  ngram_range: {vectorizer.ngram_range}")

    except Exception as e:
        print(f"加载向量化器失败: {e}")


def test_model_prediction():
    """测试模型预测功能"""
    print("\n" + "=" * 60)
    print("4. 测试模型预测功能")
    print("=" * 60)

    try:
        # 加载模型和向量化器
        model = joblib.load('best_sms_classifier.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')

        # 测试短信
        test_messages = [
            "家长您好：新学期数学提高班开课啦，报名电话13800138000",
            "晚上一起吃饭吧，顺便讨论下周末的旅游项目",
            "恭喜您获得免费抽奖机会，点击链接领取大奖！",
            "商业秘密的保护对于企业至关重要"
        ]

        # 定义与训练时相同的预处理函数
        import jieba

        def preprocess_text(text):
            """简化的预处理函数"""
            stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人'])
            words = jieba.lcut(text)
            filtered_words = [w for w in words if w not in stopwords and len(w.strip()) > 1]
            return ' '.join(filtered_words)

        print("预测结果:")
        for i, msg in enumerate(test_messages, 1):
            # 预处理
            processed = preprocess_text(msg)

            # 向量化
            vectorized = vectorizer.transform([processed])

            # 预测
            prediction = model.predict(vectorized)[0]

            # 如果有概率预测
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(vectorized)[0]
                normal_prob = proba[0]
                spam_prob = proba[1]
            else:
                normal_prob = 0.5
                spam_prob = 0.5

            print(f"\n短信{i}: '{msg[:30]}...'")
            print(f"  预测: {'✅ 正常短信' if prediction == 0 else '🚫 垃圾短信'}")
            print(f"  置信度: 正常{normal_prob * 100:.1f}%, 垃圾{spam_prob * 100:.1f}%")

    except Exception as e:
        print(f"测试预测功能失败: {e}")


if __name__ == "__main__":
    print("垃圾短信识别模型文件查看器")
    print("=" * 60)

    # 检查文件是否存在
    import os

    files = ['model_summary.json', 'best_sms_classifier.pkl', 'tfidf_vectorizer.pkl']
    missing_files = [f for f in files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        print("请确保您已运行垃圾短信识别程序并生成了这些文件")
    else:
        view_model_summary()
        view_best_model()
        view_vectorizer()
        test_model_prediction()

    print("\n" + "=" * 60)
    print("查看完成！")