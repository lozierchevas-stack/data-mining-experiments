# -*- coding: utf-8 -*-
"""
二手房数据分析预测系统
完整版 - 解决numpy数组调用问题
功能：数据预处理、可视化分析、房价预测
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 1. 数据处理模块 ====================
class DataProcessor:
    """数据处理模块 - 去掉空数据、去掉数值单位"""

    def __init__(self):
        self.original_shape = None

    def load_sample_data(self):
        """生成示例二手房数据"""
        print("生成示例二手房数据...")

        np.random.seed(42)
        n_samples = 2000

        # 生成模拟数据
        data = {
            '区域': np.random.choice(['朝阳区', '海淀区', '西城区', '东城区', '丰台区', '石景山区', '通州区'],
                                     n_samples),
            '均价': np.random.randint(30000, 150000, n_samples),
            '户型': np.random.choice(['1室1厅', '2室1厅', '2室2厅', '3室1厅', '3室2厅', '4室2厅'], n_samples,
                                     p=[0.1, 0.3, 0.25, 0.2, 0.1, 0.05]),
            '装修程度': np.random.choice(['精装修', '简装修', '毛坯', '豪华装修'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
            '面积': np.random.randint(50, 200, n_samples),
            '楼层': np.random.choice(['低层', '中层', '高层'], n_samples, p=[0.3, 0.5, 0.2]),
            '建造年份': np.random.randint(1990, 2023, n_samples),
            '地铁距离(m)': np.random.randint(0, 3000, n_samples),
            '学校距离(m)': np.random.randint(0, 5000, n_samples),
            '售价(万元)': np.random.randint(200, 2000, n_samples)
        }

        df = pd.DataFrame(data)

        # 添加一些空值
        for col in ['均价', '面积', '售价(万元)']:
            null_indices = np.random.choice(df.index, size=int(len(df) * 0.05), replace=False)
            df.loc[null_indices, col] = np.nan

        # 添加单位
        df['均价'] = df['均价'].apply(lambda x: f"{x}元/平米" if pd.notnull(x) else np.nan)
        df['面积'] = df['面积'].apply(lambda x: f"{x}平米" if pd.notnull(x) else np.nan)
        df['售价(万元)'] = df['售价(万元)'].apply(lambda x: f"{x}万元" if pd.notnull(x) else np.nan)

        self.original_shape = df.shape
        print(f"生成数据形状: {df.shape}")
        return df

    def remove_empty_data(self, df):
        """去掉空数据"""
        print("\n步骤1: 去掉空数据...")
        initial_shape = df.shape
        df_cleaned = df.dropna()
        final_shape = df_cleaned.shape
        removed_count = initial_shape[0] - final_shape[0]
        print(f"  原始数据: {initial_shape[0]} 行")
        print(f"  清理后数据: {final_shape[0]} 行")
        print(f"  移除空值行: {removed_count} 行")
        return df_cleaned

    def remove_numeric_units(self, df):
        """去掉数值单位"""
        print("\n步骤2: 去掉数值单位...")
        df_numeric = df.copy()

        # 定义需要处理的列及其单位
        unit_columns = {
            '均价': '元/平米',
            '面积': '平米',
            '售价(万元)': '万元',
            '地铁距离(m)': 'm',
            '学校距离(m)': 'm'
        }

        for col, unit in unit_columns.items():
            if col in df_numeric.columns:
                # 移除单位并转换为数值类型
                df_numeric[col] = df_numeric[col].astype(str).str.replace(unit, '', regex=False)
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                print(f"  处理列 '{col}': 移除单位 '{unit}'")

        return df_numeric

    def preprocess_pipeline(self, data_path=None):
        """完整的数据预处理管道"""
        if data_path:
            print(f"从文件加载数据: {data_path}")
            df = pd.read_csv(data_path, encoding='utf-8')
        else:
            df = self.load_sample_data()

        # 1. 去掉空数据
        df_cleaned = self.remove_empty_data(df)

        # 2. 去掉数值单位
        df_processed = self.remove_numeric_units(df_cleaned)

        print(f"\n预处理完成!")
        print(f"最终数据形状: {df_processed.shape}")
        print(f"数据类型:")
        print(df_processed.dtypes)

        return df_processed


# ==================== 2. 图表工具模块 ====================
class ChartTools:
    """图表工具模块 - 修复numpy数组调用问题"""

    @staticmethod
    def safe_get_axis(ax):
        """安全获取坐标轴对象"""
        if isinstance(ax, np.ndarray):
            if ax.size > 0:
                return ax.flat[0]
            else:
                fig, new_ax = plt.subplots()
                return new_ax
        return ax

    @staticmethod
    def create_district_price_bar(df, output_path='各区二手房均价条形图.png'):
        """绘制各区二手房均价的条形图"""
        print("\n绘制各区二手房均价条形图...")

        if '区域' not in df.columns or '均价' not in df.columns:
            print("错误: 缺少必要的数据列")
            return

        # 计算各区均价
        district_avg_price = df.groupby('区域')['均价'].mean().sort_values(ascending=False)

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.bar(range(len(district_avg_price)), district_avg_price.values,
                      color='skyblue', edgecolor='black', alpha=0.8)

        ax.set_title('各区二手房均价', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('区域', fontsize=12)
        ax.set_ylabel('均价 (元/平米)', fontsize=12)
        ax.set_xticks(range(len(district_avg_price)))
        ax.set_xticklabels(district_avg_price.index, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(district_avg_price.values),
                    f'{height:,.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 图表已保存: {output_path}")

        return district_avg_price

    @staticmethod
    def create_renovation_bar(df, output_path='全市二手房装修程度条形图.png'):
        """绘制全市二手房装修程度的条形图"""
        print("\n绘制全市二手房装修程度条形图...")

        if '装修程度' not in df.columns:
            print("错误: 缺少装修程度数据")
            return

        # 统计装修程度
        renovation_counts = df['装修程度'].value_counts()

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        bars = ax.bar(range(len(renovation_counts)), renovation_counts.values,
                      color=colors[:len(renovation_counts)], edgecolor='black', alpha=0.8)

        ax.set_title('全市二手房装修程度分布', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('装修程度', fontsize=12)
        ax.set_ylabel('数量 (套)', fontsize=12)
        ax.set_xticks(range(len(renovation_counts)))
        ax.set_xticklabels(renovation_counts.index)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(renovation_counts.values),
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 图表已保存: {output_path}")

        return renovation_counts

    @staticmethod
    def create_popular_layout_price_bar(df, top_n=5, output_path='热门户型均价条形图.png'):
        """绘制热门户型均价的条形图"""
        print(f"\n绘制前{top_n}热门户型均价条形图...")

        if '户型' not in df.columns or '均价' not in df.columns:
            print("错误: 缺少必要的数据列")
            return

        # 计算各户型数量和均价
        layout_stats = df.groupby('户型').agg({
            '均价': 'mean',
            '面积': 'count'
        }).rename(columns={'面积': '数量'})

        # 按数量排序，取前top_n
        popular_layouts = layout_stats.nlargest(top_n, '数量')

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 数量条形图
        bars1 = axes[0].bar(range(len(popular_layouts)), popular_layouts['数量'].values,
                            color='lightcoral', edgecolor='black', alpha=0.8)
        axes[0].set_title(f'前{top_n}热门户型数量', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('户型', fontsize=12)
        axes[0].set_ylabel('数量 (套)', fontsize=12)
        axes[0].set_xticks(range(len(popular_layouts)))
        axes[0].set_xticklabels(popular_layouts.index, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')

        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(popular_layouts['数量'].values),
                         f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 均价条形图
        bars2 = axes[1].bar(range(len(popular_layouts)), popular_layouts['均价'].values,
                            color='lightgreen', edgecolor='black', alpha=0.8)
        axes[1].set_title(f'前{top_n}热门户型均价', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('户型', fontsize=12)
        axes[1].set_ylabel('均价 (元/平米)', fontsize=12)
        axes[1].set_xticks(range(len(popular_layouts)))
        axes[1].set_xticklabels(popular_layouts.index, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')

        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(popular_layouts['均价'].values),
                         f'{height:,.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 图表已保存: {output_path}")

        return popular_layouts

    @staticmethod
    def create_price_area_scatter(df, output_path='价格面积散点图.png'):
        """绘制价格与面积的关系散点图"""
        print("\n绘制价格与面积关系散点图...")

        if '均价' not in df.columns or '面积' not in df.columns:
            print("错误: 缺少必要的数据列")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(df['面积'], df['均价'], alpha=0.6, c='royalblue',
                             edgecolors='black', linewidth=0.5, s=50)

        ax.set_title('二手房价格 vs 面积', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('面积 (平米)', fontsize=12)
        ax.set_ylabel('均价 (元/平米)', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 图表已保存: {output_path}")

        return fig, ax

    @staticmethod
    def create_comprehensive_analysis(df, output_path='综合分析图表.png'):
        """创建综合分析图表"""
        print("\n创建综合分析图表...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 区域均价条形图
        if '区域' in df.columns and '均价' in df.columns:
            district_avg_price = df.groupby('区域')['均价'].mean().sort_values(ascending=False)
            bars = axes[0, 0].bar(range(len(district_avg_price)), district_avg_price.values,
                                  color='skyblue', edgecolor='black', alpha=0.8)
            axes[0, 0].set_title('各区二手房均价', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('区域')
            axes[0, 0].set_ylabel('均价 (元/平米)')
            axes[0, 0].set_xticks(range(len(district_avg_price)))
            axes[0, 0].set_xticklabels(district_avg_price.index, rotation=45, ha='right')
            axes[0, 0].grid(True, alpha=0.3, axis='y')

        # 2. 装修程度饼图
        if '装修程度' in df.columns:
            renovation_counts = df['装修程度'].value_counts()
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
            wedges, texts, autotexts = axes[0, 1].pie(
                renovation_counts.values,
                labels=renovation_counts.index,
                colors=colors[:len(renovation_counts)],
                autopct='%1.1f%%',
                startangle=90
            )
            axes[0, 1].set_title('装修程度分布', fontsize=14, fontweight='bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        # 3. 户型均价条形图
        if '户型' in df.columns and '均价' in df.columns:
            layout_avg_price = df.groupby('户型')['均价'].mean().nlargest(6)
            bars = axes[1, 0].bar(range(len(layout_avg_price)), layout_avg_price.values,
                                  color='lightgreen', edgecolor='black', alpha=0.8)
            axes[1, 0].set_title('各户型均价', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('户型')
            axes[1, 0].set_ylabel('均价 (元/平米)')
            axes[1, 0].set_xticks(range(len(layout_avg_price)))
            axes[1, 0].set_xticklabels(layout_avg_price.index, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. 楼层价格箱线图
        if '楼层' in df.columns and '均价' in df.columns:
            floor_data = [df[df['楼层'] == floor]['均价'] for floor in df['楼层'].unique()]
            box = axes[1, 1].boxplot(floor_data, labels=df['楼层'].unique(), patch_artist=True)
            for patch, color in zip(box['boxes'], ['lightcoral', 'lightblue', 'lightgreen']):
                patch.set_facecolor(color)
            axes[1, 1].set_title('不同楼层价格分布', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('楼层')
            axes[1, 1].set_ylabel('均价 (元/平米)')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 图表已保存: {output_path}")


# ==================== 3. 预测模型模块 ====================
class PricePredictor:
    """房价预测模块"""

    def __init__(self):
        self.model = None
        self.feature_importance = None

    def prepare_features(self, df):
        """准备特征数据"""
        print("\n准备房价预测特征...")

        # 选择特征
        features = ['面积', '建造年份', '地铁距离(m)', '学校距离(m)']

        # 只保留存在的特征
        available_features = [f for f in features if f in df.columns]

        if '均价' not in df.columns:
            print("错误: 缺少均价数据")
            return None, None, None

        X = df[available_features].copy()
        y = df['均价'].values

        # 处理缺失值
        X = X.fillna(X.mean())

        return X, y, available_features

    def train_model(self, X, y, features):
        """训练预测模型"""
        print("\n训练房价预测模型...")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"  训练集: {X_train.shape[0]} 个样本")
        print(f"  测试集: {X_test.shape[0]} 个样本")

        # 训练随机森林模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        # 评估模型
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"\n模型性能评估:")
        print(f"  平均绝对误差 (MAE): {mae:.2f} 元/平米")
        print(f"  均方误差 (MSE): {mse:.2f}")
        print(f"  均方根误差 (RMSE): {rmse:.2f} 元/平米")
        print(f"  R²分数: {r2:.4f}")

        self.model = model
        self.feature_importance = dict(zip(features, model.feature_importances_))

        return y_test, y_pred

    def visualize_predictions(self, y_test, y_pred, features, output_path='房价预测结果.png'):
        """可视化预测结果"""
        print("\n生成预测结果图表...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 实际值 vs 预测值
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='royalblue', edgecolors='black', linewidth=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                        'r--', lw=2, label='完美预测线')
        axes[0, 0].set_title('实际值 vs 预测值', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('实际价格 (元/平米)')
        axes[0, 0].set_ylabel('预测价格 (元/平米)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 预测误差分布
        errors = y_pred - y_test
        axes[0, 1].hist(errors, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_title('预测误差分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('预测误差 (元/平米)')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 特征重要性
        if self.feature_importance:
            sorted_features = dict(sorted(self.feature_importance.items(),
                                          key=lambda x: x[1], reverse=True))
            bars = axes[1, 0].barh(range(len(sorted_features)), list(sorted_features.values()),
                                   color='lightgreen', edgecolor='black', alpha=0.8)
            axes[1, 0].set_title('特征重要性', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('重要性分数')
            axes[1, 0].set_yticks(range(len(sorted_features)))
            axes[1, 0].set_yticklabels(sorted_features.keys())
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            axes[1, 0].invert_yaxis()

        # 4. 残差图
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, color='orange', edgecolors='black', linewidth=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_title('预测残差图', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('预测价格 (元/平米)')
        axes[1, 1].set_ylabel('残差 (实际-预测)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ 图表已保存: {output_path}")


# ==================== 4. 主系统 ====================
class SecondHouseAnalysisSystem:
    """二手房数据分析预测系统 - 主类"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.chart_tools = ChartTools()
        self.predictor = PricePredictor()
        self.processed_data = None

    def run_analysis(self, data_path=None):
        """运行完整分析"""
        print("=" * 60)
        print("二手房数据分析预测系统")
        print("=" * 60)

        # 1. 数据预处理
        print("\n阶段1: 数据预处理")
        print("-" * 40)
        self.processed_data = self.data_processor.preprocess_pipeline(data_path)

        if self.processed_data is None or len(self.processed_data) == 0:
            print("错误: 数据预处理失败")
            return

        # 2. 基本统计分析
        print("\n" + "=" * 60)
        print("阶段2: 基本统计分析")
        print("=" * 60)

        print(f"数据形状: {self.processed_data.shape}")
        print(f"数据列: {list(self.processed_data.columns)}")
        print(f"\n数值列统计:")
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            print(f"  {col}:")
            print(f"    均值: {self.processed_data[col].mean():.2f}")
            print(f"    中位数: {self.processed_data[col].median():.2f}")
            print(f"    标准差: {self.processed_data[col].std():.2f}")
            print(f"    范围: [{self.processed_data[col].min():.2f}, {self.processed_data[col].max():.2f}]")

        # 3. 可视化分析
        print("\n" + "=" * 60)
        print("阶段3: 可视化分析")
        print("=" * 60)

        # 3.1 各区二手房均价条形图
        district_prices = self.chart_tools.create_district_price_bar(self.processed_data)

        # 3.2 全市二手房装修程度条形图
        renovation_counts = self.chart_tools.create_renovation_bar(self.processed_data)

        # 3.3 热门户型均价条形图
        popular_layouts = self.chart_tools.create_popular_layout_price_bar(self.processed_data, top_n=5)

        # 3.4 价格与面积关系图
        self.chart_tools.create_price_area_scatter(self.processed_data)

        # 3.5 综合分析图表
        self.chart_tools.create_comprehensive_analysis(self.processed_data)

        # 4. 房价预测
        print("\n" + "=" * 60)
        print("阶段4: 房价预测")
        print("=" * 60)

        # 准备特征
        X, y, features = self.predictor.prepare_features(self.processed_data)

        if X is not None and y is not None:
            # 训练模型
            y_test, y_pred = self.predictor.train_model(X, y, features)

            # 可视化预测结果
            self.predictor.visualize_predictions(y_test, y_pred, features)

        # 5. 保存结果
        print("\n" + "=" * 60)
        print("阶段5: 保存结果")
        print("=" * 60)

        # 保存处理后的数据
        self.processed_data.to_csv('processed_second_hand_houses.csv', index=False, encoding='utf-8-sig')
        print("✅ 处理后的数据已保存: processed_second_hand_houses.csv")

        # 生成分析报告
        self.generate_report()

        print("\n" + "=" * 60)
        print("分析完成!")
        print("=" * 60)

        return self.processed_data

    def generate_report(self):
        """生成分析报告"""
        print("\n生成分析报告...")

        report_data = {
            "数据总量": len(self.processed_data),
            "特征数量": len(self.processed_data.columns),
            "数值特征": list(self.processed_data.select_dtypes(include=[np.number]).columns),
            "分类特征": list(self.processed_data.select_dtypes(include=['object']).columns)
        }

        # 如果有区域数据
        if '区域' in self.processed_data.columns:
            report_data["区域数量"] = self.processed_data['区域'].nunique()
            report_data["区域列表"] = list(self.processed_data['区域'].unique())

        # 如果有均价数据
        if '均价' in self.processed_data.columns:
            report_data["平均价格"] = float(self.processed_data['均价'].mean())
            report_data["最高价格"] = float(self.processed_data['均价'].max())
            report_data["最低价格"] = float(self.processed_data['均价'].min())

        # 保存报告
        import json
        with open('analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print("✅ 分析报告已保存: analysis_report.json")

        # 生成文本报告
        with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
            f.write("二手房数据分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"数据总量: {report_data.get('数据总量', 0)} 套\n")
            f.write(f"特征数量: {report_data.get('特征数量', 0)} 个\n")
            f.write(f"区域数量: {report_data.get('区域数量', 0)} 个\n")
            f.write(f"平均价格: {report_data.get('平均价格', 0):.2f} 元/平米\n")
            f.write(f"价格范围: {report_data.get('最低价格', 0):.2f} - {report_data.get('最高价格', 0):.2f} 元/平米\n")

        print("✅ 分析摘要已保存: analysis_summary.txt")


# ==================== 5. 主程序 ====================
if __name__ == "__main__":
    # 创建系统实例
    system = SecondHouseAnalysisSystem()

    # 运行分析
    # 如果您有自己的数据文件，可以传入文件路径：
    # processed_data = system.run_analysis("your_data_file.csv")

    # 使用示例数据运行
    try:
        processed_data = system.run_analysis()

        # 显示前几行数据
        print("\n处理后的数据样例:")
        print(processed_data.head())

    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()
