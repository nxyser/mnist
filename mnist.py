import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib
import platform
import os
from sklearn.model_selection import train_test_split
import time
import seaborn as sns


# 设置中文字体支持
def set_chinese_font():
    # 根据操作系统设置字体路径
    system = platform.system()
    if system == 'Windows':
        # Windows 系统通常自带微软雅黑
        font_path = 'C:/Windows/Fonts/msyh.ttc'
        font_name = 'Microsoft YaHei'


    # 检查字体文件是否存在
    if os.path.exists(font_path):
        # 添加字体到matplotlib
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        # 设置全局字体
        matplotlib.rc('font', family=font_name)
        print(f"已设置中文字体: {font_name}")
    else:
        # 如果找不到字体文件，尝试使用默认支持中文的字体
        try:
            plt.rcParams['font.family'] = ['Heiti TC', 'STHeiti', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                                           'sans-serif']
            print("使用备选中文字体")
        except:
            print("警告: 找不到合适的中文字体，可能无法正确显示中文")


# 调用设置中文字体函数
set_chinese_font()

# 设置其他绘图参数
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 设置随机种子以确保结果可复现
tf.random.set_seed(42)
np.random.seed(42)

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# 数据预处理
def preprocess_data(X, y):
    # 归一化像素值到0-1范围
    X = X.astype('float32') / 255.0
    # 添加通道维度 (28, 28) -> (28, 28, 1)
    X = np.expand_dims(X, axis=-1)
    # 将标签转换为one-hot编码
    y = tf.keras.utils.to_categorical(y, 10)
    return X, y


X_train, y_train = preprocess_data(X_train, y_train)
X_test, y_test = preprocess_data(X_test, y_test)

# 划分训练集和验证集 (80%训练, 20%验证)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"验证集形状: {X_val.shape}, {y_val.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")


# 构建基准CNN模型
def build_baseline_model():
    model = models.Sequential([
        # 第一卷积层
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        # 第二卷积层
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # 展平层
        layers.Flatten(),

        # 全连接层
        layers.Dense(128, activation='relu'),

        # 输出层
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 构建改进的CNN模型（添加批量归一化和Dropout）
def build_improved_model():
    model = models.Sequential([
        # 第一卷积层 + 批量归一化
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 第二卷积层 + 批量归一化
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # 展平层
        layers.Flatten(),

        # 全连接层 + Dropout
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # 输出层
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# 训练模型并记录历史
def train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=64, model_name='model'):
    print(f"训练 {model_name}...")
    start_time = time.time()

    # 添加EarlyStopping回调以防止过拟合
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"{model_name} 训练完成! 耗时: {training_time:.2f}秒")

    return history


# 创建模型
baseline_model = build_baseline_model()
improved_model = build_improved_model()

# 训练基准模型
baseline_history = train_model(baseline_model, X_train, y_train, X_val, y_val, model_name='基准模型')

# 训练改进模型
improved_history = train_model(improved_model, X_train, y_train, X_val, y_val, model_name='改进模型')


# 评估模型
def evaluate_model(model, X_test, y_test, model_name):
    print(f"评估 {model_name}...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{model_name} 测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}")
    return test_acc, test_loss


# 评估两个模型
baseline_acc, baseline_loss = evaluate_model(baseline_model, X_test, y_test, '基准模型')
improved_acc, improved_loss = evaluate_model(improved_model, X_test, y_test, '改进模型')


# 绘制训练历史
# 绘制训练历史（修改后）
def plot_history(histories, model_names):
    plt.figure(figsize=(15, 10))

    # 获取最大迭代次数
    max_epochs = max(len(hist.history['accuracy']) for hist in histories)

    # 准确率曲线
    plt.subplot(2, 2, 1)
    for history, name in zip(histories, model_names):
        epochs = range(1, len(history.history['accuracy']) + 1)
        plt.plot(epochs, history.history['accuracy'], label=f'{name}训练集')
        plt.plot(epochs, history.history['val_accuracy'], '--', label=f'{name}验证集')
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('迭代次数')
    plt.xticks(np.arange(1, max_epochs + 1, step=1))  # 设置x轴间隔为1
    plt.legend()
    plt.grid(True)

    # 损失曲线
    plt.subplot(2, 2, 2)
    for history, name in zip(histories, model_names):
        epochs = range(1, len(history.history['loss']) + 1)
        plt.plot(epochs, history.history['loss'], label=f'{name}训练集')
        plt.plot(epochs, history.history['val_loss'], '--', label=f'{name}验证集')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('迭代次数')
    plt.xticks(np.arange(1, max_epochs + 1, step=1))  # 设置x轴间隔为1
    plt.legend()
    plt.grid(True)

    # 测试集性能比较
    plt.subplot(2, 2, 3)
    models = ['基准模型', '改进模型']
    accuracies = [baseline_acc, improved_acc]
    losses = [baseline_loss, improved_loss]

    plt.bar(models, accuracies, color=['skyblue', 'lightgreen'])
    plt.title('测试集准确率比较')
    plt.ylabel('准确率')
    plt.ylim(0.95, 1.0)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.003, f'{v:.4f}', ha='center')

    plt.subplot(2, 2, 4)
    plt.bar(models, losses, color=['skyblue', 'lightgreen'])
    plt.title('测试集损失比较')
    plt.ylabel('损失')
    plt.ylim(0, 0.1)
    for i, v in enumerate(losses):
        plt.text(i, v + 0.003, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# 绘制比较图
plot_history([baseline_history, improved_history], ['基准', '改进'])


# 可视化预测结果
def visualize_predictions(model, X_test, y_test, model_name, num_samples=12):
    # 随机选择样本
    indices = np.random.choice(range(len(X_test)), num_samples)
    sample_images = X_test[indices]
    sample_labels = np.argmax(y_test[indices], axis=1)

    # 预测
    predictions = model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    # 绘制结果
    plt.figure(figsize=(14, 8))
    for i, idx in enumerate(indices):
        plt.subplot(3, 4, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap='gray')

        # 标记预测结果（绿色正确，红色错误）
        color = 'green' if predicted_labels[i] == sample_labels[i] else 'red'
        plt.title(f"预测: {predicted_labels[i]}\n真实: {sample_labels[i]}", color=color)
        plt.axis('off')

    plt.suptitle(f'{model_name}预测结果示例', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{model_name}_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


# 可视化两个模型的预测结果
visualize_predictions(baseline_model, X_test, y_test, '基准模型')
visualize_predictions(improved_model, X_test, y_test, '改进模型')

