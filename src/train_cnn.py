import sys
print("当前脚本使用的Python解释器路径:", sys.executable)
from faknow.run import run

run(
    model='textcnn',
    train_path='train.json',
    val_path='val.json',
    test_path='test.json',
    text_field='text',
    label_field='label',
    max_len=128,           # 序列最大长度（根据数据平均长度可调）
    embedding_dim=128,     # 词向量维度
    num_filters=100,       # 卷积核数量
    filter_sizes='3,4,5',  # 卷积核尺寸
    num_classes=2,
    batch_size=64,
    epochs=20,
    lr=0.001,
    device='cuda'          # 使用您的 RTX 4060
)