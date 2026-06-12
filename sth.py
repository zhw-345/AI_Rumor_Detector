import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
import numpy as np

# ---------------------------- 1. 数据预处理 ----------------------------
class TextDataset(Dataset):
    def __init__(self, data, vocab, max_len=128, tokenizer=None):
        self.X = [self._text_to_seq(item['text'], vocab, max_len, tokenizer) for item in data]
        self.y = [item['label'] for item in data]

    #预处理，将不同长度的句子转换为一个固定长度的整数序列（这才是神经网络能处理的）
    def _text_to_seq(self, text, vocab, max_len, tokenizer):
        if tokenizer:
            tokens = tokenizer(text)
        else:
            tokens = text.split()   # 英文用空格
        seq = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens[:max_len]]
        if len(seq) < max_len:
            seq += [vocab.get('<PAD>', 0)] * (max_len - len(seq))
        return seq

    def __len__(self):
        return len(self.X)

    #根据索引，将文本和标签分别映射为两个张量，这两个张量就是返回值
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

#构建一个有30000字符的映射表，第一个是用于补全长度的占位符，第二个是用于替代表中不存在词汇的替代符，后面29998个字符是出现频率最高的字符
def build_vocab(data, max_vocab=30000, tokenizer=None):
    counter = Counter()
    for item in data:
        text = item['text']
        if tokenizer:
            tokens = tokenizer(text)
        else:
            tokens = text.split()
        counter.update(tokens)
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(max_vocab-2))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

# 如果使用中文，引入 jieba
# import jieba
# tokenizer = lambda x: list(jieba.cut(x))
tokenizer = None   # 英文直接用 split

# 加载数据
with open('train.json', encoding='utf-8') as f:
    train_data = json.load(f)
with open('val.json', encoding='utf-8') as f:
    val_data = json.load(f)
with open('test.json', encoding='utf-8') as f:
    test_data = json.load(f)

vocab = build_vocab(train_data + val_data + test_data, max_vocab=30000, tokenizer=tokenizer)
print(f"词表大小: {len(vocab)}")

# 超参数
#句子最大词数
MAX_LEN = 128
#batch数目
BATCH_SIZE = 64

train_dataset = TextDataset(train_data, vocab, MAX_LEN, tokenizer)
val_dataset = TextDataset(val_data, vocab, MAX_LEN, tokenizer)
test_dataset = TextDataset(test_data, vocab, MAX_LEN, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ---------------------------- 2. TextCNN 模型 ----------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[3,4,5], num_classes=2, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)               # (batch, seq_len, emb_dim)
        emb = emb.unsqueeze(1)                # (batch, 1, seq_len, emb_dim)
        conv_outs = []
        for conv in self.convs:
            conv_out = conv(emb)              # (batch, num_filters, seq_len-k+1, 1)
            conv_out = torch.relu(conv_out.squeeze(3))
            pooled = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)   # (batch, num_filters)
            conv_outs.append(pooled)
        cat = torch.cat(conv_outs, dim=1)      # (batch, len(filter_sizes)*num_filters)
        cat = self.dropout(cat)
        return self.fc(cat)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TextCNN(vocab_size=len(vocab), embedding_dim=128, num_filters=100, filter_sizes=[3,4,5], num_classes=2).to(device)
print(model)

# ---------------------------- 3. 训练与评估 ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(loader):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(loader, return_labels=False):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.extend(pred.argmax(dim=1).cpu().numpy())
            trues.extend(y.cpu().numpy())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    rec = recall_score(trues, preds)
    if return_labels:
        return acc, f1, rec, trues, preds
    return acc, f1, rec

def train():
# 训练
    EPOCHS = 20
    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(train_loader)
        val_acc, val_f1, val_rec = evaluate(val_loader)
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_rec:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_textcnn.pt')
            print("  -> 保存最佳模型")

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load('best_textcnn.pt'))
    test_acc, test_f1, test_rec, true_labels, pred_labels = evaluate(test_loader, return_labels=True)
    print("\n========== 测试集结果 ==========")
    print(f"准确率 (Accuracy):  {test_acc:.4f}")
    print(f"召回率 (Recall):    {test_rec:.4f}")
    print(f"F1 值:             {test_f1:.4f}")
    print("\n混淆矩阵:")
    print(confusion_matrix(true_labels, pred_labels))
    print("\n分类报告:")
    print(classification_report(true_labels, pred_labels, target_names=['非谣言', '谣言']))