import json
from sklearn.model_selection import train_test_split

with open('pheme_faknow.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)

# 先分出训练集 (70%) 和临时集 (30%)
train_data, temp_data = train_test_split(all_data, test_size=0.3, random_state=42)
# 再平分临时集为验证集 (15%) 和测试集 (15%)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 保存
for name, data_split in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
    with open(f'{name}.json', 'w', encoding='utf-8') as f:
        json.dump(data_split, f, ensure_ascii=False, indent=2)
    print(f"{name}.json: {len(data_split)} 条")