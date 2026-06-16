#说明，该程序用于提取所需数据，并将其转换为规定格式

import json
import os

# 设置根目录
base_dir = "AI_Rumor_Detector"
output_file = "pheme_faknow.json"

data = []

# 遍历所有事件文件夹
for event in os.listdir(base_dir):
    if event.startswith('.') or event.startswith('._'):
        continue
    event_path = os.path.join(base_dir, event)
    if not os.path.isdir(event_path):
        continue

    for category in ['rumours', 'non-rumours']:
        category_path = os.path.join(event_path, category)
        if not os.path.isdir(category_path):
            continue
        label = 1 if category == 'rumours' else 0

        for thread in os.listdir(category_path):
            if thread.startswith('.') or thread.startswith('._'):
                continue
            thread_path = os.path.join(category_path, thread)
            if not os.path.isdir(thread_path):
                continue

            # 查找 source-tweets 或 source-tweet 目录
            src_dir = None
            for possible in ['source-tweets', 'source-tweet']:
                candidate = os.path.join(thread_path, possible)
                if os.path.isdir(candidate):
                    src_dir = candidate
                    break
            if src_dir is None:
                continue

            for file in os.listdir(src_dir):
                if file.startswith('.') or file.startswith('._'):   # 关键过滤
                    continue
                if not file.endswith('.json'):
                    continue
                file_path = os.path.join(src_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        tweet = json.load(f)
                        text = tweet.get('text') or tweet.get('full_text') or ''
                        if text.strip():
                            data.append({"text": text, "label": label})
                except Exception as e:
                    print(f"跳过 {file_path}: {e}")

print(f"成功提取 {len(data)} 条新闻")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"保存至 {output_file}")