import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
from EvdienceRAG import DetectionResult, RumorRAGPipeline
from sth import TextCNN
import torch
import argparse
import json
from collections import Counter

CNNPATH = "./best_textcnn.pt"

def model_init():
    
    pipline = RumorRAGPipeline()
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
    with open('./data/train.json', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/val.json', encoding='utf-8') as f:
        val_data = json.load(f)
    with open('./data/test.json', encoding='utf-8') as f:
        test_data = json.load(f)

    vocab = build_vocab(train_data + val_data + test_data, max_vocab=30000, tokenizer=tokenizer)
    model = TextCNN(vocab=vocab, embedding_dim=100, num_filters=100, filter_sizes=[2,3,4,5], num_classes=2)
    model.load_state_dict(torch.load(CNNPATH, weights_only=False))
    model.eval()
    return model, pipline, vocab

def predict(text_ids, model):
    """
    text_ids: shape (1, max_len) 的 LongTensor
    返回: (prediction_str, confidence)
    """
    model.eval()
    with torch.no_grad():
        logits = model(text_ids)            # (1, num_classes)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
        prediction = "false" if pred_idx.item() == 1 else "true"
    return prediction, conf.item()

def explain(text_str, text_ids, model, pipline: RumorRAGPipeline, confidence=None):
    '''
    text_str: 原始文本字符串（给 LLM 用）
    text_ids: shape (1, max_len) 的 LongTensor（给模型用）
    confidence: 若为 None 则用模型自己的置信度
    '''
    prediction, model_conf = predict(text_ids, model)
    if confidence is None:
        confidence = model_conf
    pipline_input = DetectionResult(text=text_str, prediction=prediction, confidence=confidence)
    pipline_output = pipline.explain(pipline_input)
    return pipline_output

def input_handler(input_form, file_path=None):
    inputs = []
    if input_form == "console" and file_path == None:
        # handle the input from the console
        print("Enter text lines (type 'exit' to finish):")
        while True:
            line = input("> ")
            if line.strip().lower() == "exit":
                break
            inputs.append(line)
        return inputs
    
    if input_form == "json" and file_path is not None:
        # handle the input from json file
        # json file should like :
        # ["your first text", "your second text"...]
        import json
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                inputs = [str(item) for item in data]  # 转为字符串列表
            else:
                print("ERROR: JSON file must contain a list of strings.")
        except Exception as e:
            print(f"ERROR reading JSON file: {e}")
        return inputs

    if input_form == "txt" and file_path is not None:
        # handle the input from txt file
        # txt file should like : 
        # your first text
        # your second text
        # ...
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 读取所有行，保留空行，去掉末尾的换行符
                inputs = [line.rstrip('\n') for line in f]
        except Exception as e:
            print(f"ERROR reading text file: {e}")
        return inputs
    
    print("ERROR")
    return inputs

def process_inputs(inputs, vocab, max_len=128, tokenizer=None):
    """
    将输入的字符串列表转换为模型可用的 Tensor 矩阵
    
    参数:
    - inputs: 字符串列表，例如 ["Hello world", "Good morning"]
    - vocab: 训练时构建好的词表 dict
    - max_len: 序列最大长度
    - tokenizer: 分词器函数（英文默认 None 用 split，中文可用 jieba.cut）
    
    返回:
    - 一个形状为 (len(inputs), max_len) 的 torch.LongTensor
    """
    processed_seqs = []
    
    for text in inputs:
        # 1. 分词
        if tokenizer:
            tokens = tokenizer(text)
        else:
            tokens = text.split()  # 默认英文空格分词
            
        # 2. 截断并映射为索引 (找不到的词用 <UNK> 的索引，默认 1)
        unk_idx = vocab.get('<UNK>', 1)
        seq = [vocab.get(t, unk_idx) for t in tokens[:max_len]]
        
        # 3. 填充 (用 <PAD> 的索引，默认 0)
        if len(seq) < max_len:
            pad_idx = vocab.get('<PAD>', 0)
            seq += [pad_idx] * (max_len - len(seq))
            
        processed_seqs.append(seq)
        
    # 4. 转换为 PyTorch 张量 (Long 类型，对应模型 embedding 层的输入要求)
    return torch.tensor(processed_seqs, dtype=torch.long)

def print_result(dic):
    """
    打印多个文本的解释结果。
    
    Args:
        dic: dict, key 为输入文本，value 为 explain() 返回的结果对象，
             该对象应包含 "explanation_raw" 和 "evidence" 等字段。
    """
    for idx, (input_text, exp) in enumerate(dic.items(), 1):
        print(f"\n{'='*60}")
        print(f"INPUT {idx}: {input_text[:100]}{'...' if len(input_text) > 100 else ''}")
        print(f"{'='*60}")
        
        # 打印原始解释文本
        print(exp.get("explanation_raw", "No explanation_raw found."))
        print(f"{'='*60}")
        
        # 打印证据来源信息
        evidence_list = exp.get("evidence", [])
        if evidence_list:
            # 提取所有证据来源数据库（假设每个 evidence 有 source_db 属性）
            sources = list(set(e.source_db for e in evidence_list if hasattr(e, 'source_db')))
            print(f"\nEvidence sources used: {sources}")
            
            # 打印前5个证据的详细来源（假设每个 evidence 有 doc.metadata['source']）
            top_sources = []
            for e in evidence_list[:5]:
                if hasattr(e, 'doc') and hasattr(e.doc, 'metadata'):
                    source = e.doc.metadata.get('source', '?')
                else:
                    source = '?'
                top_sources.append(source)
            print(f"Top evidence sources detail: {top_sources}")
        else:
            print("\nNo evidence available.")
        
        print()  # 额外空行分隔不同输入


def main():
    
    parser = argparse.ArgumentParser(description="TextCNN 推理程序")
    parser.add_argument(
        "-i", "--input_type",
        choices=["console", "txt", "json"],
        required=True,
        help="输入类型：console（控制台交互）、txt（文本文件）、json（JSON文件）"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="当 input_type 为 txt 或 json 时，指定文件路径"
    )
    parser.add_argument(
        "-s", "--silent",
        action="store_true",
        default=False,
        help="静默输出，不使用LLM解释"
    )

    args = parser.parse_args()

    # 输入形式校验：如果是 console 则不能有 file；如果是 txt/json 则必须有 file
    if args.input_type == "console":
        if args.file is not None:
            parser.error("--input_type console 不需要 --file 参数")
        file_path = None
    else:  # txt 或 json
        if args.file is None:
            parser.error(f"--input_type {args.input_type} 必须提供 --file 参数")
        file_path = args.file

    # 调用之前的 input_handler 函数
    inputs = input_handler(args.input_type, file_path)
    
    if not inputs:
        print("没有获得任何输入，程序退出。")
        return
    model, pipline, vocab = model_init()
    input_ids = process_inputs(inputs, vocab, max_len=128, tokenizer=None)
    if not args.silent:
        dic = {}
        for text_str, ids_row in zip(inputs, input_ids):
            ids_batch = ids_row.unsqueeze(0)       # (128,) -> (1, 128)
            exp = explain(text_str, ids_batch, model, pipline)
            dic[text_str] = exp
        print_result(dic)
    else:
        for i, (text_str, ids_row) in enumerate(zip(inputs, input_ids)):
            ids_batch = ids_row.unsqueeze(0)
            prediction, model_conf = predict(ids_batch, model)
            print(f"\n[{i}] 处理文本: {text_str[:60]}...")
            print(f"[{i}] CNN 预测: {prediction} (conf={model_conf:.4f})")
if __name__ == "__main__":
    main()