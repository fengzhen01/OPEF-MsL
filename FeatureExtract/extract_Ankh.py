import os
import torch
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel


def read_data_file_trip(filename):
    with open(filename) as f:
        data = f.readlines()

    results = []
    block = len(data) // 2
    for index in range(block):
        item1 = data[index * 2 + 0].split()
        name = item1[0].strip()
        seq = item1[1].strip()
        results.append([name, seq])
    return results


def extract_ankh_embeddings(file, dest_folder, model, tokenizer, device):
    data = read_data_file_trip(file)
    for idx, (name, seq) in enumerate(data, 1):
        print(f"[{idx}/{len(data)}] Processing: {name}")

        out_file = os.path.join(dest_folder, name + '.data')
        if os.path.exists(out_file):
            print(f"{name} already exists, skipping.")
            continue

        seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")

        inputs = tokenizer(seq, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[0]  # shape: [seq_len, hidden_dim]

        seq_len = (attention_mask[0] == 1).sum().item()
        embedding = embedding[1:seq_len - 1].cpu().numpy()

        np.savetxt(out_file, embedding, delimiter=',', fmt='%s')
        torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # ankh_model_name = "facebook/ankh-large"  # 或 "facebook/ankh-base"
    # print("加载 Ankh 模型和 tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(ankh_model_name)
    # model = AutoModel.from_pretrained(ankh_model_name).to(device)
    # model.eval()

    local_model_path = "D:/fengzhen/1NucGMTL-main/Ankh/Ankh_Large/"
    print("从本地加载 Ankh-Large 模型...")
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = T5EncoderModel.from_pretrained(local_model_path).to(device)
    model.eval()

    # input_files = [
    #     'D:/fengzhen/1NucGMTL-main/DataSet/UniProtSMB/SMB2_Train.txt',
    #     'D:/fengzhen/1NucGMTL-main/DataSet/UniProtSMB/SMB2_Test.txt'
    # ]
    input_files = [
        'D:/fengzhen/1NucGMTL-main/DataSet/ATP/ATP549.txt',
        'D:/fengzhen/1NucGMTL-main/DataSet/ATP/ATP41.txt'
    ]
    output_dir = 'D:/fengzhen/2embedding/Ankh_embedding_ATP549+41/'
    os.makedirs(output_dir, exist_ok=True)

    for file in input_files:
        print(f"\n=== 开始处理文件: {file} ===")
        extract_ankh_embeddings(file, output_dir, model, tokenizer, device)

    print("\n==== 全部完成 ====")
