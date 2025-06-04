import threading
from concurrent.futures import ThreadPoolExecutor
import os
import json
import asyncio
import argparse
from agent import QAAgent
from conversation_manager import ConversationManager

model = "gemini-2.0-flash"
GPT_API_KEY = "AIzaSyABhTC9jRFnkab2GVPxinAHH-3zGlNmvMA"

# headers = {
#     "Authorization": f"Bearer {GPT_API_KEY}"
# }

headers = {
    "Content-Type": "application/json",
    "gemini-api-key": GPT_API_KEY  # Gemini API 的认证方式
}

meta_save_path = './vfreshqa_datasets_v2/'#保存地址


write_lock = threading.Lock()

def safe_write(file_path, data):
    with write_lock:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def process_item(item, conversation_manager, meta_save_path, dataset_name):
    input_question = item['question']
    idx = item['question_id']
    image_url = item['image_url']
    
    answer, current_message = conversation_manager.manage_conversation(
        input_question=input_question, image_url=image_url, idx=idx
    )
    
    # 将结果保存到 item 中
    item['prediction'] = answer
    # 保存结果
    output_path = os.path.join(meta_save_path, dataset_name, "output_from_gpt4v.jsonl")
    safe_write(output_path, item)

def main(test_dataset,dataset_name, meta_save_path):
    
    num_threads = 1

    qa_agent = QAAgent(model=model, headers=headers)

    with open(test_dataset, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f.readlines()]

    output_path = os.path.join(meta_save_path, dataset_name, "output_from_gpt4v.jsonl")
    if os.path.exists(output_path):
        with open(output_path, "r", encoding='utf-8') as fin:
            print(output_path)
            done_id = [json.loads(data)['question_id'] for data in fin.readlines()]
            datas = [data for data in datas if data['question_id'] not in done_id]

    save_path = os.path.join(meta_save_path, dataset_name, "search_images_gpt4v")
    os.makedirs(save_path, exist_ok=True)

    conversation_manager = ConversationManager(qa_agent=qa_agent, dataset_name=dataset_name, save_path=save_path)
    for item in datas:
        process_item(item, conversation_manager, meta_save_path, dataset_name)

    

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="运行指定的数据集")
    parser.add_argument("--test_dataset", type=str, required=True, help="数据集路径")
    parser.add_argument("--dataset_name", type=str, required=True, help="数据集名称")
    parser.add_argument("--meta_save_path", type=str, required=True, help="存储路径")
    

    args = parser.parse_args()

    # 调用 main 函数并传递解析后的参数
    main(args.test_dataset,args.dataset_name, args.meta_save_path)
