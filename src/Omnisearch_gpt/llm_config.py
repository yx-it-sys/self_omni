import requests
import time
from google import genai
workers=4
skip_lines = 0

LIMIT = 1000000000000 
retry_attempt = 10


def call_gpt(model, messages, idx, headers):
    # 准备请求数据，包括模型、对话信息等参数
    data = {
        "model": model,
        "messages": messages,
        "n": 1,  # 回答数量
        "max_tokens": 4096
    }
    
    answer = None
    while answer is None:
        try:

            r = requests.post(
                #'https://api.chatanywhere.tech/v1/chat/completions', 
                'https://api.openai.com/v1/chat/completions',
                json=data,
                headers=headers
            )
            resp = r.json()

            if r.status_code != 200:
                print('请求失败，重试中！')
                print(resp)
                continue

            if 'choices' in resp and resp['choices'][0].get('finish_reason') in ['content_filter', 'ResponsibleAIPolicyViolation']:
                print('内容不符合策略要求，返回空结果')
                return (False, idx, "", "", 0, 0, 0)
            message = resp['choices'][0]['message']
            answer = message['content']

            return (True, idx, message, answer)

        except Exception as e:
            print(e)
            print('发生异常，重试中！')
            time.sleep(1)  # 等待一段时间再重试
            continue

def call_gemini(model, messages, idx, api_key):
    # genai.configure(api_key=api_key)
    contents = []
    for msg in messages:
        contents.append(
            {
                "role":msg["role"],
                "parts":msg["parts"]
            }
        )
        answer = None
        retry_count = 0
        max_tries = 10

        while answer is None and retry_count < max_tries:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model = model,
                    contents = contents,
                )
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    print('内容不符合策略要求，返回空结果')
                    # For Gemini, the 'message' equivalent is the content within the response.
                    # Returning an empty message and answer for blocked content
                    return (False, idx, {"role": "model", "content": ""}, "")

                # Access the generated text
                answer = response.text
                
                # The 'message' equivalent would be a dictionary with 'role' and 'content'
                # representing the model's reply.
                message = {"role": "model", "content": answer}

                return (True, idx, message, answer)

            except Exception as e:
                print(f'发生异常，重试中！错误: {e}')
                retry_count += 1
                time.sleep(1)  # Wait for a bit before retrying
            continue

    # If all retries fail
    print('达到最大重试次数，无法获取回复。')
    return (False, idx, {"role": "model", "content": ""}, "")