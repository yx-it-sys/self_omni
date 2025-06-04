from search_api import fine_search
import prompt as prompt
import requests
import base64
import json
from io import BytesIO
from PIL import Image

class ConversationManager:
    def __init__(self, qa_agent, dataset_name, save_path):
        self.qa_agent = qa_agent
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.conversation_num = 0
        self.total_image_quota = 9

    def manage_conversation(self, input_question, image_url, idx):
        self.conversation_num = 0
        # Gemini2.0的要求，编码图像
        image_content_bytes = None
        print(f"正在下载图片: {image_url}")
        image_response = requests.get(image_url, stream=True)
        image_response.raise_for_status() # 检查HTTP状态码
        image_content_bytes = image_response.content
        print("图片下载成功。")
        encoded_image_string = None
        if image_content_bytes:
            encoded_image_string = base64.b64encode(image_content_bytes).decode('utf-8')
            print("图片已编码为 Base64 字符串。")
        
        # Gemini API请求格式
        messages = [
            {
                "role": "user",
                "parts": [
                    {"text":prompt.sys_prompt_1.format(input_question)},
                    {"inlineData": {
                        "mimeType": "image/jpeg", # 使用识别到的 MIME 类型
                        "data": encoded_image_string
                    }}
                ]
            }
        ]

        current_message = messages

        # message来自调用Gemini，是：{"role":user, "content":answer}
        success, idx, message, answer = self.qa_agent.ask_gpt(messages, idx)
        print("first response:", answer)

        while self.conversation_num < 5:
            if "Final Answer" in answer:
                tmp_d = {"role": "assistant"}
                tmp_d.update(message)
                # 将LLM生成的内容加到message中，此时current_message有"role":"user","parts","role":"assistant","role": "model", "content": answer
                current_message.append(tmp_d)
                print(answer)
                print("-------")
                print(answer.split("Final Answer: ")[-1])
                return answer.split("Final Answer: ")[-1], current_message
            
            if any(phrase in answer for phrase in ["Image Retrieval with Input Image", "Text Retrieval", "Image Retrieval with Text Query"]):
                tmp_d = {"role": "assistant"}
                tmp_d.update(message)
                #将模型的回复加入上下文
                current_message.append(tmp_d)
                #执行检索操作
                sub_question = answer.split('<Sub-Question>\n')[-1].split('\n')[0]
                search_images, search_text = self.handle_retrieval(answer, image_url, idx)
                
                contents = self.prepare_contents(search_images,messages,sub_question,idx, search_text, image_url)
                
                #将检索到的信息加入上下文
                current_message.append({"role": "user", "parts": [{"text":contents}]})
                # print(current_message[0]["parts"][0]["text"])
                # print(current_message[1])
                # 上下文传入Prompt
                success, idx, message, answer = self.qa_agent.ask_gpt(current_message, idx)
                print("conversation step:", self.conversation_num, answer)
                if not success:
                    print("Request failed.")
                    break
            print(self.conversation_num)
            self.conversation_num += 1
        print(answer)
        print(self.conversation_num)
        print("OVER!")
        return answer, current_message

    def handle_retrieval(self, answer, image_url, idx):
        if 'Image Retrieval with Input Image' in answer:
            return fine_search(image_url, 'img_search_img', self.save_path, self.dataset_name, idx, self.conversation_num)
        elif 'Text Retrieval' in answer:
            query = self.extract_query(answer, 'Text Retrieval')
            return fine_search(query, 'text_search_text', self.save_path, self.dataset_name, idx, self.conversation_num)
        elif 'Image Retrieval with Text Query' in answer:
            query = self.extract_query(answer, 'Image Retrieval with Text Query')
            return fine_search(query, 'text_search_img', self.save_path, self.dataset_name, idx, self.conversation_num)

    def extract_query(self, answer, retrieval_type):
        return answer.split(retrieval_type)[-1].replace(':', '').replace('"', '').replace('>', '')
    
    def prepare_contents(self, search_images,messages,sub_question,idx,search_text, image_url):
        if len(search_images) > 0:
            # 该分支表明，如果检索到了图片和相关信息（知识图谱），直接返回检索内容加入Prompt上下文中。
            #断言失败的时候显示(search_text)
            #assert len(search_images) == len(search_text), (search_text)
            contents = [{"text": "Contents of retrieved images: "}]
            use_imgs_num = min(5, self.total_image_quota)
            self.total_image_quota -= use_imgs_num
            for img, txt in zip(search_images[:use_imgs_num], search_text[:use_imgs_num]):
                image_url = img[0]
                image_response = requests.get(image_url, stream=True)
                image_response.raise_for_status() # 检查HTTP状态码
                image_content_bytes = image_response.content
                encoded_image_string = None
                if image_content_bytes:
                    encoded_image_string = base64.b64encode(image_content_bytes).decode('utf-8')
                contents.extend([
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg", # 使用识别到的 MIME 类型
                            "data": encoded_image_string
                        },
                    },
                    {
                        "text": "Description: "+txt
                    }
                ])
        else:
            # 该分支表明，如果没有检索出图片的话，只有检索得到的文本，那么将文本和问题图片一并加入Prompt中，
            # 还把子问题也加入到Prompt中，让LLM根据检索得到的文本内容生成外部知识，
            # 将LLM生成的外部知识作为上下文（反馈）传到Planning Agent中
            contents = [
                {
                    "text": "Below are related documents retrieved, which may be helpful for answering questions later on:"
                }
            ]
            for txt in search_text:
                contents.append({
                    "text": txt
                })
            
            contents.append({
                "text": "\nInput Image:"
            })

            image_response = requests.get(image_url, stream=True)
            image_response.raise_for_status() # 检查HTTP状态码
            image_content_bytes = image_response.content
            encoded_image_string = None
            if image_content_bytes:
                encoded_image_string = base64.b64encode(image_content_bytes).decode('utf-8')

            contents.append({
                        "inlineData": {
                            "mimeType": "image/jpeg", # 使用识别到的 MIME 类型
                            "data": encoded_image_string
                        },
                    },
            )
            contents.append({
                "text": sub_question + " Please Generate its answer:"
            })
            sub_messages = [
                {
                    "role": "user",
                    "parts": contents
                }
            ]
           
            success=True
            answer=self.qa_agent.ask_gpt(sub_messages,idx)
            contents = [{"text": "Contents of retrieved documents: "}]
            if success:
                # 将检索结果用于对子问题进行解答，转入Planning Agent中，Agent会根据回答继续迭代或终止。
                contents.extend([{"text": answer}])
            else:
                for txt in search_text:
                    contents.extend([
                        {
                            "text": txt
                        }
                        ])
        return contents

