from llm_config import call_gpt 

class QAAgent:
    def __init__(self, model, headers):
        self.model = model
        self.headers = headers
    
    def ask_gpt(self, messages, idx):
        success, idx, message, answer = call_gpt(
            self.model, messages, idx, self.headers
        )
            
        return success, idx, message, answer

