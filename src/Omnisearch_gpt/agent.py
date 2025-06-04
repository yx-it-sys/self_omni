from llm_config import call_gemini


class QAAgent:
    def __init__(self, model, headers):
        self.model = model
        self.headers = headers
    def ask_gpt(self, messages, idx):
        success, idx, message, answer = call_gemini(
            self.model, messages, idx, self.headers["gemini-api-key"]
        )
            
        return success, idx, message, answer

