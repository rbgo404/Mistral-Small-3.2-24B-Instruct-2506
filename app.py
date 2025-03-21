from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from typing import Optional
import inferless
import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration, BitsAndBytesConfig


@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Give me 5 non-formal ways to say 'See you later' in French.")
    system_prompt: Optional[str] = "You are a conversational agent that always answers straight to the point."
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    repetition_penalty: Optional[float] = 1.18
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 100
    do_sample: Optional[bool] = False

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Test output")

class InferlessPythonModel:
    def initialize(self):
        model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        snapshot_download(repo_id=model_id)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.chat_template = "{%- set today = strftime_now(\"%Y-%m-%d\") %}\n{%- set default_system_message = \"You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYour knowledge base was last updated on 2023-10-01. The current date is \" + today + \".\\n\\nWhen you're not sure about some information, you say that you don't have the information and don't make up anything.\\nIf the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \\\"What are some good restaurants around me?\\\" => \\\"Where are you?\\\" or \\\"When is the next flight to Tokyo\\\" => \\\"Where do you travel from?\\\")\" %}\n\n{{- bos_token }}\n\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = default_system_message %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}\n\n{%- for message in loop_messages %}\n    {%- if message['role'] == 'user' %}\n\t    {%- if message['content'] is string %}\n            {{- '[INST]' + message['content'] + '[/INST]' }}\n\t    {%- else %}\n\t\t    {{- '[INST]' }}\n\t\t    {%- for block in message['content'] %}\n\t\t\t    {%- if block['type'] == 'text' %}\n\t\t\t\t    {{- block['text'] }}\n\t\t\t    {%- elif block['type'] == 'image' or block['type'] == 'image_url' %}\n\t\t\t\t    {{- '[IMG]' }}\n\t\t\t\t{%- else %}\n\t\t\t\t    {{- raise_exception('Only text and image blocks are supported in message content!') }}\n\t\t\t\t{%- endif %}\n\t\t\t{%- endfor %}\n\t\t    {{- '[/INST]' }}\n\t\t{%- endif %}\n    {%- elif message['role'] == 'system' %}\n        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- else %}\n        {{- raise_exception('Only user, system and assistant roles are supported!') }}\n    {%- endif %}\n{%- endfor %}"
        
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="cuda",
        )

    def infer(self, request: RequestObjects) -> ResponseObjects:
        messages = [
                        {
                            "role": "system",
                            "content": request.system_prompt
                        },
                        {
                            "role": "user",
                            "content": request.prompt
                        },
                    ]
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generation = self.model.generate(
                tokenized_chat,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
                repetition_penalty=request.repetition_penalty,
            )
            generated_text = self.tokenizer.decode(generation[0], skip_special_tokens=True)

        return ResponseObjects(generated_text=generated_text)

    def finalize(self):
        self.model = None
