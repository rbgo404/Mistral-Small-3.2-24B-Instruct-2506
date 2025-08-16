import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from pydantic import BaseModel, Field
from typing import Optional
import inferless

from datetime import datetime, timedelta
import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download
from transformers import Mistral3ForConditionalGeneration




@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Give me 5 non-formal ways to say 'See you later' in French.")
    image_url: Optional[str] = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"
    system_prompt: str = Field(default="You are a conversational agent that always answers straight to the point.")
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.1)
    repetition_penalty: float = Field(default=1.18)
    top_k: int = Field(default=40)
    max_tokens: int= Field(default=100)
    do_sample: bool = Field(default=False)

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Test output")

class InferlessPythonModel:
    def load_system_prompt(self,repo_id: str, filename: str) -> str:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        with open(file_path, "r") as file:
            system_prompt = file.read()
        today = datetime.today().strftime("%Y-%m-%d")
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        model_name = repo_id.split("/")[-1]
        return system_prompt.format(name=model_name, today=today, yesterday=yesterday)

    def initialize(self):
        model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        self.SYSTEM_PROMPT = self.load_system_prompt(model_id, "SYSTEM_PROMPT.txt")       
        self.tokenizer = MistralTokenizer.from_hf_hub(model_id)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    def infer(self, inputs: RequestObjects) -> ResponseObjects:       
        user_message = [
            {
                "type": "text",
                "text": inputs.prompt,
            }]
        if inputs.image_url:
            print("inputs.image_url: ",inputs.image_url,flush=True)
            user_message.append({"type": "image_url", "image_url": {"url": inputs.image_url}})
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_message,
            },
        ]
        tokenized = self.tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
        
        input_ids = torch.tensor([tokenized.tokens])
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.tensor(tokenized.images[0], dtype=torch.bfloat16).unsqueeze(0)
        image_sizes = torch.tensor([pixel_values.shape[-2:]])
        
        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_new_tokens=1000,
        )[0]
        
        decoded_output = self.tokenizer.decode(output[len(tokenized.tokens) :])

        return ResponseObjects(generated_text=decoded_output)

    def finalize(self):
        self.model = None
