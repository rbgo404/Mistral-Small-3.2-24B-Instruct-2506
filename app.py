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
    max_tokens: int = Field(default=100)
    do_sample: bool = Field(default=False)

@inferless.response
class ResponseObjects(BaseModel):
    generated_text: str = Field(default="Test output")

class InferlessPythonModel:
    def load_system_prompt(self, repo_id: str, filename: str) -> str:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        with open(file_path, "r") as file:
            system_prompt = file.read()
        today = datetime.today().strftime("%Y-%m-%d")
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        model_name = repo_id.split("/")[-1]
        return system_prompt.format(name=model_name, today=today, yesterday=yesterday)

    def initialize(self):
        model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.SYSTEM_PROMPT = self.load_system_prompt(model_id, "SYSTEM_PROMPT.txt")       
        self.tokenizer = MistralTokenizer.from_hf_hub(model_id)
        
        # Load model with explicit device mapping
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Changed from "cuda" to "auto"
        )
        
        # Ensure model is on the correct device
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)

    def infer(self, inputs: RequestObjects) -> ResponseObjects:       
        user_message = [
            {
                "type": "text",
                "text": inputs.prompt,
            }]
        if inputs.image_url:
            print("inputs.image_url: ", inputs.image_url, flush=True)
            user_message.append({"type": "image_url", "image_url": {"url": inputs.image_url}})
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_message,
            },
        ]
        
        tokenized = self.tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
        
        # Ensure all tensors are on the same device
        input_ids = torch.tensor([tokenized.tokens], device=self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # Handle images if present
        if tokenized.images:
            pixel_values = torch.tensor(tokenized.images[0], dtype=torch.bfloat16, device=self.device).unsqueeze(0)
            image_sizes = torch.tensor([pixel_values.shape[-2:]], device=self.device)
        else:
            pixel_values = None
            image_sizes = None
        
        # Generate with proper device placement
        with torch.no_grad():  # Save memory
            if pixel_values is not None:
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    max_new_tokens=inputs.max_tokens,
                    temperature=inputs.temperature if inputs.do_sample else None,
                    top_p=inputs.top_p if inputs.do_sample else None,
                    top_k=inputs.top_k if inputs.do_sample else None,
                    do_sample=inputs.do_sample,
                    repetition_penalty=inputs.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else None
                )[0]
            else:
                # Handle text-only generation
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=inputs.max_tokens,
                    temperature=inputs.temperature if inputs.do_sample else None,
                    top_p=inputs.top_p if inputs.do_sample else None,
                    top_k=inputs.top_k if inputs.do_sample else None,
                    do_sample=inputs.do_sample,
                    repetition_penalty=inputs.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else None
                )[0]
        
        # Decode the output
        decoded_output = self.tokenizer.decode(output[len(tokenized.tokens):])

        return ResponseObjects(generated_text=decoded_output)

    def finalize(self):
        torch.cuda.empty_cache()
        self.model = None
