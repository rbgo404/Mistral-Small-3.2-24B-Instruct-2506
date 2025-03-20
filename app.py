from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from typing import Optional
import inferless
import torch
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration, BitsAndBytesConfig

@inferless.request
class RequestObjects(BaseModel):
    prompt: str = Field(default="Who are you?")
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
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="cuda",
        )

    def infer(self, request: RequestObjects) -> ResponseObjects:
        inputs = self.tokenizer(request.prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generation = self.model.generate(
                **inputs,
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
