from transformers import AutoModel, AutoProcessor
from PIL import Image
import torch
import warnings

# Suppress the warning
warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary")

# Your code here...


model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)

image_path='C:/Users/Admin/Downloads/teen-photo-8466399_1280.jpg'

prompt = "description of image"

image = Image.open(image_path)
inputs = processor(text=[prompt], images=[image], return_tensors="pt")
with torch.inference_mode():
     output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=256,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id
    )

prompt_len = inputs["input_ids"].shape[1]
decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
print(decoded_text)