from flask import Flask, render_template, request
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
from openai import OpenAI
import io
import os

# Configure OpenAI API key (replace with your actual key)
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = Flask(__name__)

# Load model and processor only once during initialization
model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)


def process_image_and_generate_captions(image_data, user_preference, social_media, additional_info=None):
    # Open image from bytes data
    image = Image.open(io.BytesIO(image_data))

    prompt = "Please describe the image in 1 sentence."
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
    predicted_caption = processor.batch_decode(output[:, prompt_len:])[0]

    new_caption = generate_new_caption(predicted_caption, user_preference, social_media)
    


def generate_new_caption(predicted_caption, user_preference, social_media):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"I want to come up with a 4 creative caption for the image. The predicted caption is: '{predicted_caption}'. Can you suggest a new, unique and descriptive 4 caption based on this?"},
        {"role": "assistant", "content": f"Sure, I can help with that! What style or theme would you like for the new caption?"},
        {"role": "user", "content": f"Provide me with a {user_preference} and 4 captions for {social_media} along with 2 appropriate hashtags. Also add appropriate emojis wherever needed."}
    ]
    

    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=200,  # Adjust the max_tokens based on the expected response length
            stop=None,
            temperature=0.8
        )
        new_caption = response.choices[0].message.content.strip()
        return new_caption

    except OpenAI.Error as e:
        print(f"Error calling OpenAI API: {e}")  # Add print statement for debugging
        return None  # Indicate new caption generation failure

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image_file = request.files["image"]
        user_preference = request.form["user_preference"]
        social_media = request.form["social_media"]
       

        # Read image data directly from the file object
        image_data = image_file.read()

        # Process the uploaded image and generate captions
        predicted_caption, new_caption = process_image_and_generate_captions(image_data, user_preference, social_media)
       
        return render_template("hetviuno.html", predicted_caption=predicted_caption, new_caption=new_caption)

    return render_template("hetviuno.html")

if __name__ == "__main__":
    app.run()

