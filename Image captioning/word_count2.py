from flask import Flask, render_template, request
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
from openai import OpenAI
import os
import time

# Configure OpenAI API key (replace with your actual key)
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = Flask(__name__)

# Load model and processor only once during initialization
model = AutoModel.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("unum-cloud/uform-gen2-qwen-500m", trust_remote_code=True)




def process_image_and_generate_captions(image_file, user_preference, social_media, additional_information=None, word_count=None):
    image = Image.open(image_file)
    start_time = time.time() 
    
    #prompt = "Please describe the image in detail"
    prompt = "What story does this image tell?"
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

    if word_count and word_count <= 25:
        num_captions = 3
    else:
        num_captions = 1

    new_captions = []
    for i in range(num_captions):
        new_caption = generate_new_caption(predicted_caption, user_preference, social_media, additional_information, word_count)
        new_captions.append(new_caption)
        
    end_time = time.time()  # Record the end time
    time_taken = end_time - start_time  # Calculate the time taken in seconds
    print(f"Time taken : {time_taken:.2f} seconds")     

    if num_captions == 1:
        return predicted_caption, new_captions[0]
    else:
        return predicted_caption, new_captions[0], new_captions[1], new_captions[2]




def validate_image(image_file):
    """
    Checks if the uploaded image is valid (JPG or PNG under 4MB).

    Args:
        image_file (werkzeug.datastructures.FileStorage): The uploaded image file.

    Returns:
        bool: True if the image is valid, False otherwise.
    """
    if not image_file or image_file.filename == "":
        print("No image uploaded.")
        return False
    
    


    # Check image format
    #allowed_extensions = ["JPEG", "PNG","jpg","png"]
    #if image_file.mimetype not in [f"image/{ext}" for ext in allowed_extensions]:
        print(f"Unsupported image format. Please upload a JPG or PNG image.")
        return False
    img = Image.open(image_file)
    
    if img.format not in ['JPEG', 'PNG']:
        print(f"Unsupported image format. Please upload a JPG or PNG image.")
        return False
    
    
    
    # Check image size
    #file_size = image_file.content_length
    file_size = len(image_file.read())
    if file_size > 4 * 1024 * 1024:  # 4MB limit
        print(f"Image file size exceeds 4MB limit.")
        return False
    
    

    return True


def generate_new_caption(predicted_caption, user_preference, social_media, additional_information=None, word_count=None):
    
    messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"I want to come up with a creative caption for the image. The predicted caption is: '{predicted_caption}',with  some hashtags. "},  # Include "avoid declarative" stop sequence
    {"role": "assistant", "content": "Sure, I can help with that!"},
    {"role": "assistant", "content": "What style or theme would you like for the new caption?"},
    {"role": "user", "content": f"I'd like a caption that's {user_preference} and suitable for {social_media}. Can you suggest a new, unique, and descriptive caption that combines these elements, avoiding declarative sentences and aiming for exactly {word_count} words if possible?"},
    {"role": "assistant", "content": "Sure, I can help with that!"}
]


    # If additional information is provided, incorporate it into the prompt
    if additional_information:
        # Ensure that each sentence of the additional information is treated as a separate message
        additional_info_messages = [{"role": "assistant", "content": f"Additional information about the image: {info}"} for info in additional_information.split('.')]
        messages = additional_info_messages + messages

    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=300,  # Adjust the max_tokens based on the expected response length
            stop=None,
            temperature=0.7
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
        additional_information = request.form.get("additional_information", None)
        word_count = int(request.form.get("word_count")) if request.form.get("word_count") else None

        if not validate_image(image_file):
        
            # Show error message or redirect to error page
            return render_template("error.html", error_message="Invalid image format or size.")

        # Save the uploaded image (assuming validation passed)
        #image_path = "static/uploaded_image.jpg"
        #image_file.save(image_path)

        # Process the uploaded image and generate captions
        predicted_caption, *new_captions = process_image_and_generate_captions(image_file, user_preference, social_media, additional_information, word_count)
        print(f"Word Count: {word_count}") 
        
        return render_template("word_count2.html", predicted_caption=predicted_caption, new_caption1=new_captions[0] if new_captions else None, new_caption2=new_captions[1] if len(new_captions) > 1 else None, new_caption3=new_captions[2] if len(new_captions) > 2 else None)

    return render_template("word_count2.html")


if __name__ == "__main__":
    app.run(debug=True)

