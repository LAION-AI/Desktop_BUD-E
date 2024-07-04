# Import necessary libraries
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from transformers import AutoProcessor, AutoModelForCausalLM
import time
import torch
from threading import Lock  # Import Lock

# Initialize Flask app and a lock
app = Flask(__name__)
lock = Lock()  # Global lock for serialized access

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Florence-2 model and processor
print("Loading Florence-2 model and processor...")
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
model = model.to(device)  # Move model to GPU
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
print("Model and processor loaded successfully.")

@app.route('/caption', methods=['POST'])
def caption_image():
    with lock:
        print("Received caption request")
        
        if 'image' not in request.json:
            print("Error: No image data in request")
            return jsonify({"error": "No image data provided"}), 400
        
        try:
            image_data = base64.b64decode(request.json['image'])
            image = Image.open(io.BytesIO(image_data))
            print("Image decoded successfully")

            prompt = "<MORE_DETAILED_CAPTION>" 
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            print("Inputs prepared for the model")
            s = time.time()
            
            generated_ids = model.generate(
                input_ids=inputs["input_ids"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
                max_new_tokens=512,
                do_sample=True,
                num_beams=3
            )
            print("Caption generated", time.time() - s)

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
            print("Caption processed")

            return jsonify({"caption": parsed_answer}), 200

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route('/ocr', methods=['POST'])
def ocr_image():
    with lock:
        print("Received ocr request")
        
        if 'image' not in request.json:
            print("Error: No image data in request")
            return jsonify({"error": "No image data provided"}), 400
        
        try:
            image_data = base64.b64decode(request.json['image'])
            image = Image.open(io.BytesIO(image_data))
            print("Image decoded successfully")

            prompt = "<OCR>" 
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            print("Inputs prepared for the model")
            s = time.time()
            print(device)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"].to(device),
                pixel_values=inputs["pixel_values"].to(device),
                max_new_tokens=512,
                do_sample=True,
                num_beams=3
            )
            print("OCR generated", time.time() - s)

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
            print("OCR processed")

            return jsonify({"ocr": parsed_answer}), 200

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5002)

