import requests
import base64
from PIL import Image
import io
import time
from pyautogui import screenshot
import threading

# Function to handle image captioning
def handle_captioning(img_byte_arr, results):
    caption = send_image_for_captioning(img_byte_arr)
    results['caption'] = caption

# Function to handle OCR
def handle_ocr(img_byte_arr, results):
    ocr_result = send_image_for_ocr(img_byte_arr)
    results['ocr'] = ocr_result
    
def send_image_for_captioning(image_data):
    # Server URL
    url = "http://213.173.96.19:5002/caption"  # Replace with your server's IP address

    try:
        # Encode image data to base64
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        print("Image encoded to base64")

        # Prepare the request payload
        payload = {"image": encoded_image}

        # Send POST request to the server
        print("Sending request to server...")
        response = requests.post(url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            caption = response.json()["caption"]
            print("Received caption from server:")
            print(caption)
            return caption
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(f"Error message: {response.json().get('error', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    # Delay execution for 3 seconds
    print("Waiting for 3 seconds before taking a screenshot...")
    time.sleep(3)

    # Take a screenshot and open it with PIL
    print("Taking a screenshot...")
    screenshot_image = screenshot()  # Uses PyAutoGUI to take a screenshot
    width, height = screenshot_image.size
    new_height = 500
    new_width = int((new_height / height) * width)
    
    # Resizing with the correct resampling filter
    resized_image = screenshot_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized image as JPEG
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='JPEG', quality=50)
    img_byte_arr = img_byte_arr.getvalue()

    # Send image for captioning
    caption = send_image_for_captioning(img_byte_arr)

    if caption:
        print("Caption:")
        print(caption)
    else:
        print("Failed to get caption")

if __name__ == "__main__":
    main()
