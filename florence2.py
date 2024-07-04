import requests  # For making HTTP requests
import time  # For time-related functions

import subprocess  # For running system commands
import os 
import io
import threading

florence2_server_url = "http://213.173.96.19:5002/" 
def handle_captioning(img_byte_arr, results):
    caption = send_image_for_captioning(img_byte_arr)
    results['caption'] = caption

# Function to handle OCR
def handle_ocr(img_byte_arr, results):
    ocr_result = send_image_for_ocr(img_byte_arr)
    results['ocr'] = ocr_result

def send_image_for_captioning(image_data):
    # Server URL
    url = florence2_server_url+"caption"  # Replace with your server's IP address

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
            caption = response.json()["caption"] ['<MORE_DETAILED_CAPTION>']
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


def send_image_for_ocr(image_data):
    # Server URL
    url = url = florence2_server_url+"ocr" # Replace with your server's IP address

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
            print(response.json())
            caption = response.json()["ocr"]['<OCR>']
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


def get_caption_from_clipboard():
    # Check clipboard content

    try:
       content = ImageGrab.grabclipboard()
    except:
        content = clipboard.paste()
        print(type(content))
        if isinstance(content, str):
            print("Returning text from the clipboard...")
            return content
    print(content)
    print(type(content))
    
    if isinstance(content, Image.Image):
        print("Processing an image from the clipboard...")
        if content.mode != 'RGB':
            content = content.convert('RGB')
            
        # Save image to a byte array
        img_byte_arr = io.BytesIO()
        content.save(img_byte_arr, format='JPEG', quality=60)
        img_byte_arr = img_byte_arr.getvalue()

        results = {}
        
        # Define tasks for threads
        thread1 = threading.Thread(target=handle_captioning, args=(img_byte_arr, results))
        thread2 = threading.Thread(target=handle_ocr, args=(img_byte_arr, results))

        # Start threads
        thread1.start()
        thread2.start()

        # Wait for threads to complete
        thread1.join()
        thread2.join()

        # Combine results and return
        combined_caption = results.get('caption', '') + "\nOCR RESULTS:\n" + results.get('ocr', '')
        return combined_caption

    else:
        return "No image or text data found in the clipboard."

# Functions `handle_captioning` and `handle_ocr` need to be defined elsewhere in your code.
# They should update the `results` dictionary with keys 'caption' and 'ocr' respectively.

def get_caption_from_screenshot():


    # Take a screenshot and open it with PIL
    print("Taking a screenshot...")
    screenshot_image = screenshot()  # Uses PyAutoGUI to take a screenshot
    #width, height = screenshot_image.size
    #new_height = 800
    #new_width = int((new_height / height) * width)
    
    # Resizing with the correct resampling filter
    #resized_image = screenshot_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized image as JPEG
    img_byte_arr = io.BytesIO()
    #resized_image.save(img_byte_arr, format='JPEG', quality=60)
    screenshot_image.save(img_byte_arr, format='JPEG', quality=60)
    img_byte_arr = img_byte_arr.getvalue()

    # Send image for captioning and return the result
    #caption = send_image_for_captioning(img_byte_arr)
    ocr_result= send_image_for_ocr(img_byte_arr)
    print(ocr_result)
    #caption += "\nOCR RESULTS:\n"+ocr_result
    
    results = {}
    
    thread1 = threading.Thread(target=handle_captioning, args=(img_byte_arr, results))
    thread2 = threading.Thread(target=handle_ocr, args=(img_byte_arr, results))

    # Start threads
    thread1.start()
    #time.sleep(2)
    thread2.start()

    # Wait for threads to complete
    thread1.join()
    thread2.join()
    print(results)
    # Combine results and print
    combined_caption = results['caption'] + "\nOCR RESULTS:\n"+ results['ocr']
        
    return combined_caption

