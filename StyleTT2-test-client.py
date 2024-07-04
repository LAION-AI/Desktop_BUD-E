import requests
import io
import sounddevice as sd
import soundfile as sf
import time

SERVER_URL = "http://213.173.96.19:5001"  # Replace with your server's IP and port

def synthesize_and_play(text):
    s=time.time()
    # Send the text to the server
    response = requests.post(f"{SERVER_URL}/synthesize", json={"text": text})
    
    if response.status_code == 200:
        filename = response.json()['filename']
        print(f"Audio file generated: {filename}")
        
        # Download the audio file
        audio_response = requests.get(f"{SERVER_URL}/audio/{filename}")
        
        if audio_response.status_code == 200:
            # Load the audio data
            audio_data, sample_rate = sf.read(io.BytesIO(audio_response.content))
            print("Latency:",time.time()-s)
            # Play the audio
            sd.play(audio_data, sample_rate)
            sd.wait()  # Wait until the audio is finished playing
        else:
            print(f"Failed to download audio file. Status code: {audio_response.status_code}")
    else:
        print(f"Failed to synthesize speech. Status code: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    while True:
    
        text = input("Enter text to synthesize (or 'q' to quit): ")
        if text.lower() == 'q':
            break
        
        synthesize_and_play(text)
        
        
