# Import necessary libraries
import pvporcupine  # For wake word detection
import pvrecorder  # For audio recording
import subprocess  # For running system commands
import os  # For environment variables and file operations
import signal  # For handling signals (not used in this script, but imported for potential use)
import asyncio  # For asynchronous programming
from dotenv import load_dotenv  # For loading environment variables
import shutil  # For file operations
import requests  # For making HTTP requests
import time  # For time-related functions
import threading
import clipboard

import io
import threading
from PIL import Image
from PIL import ImageGrab

import re
import subprocess
import time
import random
import requests
import base64
from PIL import Image
import io
import time
from pyautogui import screenshot
import requests
import io
import sounddevice as sd
import soundfile as sf
import time

import threading

# Import LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

# Import Deepgram components
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

from florence2 import handle_captioning
from florence2 import handle_ocr
from florence2 import send_image_for_captioning
from florence2 import send_image_for_ocr

florence2_server_url = "http://213.173.96.19:5002/" 


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



def open_site(url):
    # Use subprocess.Popen to open the browser
    process = subprocess.Popen(['open', url])
    #process = subprocess.Popen(['xdg-open', url])
    
    # Wait for 2 seconds
    time.sleep(1)
    
    # Kill the process
    process.terminate()  # Safely terminate the process
    # If terminate doesn't kill the process, you can use kill():
    # process.kill()
    
def extract_urls_to_open(input_string):
    # Define a regular expression pattern to find URLs within <open-url> tags
    pattern = r"<open-url>(https?://[^<]+)</open-url>"
    
    # Use re.findall to extract all occurrences of the pattern
    urls = re.findall(pattern, input_string)
    
    return urls


def extract_questions_to_send_to_askorkg(input_string):
    # Define a regular expression pattern to find content within <open-askorkg>...</open-orkg> tags
    pattern = r"<open-askorkg>(.*?)</open-askorkg>"
    
    # Use re.findall to extract all occurrences of the pattern
    contents = re.findall(pattern, input_string)
    
    # Return the content of the first tag pair, or None if there are no matches
    return contents[0] if contents else None


def extract_questions_to_send_to_wikipedia(input_string):
    # Define a regular expression pattern to find content within <open-askorkg>...</open-orkg> tags
    pattern = r"<open-wikipedia>(.*?)</open-wikipedia>"
    
    # Use re.findall to extract all occurrences of the pattern
    contents = re.findall(pattern, input_string)
    
    # Return the content of the first tag pair, or None if there are no matches
    return contents[0] if contents else None
    





# Load environment variables from .env file
load_dotenv()

# Define LanguageModelProcessor class
class LanguageModelProcessor:
    def __init__(self):
        # Initialize the language model (LLM)
        #self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
        # Alternatively, use OpenAI models (commented out)
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load system prompt from file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        # Create chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        # Create conversation chain
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        # Add user message to memory
        self.memory.chat_memory.add_user_message(text)

        # Record start time
        start_time = time.time()

        # Get response from LLM
        response = self.conversation.invoke({"text": text})
        
        # Record end time
        end_time = time.time()

        # Add AI response to memory
        self.memory.chat_memory.add_ai_message(response['text'])

        # Calculate elapsed time
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

# Define TextToSpeech class

class TextToSpeech:
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"

    def __init__(self):
        self.player_process = None
        self.should_stop = False

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def stop(self):
        self.should_stop = True
        if self.player_process:
            self.player_process.terminate()
            self.player_process = None

    def speak(self, text, stop_event):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        self.player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()
        first_byte_time = None

        try:
            with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
                for chunk in r.iter_content(chunk_size=1024):
                    if stop_event.is_set() or self.should_stop:
                        break
                    if chunk:
                        if first_byte_time is None:
                            first_byte_time = time.time()
                            ttfb = int((first_byte_time - start_time)*1000)
                            print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                        try:
                            self.player_process.stdin.write(chunk)
                            self.player_process.stdin.flush()
                        except BrokenPipeError:
                            print("TTS playback stopped.")
                            break
        finally:
            if self.player_process and self.player_process.stdin:
                self.player_process.stdin.close()
            if self.player_process:
                self.player_process.wait()
            self.player_process = None

class TranscriptManager:
    def __init__(self):
        self.transcript_parts = []
        self.transcription_complete = asyncio.Event()
        self.transcription_response = ""

    def reset(self):
        self.transcript_parts = []
        self.transcription_complete = asyncio.Event()
        self.transcription_response = ""

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

    def handle_full_sentence(self, full_sentence):
        self.transcription_response = full_sentence

    # TODO: Pull out and encapsulate Deepgram stuff
    async def transcribe(self):
        response = ""
        try:
            config = DeepgramClientOptions(options={"keepalive": "true"})
            deepgram: DeepgramClient = DeepgramClient("", config)
            dg_connection = deepgram.listen.asynclive.v("1")
            print("Listening...")

            async def on_message(_self, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                self.add_part(sentence)
                if result.speech_final:
                    full_sentence = self.get_full_transcript().strip()
                    if full_sentence:
                        print(f"Human: {full_sentence}")
                        self.handle_full_sentence(full_sentence)
                        self.transcription_complete.set()

            dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

            options = LiveOptions(
                model="nova-2",
                punctuate=True,
                language="en-US",
                encoding="linear16",
                channels=1,
                sample_rate=16000,
                endpointing=300,
                smart_format=True
            )

            await dg_connection.start(options)
            microphone = Microphone(dg_connection.send)
            microphone.start()
            await self.transcription_complete.wait()
            microphone.finish()
            await dg_connection.finish()
            response = self.transcription_response
            self.reset()
        except Exception as e:
            print(f"Could not open socket: {e}")
        finally:
            return response

# Create a global transcript manager instance
transcript_manager = TranscriptManager()

# Define ConversationManager class
class ConversationManager:
    def __init__(self, porcupine, recorder):
        self.llm = LanguageModelProcessor()
        self.tts = TextToSpeech()
        self.porcupine = porcupine
        self.recorder = recorder
        self.stop_event = asyncio.Event()
        self.conversation_active = False

    async def listen_for_wake_words(self):
        while self.conversation_active:
            frames = self.recorder.read()
            keyword_index = self.porcupine.process(frames)
            if keyword_index == 1:  # "Stop Buddy" detected
                print("Wake word 'Stop Buddy' detected!")
                self.stop_event.set()
                self.tts.stop()
                break
            await asyncio.sleep(0.02)  # Small delay to allow other tasks to run

    async def speak_response(self, response):
        self.recorder.start()  # Ensure recorder is started
        tts_task = asyncio.to_thread(self.tts.speak, response, self.stop_event)
        wake_word_task = asyncio.create_task(self.listen_for_wake_words())
        
        try:
            await tts_task
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            wake_word_task.cancel()
            self.recorder.stop()  # Stop recorder after TTS

    async def converse(self):
        self.conversation_active = True
        while self.conversation_active:
            self.stop_event.clear()
            self.tts = TextToSpeech()  # Create a new TTS instance for each response
            
            print("Listening for your command...")
            self.recorder.start()
            user_transcript = await transcript_manager.transcribe()
            self.recorder.stop()
            
            if "goodbye" in user_transcript.lower():
                self.conversation_active = False
                break
                
            what_buddy_sees = ""
            if any(phrase in transcript_manager.transcription_response.lower() for phrase in ["have a look", "buddy look", "look buddy", "buddy, look", "look, buddy"]):
                source = get_caption_from_screenshot if "screen" in transcript_manager.transcription_response.lower() else get_caption_from_clipboard
                what_buddy_sees = f"[BUD-E is seeing this: {source()} ] (Continue the conversation as BUD-E considering what it is seeing) "


            llm_response = self.llm.process(user_transcript+what_buddy_sees)

            extracted_url_to_open = extract_urls_to_open(llm_response)

            # Possible responses for opening a URL
            url_open_responses = [
                "Sure! Let me know if there's anything else you need.",
                "All set! Anything else you'd like to explore?",
                "The site has been opened! Feel free to ask more questions.",
                "Done! Can I assist you with anything else today?",
                "The link is now open! Let me know if you need further assistance."
            ]

            # Selecting a random response from the list
            if len(extracted_url_to_open) > 0:
                open_site(extracted_url_to_open[0])
                llm_response = random.choice(url_open_responses)

            question_for_askorkg= extract_questions_to_send_to_askorkg(llm_response)
            # Possible responses for using Ask ORKG
            ask_orkg_responses = [
                "Sure! I will use the Ask Open Knowledge Graph service to analyze the question: {0}",
                "Got it! Let's see what Ask Open Knowledge Graph has on: {0}",
                "I'm on it! Checking Ask Open Knowledge Graph for information about: {0}",
                "Excellent question! I'll consult Ask Open Knowledge Graph about: {0}",
                "One moment! I'll look that up on Ask Open Knowledge Graph for you about: {0}"
            ]

            if question_for_askorkg is not None:
                open_site("https://ask.orkg.org/search?query=" + question_for_askorkg)
                llm_response = random.choice(ask_orkg_responses).format(question_for_askorkg)
   
            question_for_wikipedia= extract_questions_to_send_to_wikipedia(llm_response)
            # Possible responses for searching Wikipedia
            wikipedia_responses = [
                "Sure! Here are the Wikipedia search results for: {0}",
                "Let me pull up Wikipedia for you to explore: {0}",
                "Checking Wikipedia for: {0}. Here's what I found!",
                "I'll search Wikipedia for that. Hold on: {0}",
                "One moment, I'm getting the information from Wikipedia on: {0}"
            ]

            if question_for_wikipedia is not None:
                open_site("https://en.wikipedia.org/w/index.php?search=" + question_for_wikipedia)
                llm_response = random.choice(wikipedia_responses).format(question_for_wikipedia)

            print(f"AI: {llm_response}")

            await self.speak_response(llm_response)

            if self.stop_event.is_set():
                print("TTS was interrupted. Ready for next command.")

        print("Conversation ended. Listening for wake words again...")

async def main():
    access_key = os.getenv("PORCUPINE_API_KEY") #  # Replace with your Picovoice AccessKey
    model_path = "hey-buddy_en_mac_v3_0_0.ppn"
    model2_path = "stop-buddy_en_mac_v3_0_0.ppn"
    #model_path = "hey-buddy_en_linux_v3_0_0.ppn"
    #model2_path = "stop-buddy_en_linux_v3_0_0.ppn"

    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[model_path, model2_path])

    print("Listening for wake word 'Hey Buddy'...")

    while True:
        try:
            recorder = pvrecorder.PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
            recorder.start()

            conversation_manager = ConversationManager(porcupine, recorder)

            while True:
                frames = recorder.read()
                keyword_index = porcupine.process(frames)
                if keyword_index == 0:  # "Hey Buddy" detected
                    print("Wake word 'Hey Buddy' detected!")
                    await conversation_manager.converse()
                    print("Conversation ended. Listening for wake word 'Hey Buddy' again...")
                    break  # Break the inner loop to create a new recorder

        except KeyboardInterrupt:
            print("Stopping...")
            break

        finally:
            recorder.stop()
            recorder.delete()

    porcupine.delete()

# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())
