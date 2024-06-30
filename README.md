This is a voice assitant that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.

It currently uses [Deepgram](www.deepgram.com) for the audio service and [Groq](https://groq.com/) the LLM. For the wake word detection it uses Porcupine from Pico Voice ( https://picovoice.ai/platform/porcupine/ ).

This demo utilizes streaming for sst and tts to speed things up.

In the future it should be made working with Open Source APIs, that you can host yourself.

# Installation
Recommendation: Make a venv and install everything in the venv. Make sure your microphone works.

Set your API keys here:
```
echo 'export PORCUPINE_API_KEY="yourgroqapikeyhere"' >> ~/.bashrc
echo 'export GROQ_API_KEY="yourgroqapikeyhere"' >> ~/.bashrc
echo 'export DEEPGRAM_API_KEY="yourdeepgramapikeyhere"' >> ~/.bashrc
source ~/.bashrc

git clone https://github.com/christophschuhmann/Desktop_BUD-E

pip install -U -r requirements.txt
 
python3 buddy.py
```
