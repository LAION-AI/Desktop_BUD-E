This is a voice assitant that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.

It currently uses [Deepgram](www.deepgram.com) for the audio service and [Groq](https://groq.com/) the LLM. For the wake word detection it uses Porcupine from Pico Voice ().

This demo utilizes streaming for sst and tts to speed things up.

In the future it should be made working with Open Source APIs, that you can host yourself.

# Installation
Recommendation: Make a venv and install everything in the venv. Make sure your microphone works.

git clone https://github.com/christophschuhmann/Desktop_BUD-E

pip install -U -r requirements.txt
 
```
python3 buddy.py
```
