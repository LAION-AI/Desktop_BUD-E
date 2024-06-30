# DEMO

This is a alpha demo showing a bot that uses Text-To-Speech, Speech-To-Text, and a language model to have a conversation with a user.

This demo is set up to use [Deepgram](www.deepgram.com) for the audio service and [Groq](https://groq.com/) the LLM.

This demo utilizes streaming for sst and tts to speed things up.

The files in `building_blocks` are the isolated components if you'd like to inspect them

# Installation
Recommendation: Make a venv and install everything in the venv. Make sure your microphone works.

git clone https://github.com/christophschuhmann/Desktop_BUD-E

pip install -U -r requirements.txt
 
```
python3 buddy.py
```
