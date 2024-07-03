"""
dependencies:

git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
pip install SoundFile torchaudio munch torch pydub pyyaml librosa nltk matplotlib accelerate transformers phonemizer einops einops-exts tqdm typing-extensions git+https://github.com/resemble-ai/monotonic_align.git
sudo apt-get install espeak-ng
git-lfs clone https://huggingface.co/yl4579/StyleTTS2-LJSpeech
mv StyleTTS2-LJSpeech/Models .
pip install flask StyleTTS2
pip install SoundFile torchaudio munch torch pydub pyyaml librosa nltk matplotlib accelerate transformers phonemizer einops einops-exts tqdm typing-extensions git+https://github.com/resemble-ai/monotonic_align.git
"""


from flask import Flask, request, jsonify
from flask import send_from_directory
import threading
import time
import json
import os  # Import the os module


import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

import nltk
nltk.download('punkt')
import os
import soundfile as sf
# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
#os.chdir("./StyleTTS2")
from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()

# %matplotlib inline

device = 'cuda' if torch.cuda.is_available() else 'cpu'

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(ref_dicts):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)

        with torch.no_grad():
            ref = model.style_encoder(mel_tensor.unsqueeze(1))
        reference_embeddings[key] = (ref.squeeze(1), audio)

    return reference_embeddings

# load phonemizer
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True, words_mismatch='ignore')

config = yaml.safe_load(open("Models/LJSpeech/config.yml"))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)


def inference(text, noise, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    text = text.replace('"', '')
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise,
              embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
              embedding_scale=embedding_scale).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()

def LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
  text = text.strip()
  text = text.replace('"', '')
  ps = global_phonemizer.phonemize([text])
  ps = word_tokenize(ps[0])
  ps = ' '.join(ps)

  tokens = textclenaer(ps)
  tokens.insert(0, 0)
  tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

  with torch.no_grad():
      input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
      text_mask = length_to_mask(input_lengths).to(tokens.device)

      t_en = model.text_encoder(tokens, input_lengths, text_mask)
      bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
      d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

      s_pred = sampler(noise,
            embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
            embedding_scale=embedding_scale).squeeze(0)

      if s_prev is not None:
          # convex combination of previous and current style
          s_pred = alpha * s_prev + (1 - alpha) * s_pred

      s = s_pred[:, 128:]
      ref = s_pred[:, :128]

      d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

      x, _ = model.predictor.lstm(d)
      duration = model.predictor.duration_proj(x)
      duration = torch.sigmoid(duration).sum(axis=-1)
      pred_dur = torch.round(duration.squeeze()).clamp(min=1)

      pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
      c_frame = 0
      for i in range(pred_aln_trg.size(0)):
          pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
          c_frame += int(pred_dur[i].data)

      # encode prosody
      en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
      F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
      out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(device)),
                              F0_pred, N_pred, ref.squeeze().unsqueeze(0))

  return out.squeeze().cpu().numpy(), s_pred
  

text="hello world"
start = time.time()
noise = torch.randn(1,1,256).to(device)
wav = inference(text, noise, diffusion_steps=5, embedding_scale=1)
rtf = (time.time() - start) / (len(wav) / 24000)
print(f"RTF = {rtf:5f}")


# Define the file path where the audio will be saved
file_path = 'output.wav'

# Specify the sample rate. For most speech processing tasks, 24000 Hz is typical.
sample_rate = 24000

# Save the waveform to a WAV file
sf.write(file_path, wav, sample_rate)

print(f"File saved successfully at {file_path}")




app = Flask(__name__)


@app.route('/add', methods=['GET'])
def add_item():
    """Adds an item to the list received via a GET request."""
    item = request.args.get('item')
    if item:
        items.append(item)
        return jsonify(success=True, message="Item added successfully."), 200
    else:
        return jsonify(success=False, message="No item provided."), 400


@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    text = request.json.get('text')
    print("received request:", text)
    if not text:
        return jsonify(success=False, message="No text provided."), 400

    try:
        # Generate speech
        noise = torch.randn(1,1,256).to(device)
        s= time.time()
        audio = inference(text, noise, diffusion_steps=3, embedding_scale=1.2)
        print("Time for inference:",time.time()-s)
        # Generate a random filename
        random_number = random.randint(10000, 99999)
        filename = f"speech_{random_number}.wav"

        # Ensure the 'audio' directory exists
        os.makedirs('audio', exist_ok=True)

        # Save the audio file
        sf.write(os.path.join('audio', filename), audio, 24000)

        return jsonify(success=True, filename=filename), 200

    except Exception as e:
        return jsonify(success=False, message=f"An error occurred: {str(e)}"), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    try:
        return send_from_directory('audio', filename)
    except FileNotFoundError:
        return jsonify(success=False, message="Audio file not found"), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
    
os.system("curl ifconfig.me")

if __name__ == '__main__':

    # Starts the Flask web server
    app.run(host='0.0.0.0', port=5001)


