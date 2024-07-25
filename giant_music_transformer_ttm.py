# -*- coding: utf-8 -*-
"""Giant_Music_Transformer_TTM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/asigalov61/Giant-Music-Transformer/blob/main/Giant_Music_Transformer_TTM.ipynb

# Giant Music Transformer TTM (ver. 3.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/

***

#### Project Los Angeles

#### Tegridy Code 2024

***

# (GPU CHECK)
"""

#@title NVIDIA GPU check
!nvidia-smi

"""# (SETUP ENVIRONMENT)"""

#@title Install dependencies
!git clone --depth 1 https://github.com/asigalov61/Giant-Music-Transformer
!pip install accelerate -U
!pip install bitsandbytes -U
!pip install einops
!pip install sentence-transformers
!pip install torch-summary
!apt install fluidsynth #Pip does not work for some reason. Only apt works

# Commented out IPython magic to ensure Python compatibility.
#@title Import modules

print('=' * 70)
print('Loading core Giant Music Transformer modules...')

import os
import pickle
import secrets
import statistics
from time import time
import tqdm

print('=' * 70)
print('Loading main Giant Music Transformer modules...')

import torch

from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

# %cd /content/Giant-Music-Transformer

import TMIDIX

from midi_to_colab_audio import midi_to_colab_audio

from x_transformer_1_23_2 import *

import random

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np

# %cd /content/
print('=' * 70)
print('Loading aux Giant Music Transformer modules...')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from IPython.display import Audio, display

from huggingface_hub import hf_hub_download

from google.colab import files

print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

"""# (LOAD MODEL)"""

#@title Load Giant Music Transformer Pre-Trained Model

#@markdown Choose model

select_model_to_load = "585M-32L-Very-Fast-Large" # @param ["585M-32L-Very-Fast-Large", "786M-44L-Fast-Extra-Large"]

#@markdown Model precision option

model_precision = "bfloat16" # @param ["bfloat16", "float16"]

#@markdown bfloat16 == Half precision/faster speed (if supported, otherwise the model will default to float16)

#@markdown float16 == Full precision/fast speed

enable_4bit_quantization = False # @param {type:"boolean"}

#@markdown Enable quantization option to reduce memory consumption on low VRAM GPUs

plot_tokens_embeddings = "None" # @param ["None", "Start Times", "Durations Velocities", "Piano Pitches", "Drums Pitches", "Aux"]

print('=' * 70)
print('Loading Giant Music Transformer', select_model_to_load,'Pre-Trained Model...')
print('Please wait...')
print('=' * 70)

full_path_to_models_dir = "/content/Giant-Music-Transformer/Models"

if select_model_to_load == '786M-44L-Fast-Extra-Large':

  model_checkpoint_file_name = 'Giant_Music_Transformer_Extra_Large_Trained_Model_18001_steps_0.2657_loss_0.9272_acc.pth'
  model_path = full_path_to_models_dir+'/Extra Large/'+model_checkpoint_file_name
  num_layers = 44
  if os.path.isfile(model_path):
    print('Model already exists...')

  else:
    hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                    filename=model_checkpoint_file_name,
                    local_dir='/content/Giant-Music-Transformer/Models/Extra Large',
                    )

else:

  model_checkpoint_file_name = 'Giant_Music_Transformer_Large_Trained_Model_36074_steps_0.3067_loss_0.927_acc.pth'
  model_path = full_path_to_models_dir+'/Large/'+model_checkpoint_file_name
  num_layers = 32
  if os.path.isfile(model_path):
    print('Model already exists...')

  else:
    hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                    filename=model_checkpoint_file_name,
                    local_dir='/content/Giant-Music-Transformer/Models/Large',
                    )

print('=' * 70)
print('Instantiating model...')

device_type = 'cuda'

if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
  dtype = 'bfloat16'
else:
  dtype = 'float16'

if model_precision == 'float16':
  dtype = 'float16'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

SEQ_LEN = 8192
PAD_IDX = 19463

if enable_4bit_quantization:

  with init_empty_weights():
    model = TransformerWrapper(
            num_tokens = PAD_IDX+1,
            max_seq_len = SEQ_LEN,
            attn_layers = Decoder(dim = 1024, depth = num_layers, heads = 32, attn_flash = True)
    )

    model = AutoregressiveWrapper(model, ignore_index=PAD_IDX, pad_value=PAD_IDX)

    print('=' * 70)
    print('Loading model checkpoint...')

    bnb_quantization_config = BnbQuantizationConfig(
      load_in_4bit=True,
      bnb_4bit_compute_dtype=torch.bfloat16,  # optional
      bnb_4bit_use_double_quant=True,         # optional
      bnb_4bit_quant_type="nf4"               # optional
    )
    model = load_and_quantize_model(
      model,
      weights_location=model_path,
      bnb_quantization_config=bnb_quantization_config,
      device_map = "auto"
    )

else:
  model = TransformerWrapper(
          num_tokens = PAD_IDX+1,
          max_seq_len = SEQ_LEN,
          attn_layers = Decoder(dim = 1024, depth = num_layers, heads = 32, attn_flash = True)
  )

  model = AutoregressiveWrapper(model, ignore_index=PAD_IDX, pad_value=PAD_IDX)

  print('=' * 70)
  print('Loading model checkpoint...')

  model.load_state_dict(torch.load(model_path))

print('=' * 70)

model.cuda()
model.eval()

print('Done!')
print('=' * 70)

if enable_4bit_quantization:
  print('Model will use 4bit quantization...')
else:
  print('Model will not use 4bit quantization')

print('=' * 70)
print('Model will use', dtype, 'precision...')
print('=' * 70)

# Model stats
print('Model summary...')
summary(model)

# Plot Token Embeddings
if plot_tokens_embeddings != 'None':
  tok_emb = model.net.token_emb.emb.weight.detach().cpu().tolist()

if plot_tokens_embeddings == 'Start Times':
  tok_range = [0, 256]

elif plot_tokens_embeddings == 'Durations Velocities':
  tok_range = [256, 2304]

elif plot_tokens_embeddings == 'Piano Pitches':
  tok_range = [2304, 2304+128]

elif plot_tokens_embeddings == 'Drums Pitches':
  tok_range = [18945-128, 18945]

elif plot_tokens_embeddings == 'Aux':
  tok_range = [18945, 19465]

if plot_tokens_embeddings != 'None':

  tok_emb1 = []

  for t in tok_emb[tok_range[0]:tok_range[1]]:
    tok_emb1.append(t)

  cos_sim = metrics.pairwise_distances(
    tok_emb1, metric='cosine'
  )
  plt.figure(figsize=(7, 7))
  plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
  im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
  plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
  plt.xlabel("Position")
  plt.ylabel("Position")
  plt.tight_layout()
  plt.plot()
  plt.savefig("/content/Giant-Music-Transformer-Tokens-Embeddings-Plot.png", bbox_inches="tight")

"""# (LOAD AUX MODEL AND DATA)"""

#@title Load Giant Music Transformer Aux Model and Data

print('=' * 70)
print('Loading Giant Music Transformer Aux Model and Data...')
print('Please wait...')
print('=' * 70)

print('Loading aux model...')
print('=' * 70)

aux_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')

print('Done!')
print('=' * 70)

print('Loading aux data...')

if os.path.isfile('/content/Giant-Music-Transformer/Aux-Data/Giant_Music_Transformer_Aux_Data.pickle'):
  print('Aux Data already exists...')

else:
  hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                  filename='Giant_Music_Transformer_Aux_Data.pickle',
                  local_dir='/content/Giant-Music-Transformer/Aux-Data',
                  )

print('=' * 70)
AUX_DATA = TMIDIX.Tegridy_Any_Pickle_File_Reader('/content/Giant-Music-Transformer/Aux-Data/Giant_Music_Transformer_Aux_Data')

print('=' * 70)

if os.path.isfile('/content/Giant-Music-Transformer/Aux-Data/Giant_Music_Transformer_Aux_Data_Embeddings_all_mpnet_base_v2.npz'):
  print('Aux Data already exists...')

else:
  hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                  filename='Giant_Music_Transformer_Aux_Data_Embeddings_all_mpnet_base_v2.npz',
                  local_dir='/content/Giant-Music-Transformer/Aux-Data',
                  )

print('=' * 70)
print('Loading aux data embeddings...')

aux_data_embeddings = np.load('/content/Giant-Music-Transformer/Aux-Data/Giant_Music_Transformer_Aux_Data_Embeddings_all_mpnet_base_v2.npz')['data']

print('Done!')
print('=' * 70)

"""# (GENERATE)"""

#@title Standard Continuation

#@markdown Text-To-Music Settings

#@markdown NOTE: You can enter any desired title or artist, or both

enter_desired_song_title = "Family Guy" #@param {type:"string"}
enter_desired_artist = "TV Themes" #@param {type:"string"}

#@markdown Generation settings

try_to_generate_outro = False #@param {type:"boolean"}
number_of_tokens_to_generate = 600 # @param {type:"slider", min:30, max:8190, step:3}
number_of_batches_to_generate = 4 #@param {type:"slider", min:1, max:16, step:1}
temperature = 0.9 # @param {type:"slider", min:0.1, max:1, step:0.05}
model_sampling_top_k_value = 20 # @param {type:"slider", min:1, max:100, step:1}

#@markdown Other settings
allow_model_to_stop_generation_if_needed = False #@param {type:"boolean"}
render_MIDI_to_audio = True # @param {type:"boolean"}

print('=' * 70)
print('Giant Music Transformer TTM Model Generator')
print('=' * 70)

search_string = ''

if enter_desired_song_title != '' and enter_desired_artist != '':
  search_string = '"' + enter_desired_song_title + '" by ' + enter_desired_artist

else:
  search_string = enter_desired_song_title + enter_desired_artist

print('Searching titles...Please wait...')

query_embedding = aux_model.encode([search_string])

similarities = util.cos_sim(query_embedding, aux_data_embeddings)

closest_index = np.argmax(similarities)

print('Done!')
print('=' * 70)
print('Selected title:', AUX_DATA[closest_index][0])
print('=' * 70)

if allow_model_to_stop_generation_if_needed:
  min_stop_token = 19462
else:
  min_stop_token = None

outy = AUX_DATA[closest_index][1]

block_marker = sum([(y * 16) for y in outy if y < 256]) / 1000

if try_to_generate_outro:
  outy.extend([18945])

torch.cuda.empty_cache()

inp = [outy] * number_of_batches_to_generate

inp = torch.LongTensor(inp).cuda()

with ctx:
  with torch.inference_mode():
    out = model.generate(inp,
                          number_of_tokens_to_generate,
                          filter_logits_fn=top_k,
                          filter_kwargs={'k': model_sampling_top_k_value},
                          temperature=temperature,
                          return_prime=True,
                          eos_token=min_stop_token,
                          verbose=True)

out0 = out.tolist()

torch.cuda.empty_cache()

print('=' * 70)
print('Done!')
print('=' * 70)

#======================================================================
print('Rendering results...')

for i in range(number_of_batches_to_generate):

  print('=' * 70)
  print('Batch #', i)
  print('=' * 70)

  out1 = out0[i]

  print('Sample INTs', out1[:12])
  print('=' * 70)

  if len(out) != 0:

      song = out1
      song_f = []

      time = 0
      dur = 0
      vel = 90
      pitch = 0
      channel = 0

      patches = [-1] * 16

      channels = [0] * 16
      channels[9] = 1

      for ss in song:

          if 0 <= ss < 256:

              time += ss * 16

          if 256 <= ss < 2304:

              dur = ((ss-256) // 8) * 16
              vel = (((ss-256) % 8)+1) * 15

          if 2304 <= ss < 18945:

              patch = (ss-2304) // 129

              if patch < 128:

                  if patch not in patches:
                    if 0 in channels:
                        cha = channels.index(0)
                        channels[cha] = 1
                    else:
                        cha = 15

                    patches[cha] = patch
                    channel = patches.index(patch)
                  else:
                    channel = patches.index(patch)

              if patch == 128:
                  channel = 9

              pitch = (ss-2304) % 129

              song_f.append(['note', time, dur, channel, pitch, vel, patch ])

      patches = [0 if x==-1 else x for x in patches]

      detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                output_signature = 'Giant Music Transformer',
                                                                output_file_name = '/content/Giant-Music-Transformer-Music-Composition_'+str(i),
                                                                track_name='Project Los Angeles',
                                                                list_of_MIDI_patches=patches
                                                                )
      print('=' * 70)
      print('Displaying resulting composition...')
      print('=' * 70)

      fname = '/content/Giant-Music-Transformer-Music-Composition_'+str(i)

      if render_MIDI_to_audio:
        midi_audio = midi_to_colab_audio(fname + '.mid')
        display(Audio(midi_audio, rate=16000, normalize=False))

      TMIDIX.plot_ms_SONG(song_f,
                          plot_title=fname,
                          block_lines_times_list=[block_marker])

"""# Congrats! You did it! :)"""