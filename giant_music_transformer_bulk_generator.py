# -*- coding: utf-8 -*-
"""Giant_Music_Transformer_Bulk_Generator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/asigalov61/Giant-Music-Transformer/blob/main/Giant_Music_Transformer_Bulk_Generator.ipynb

# Giant Music Transformer Bulk Generator (ver. 2.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/

***

#### Project Los Angeles

#### Tegridy Code 2023

***

# (GPU CHECK)
"""

#@title NVIDIA GPU check
!nvidia-smi

"""# (SETUP ENVIRONMENT)"""

#@title Install dependencies
!git clone --depth 1 https://github.com/asigalov61/Giant-Music-Transformer
!pip install huggingface_hub
!pip install torch
!pip install einops
!pip install torch-summary
!pip install tqdm
!pip install matplotlib
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

import datetime

print('=' * 70)
print('Creating I/O dirs...')

if not os.path.exists('/content/Output/Improvs'):
    os.makedirs('/content/Output/Improvs')

if not os.path.exists('/content/Output/Continuations'):
    os.makedirs('/content/Output/Continuations')

print('=' * 70)
print('Loading main Giant Music Transformer modules...')
import torch

# %cd /content/Giant-Music-Transformer

import TMIDIX

from midi_to_colab_audio import midi_to_colab_audio

from x_transformer_1_23_2 import *

import random

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
                    local_dir_use_symlinks=False)

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
                    local_dir_use_symlinks=False)

print('=' * 70)
print('Instantiating model...')

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda'

if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
  dtype = 'bfloat16'
else:
  dtype = 'float16'

if model_precision == 'float16':
  dtype = 'float16'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

SEQ_LEN = 8192

# instantiate the model

model = TransformerWrapper(
    num_tokens = 19464,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = num_layers, heads = 32, attn_flash = True)
)

model = AutoregressiveWrapper(model, ignore_index=19463)

model.cuda()
print('=' * 70)

print('Loading model checkpoint...')

model.load_state_dict(torch.load(model_path))
print('=' * 70)

model.eval()

print('Done!')
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

"""# (GENERATE)

# (BULK IMPROV)
"""

#@title Bulk Improv Generator

#@markdown NOTE: You can stop bulk generation at any time

#@markdown Improv type

improv_type = "Random Freestyle" # @param ["Random Freestyle", "Freestyle without Drums", "Freestyle with Drums", "Custom"]

#@markdown Custom Improv settings

first_note_MIDI_patch_number = 0 # @param {type:"slider", min:0, max:128, step:1}
add_drums = False #@param {type:"boolean"}

#@markdown Generation settings

number_of_generation_cycles = 10 # @param {type:"slider", min:1, max:256, step:1}
number_of_tokens_to_generate = 2046 # @param {type:"slider", min:30, max:8190, step:3}
number_of_batches_to_generate = 8 #@param {type:"slider", min:1, max:16, step:1}
temperature = 0.9 # @param {type:"slider", min:0.1, max:1, step:0.05}

#@markdown Other settings

verbose = False # @param {type:"boolean"}

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')

if improv_type == 'Random Freestyle':

  outy = [19461]
  improv_type = 'Random_Freestyle'

if improv_type == 'Freestyle without Drums':

  outy = [19461, 19330]
  improv_type = 'Freestyle_without_Drums'

if improv_type == 'Freestyle with Drums':

  outy = [19461, 19331]
  improv_type = 'Freestyle_with_Drums'

if improv_type == 'Custom':

  if add_drums:
    drumsp = 19331 # Yes
    dr = 1
  else:
    drumsp = 19330 # No
    dr = 0

  outy = [19461, drumsp, 19332+first_note_MIDI_patch_number]

  improv_type += '_P'+str(first_note_MIDI_patch_number)+'_D'+str(dr)

output_dir = '/content/Output/Improvs/'+improv_type+'_'+str(date_time) + '/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('=' * 70)
print('Giant Music Transformer Bulk Improv Model Generator')
print('=' * 70)
print('Generating', number_of_generation_cycles * number_of_batches_to_generate, 'compositions', number_of_tokens_to_generate, 'tokens each')
print('=' * 70)
print('Total number of tokens to generate:', number_of_generation_cycles*number_of_batches_to_generate * number_of_tokens_to_generate)
print('=' * 70)
print('Output directory:', output_dir)
print('=' * 70)
print('Generating...Please wait...')
print('=' * 70)

try:

  for gc in tqdm.tqdm(range(number_of_generation_cycles)):

    if verbose:
      print('Selected Improv sequence:')
      print(outy)
      print('=' * 70)

    torch.cuda.empty_cache()

    inp = [outy] * number_of_batches_to_generate

    inp = torch.LongTensor(inp).cuda()

    with ctx:
      out = model.generate(inp,
                            number_of_tokens_to_generate,
                            temperature=temperature,
                            return_prime=True,
                            verbose=verbose)

    out0 = out.tolist()

    if verbose:
      print('=' * 70)
      print('Done!')
      print('=' * 70)

    torch.cuda.empty_cache()

    #======================================================================

    if verbose:
      print('Rendering results...')

    for i in range(number_of_batches_to_generate):

      if verbose:
        print('=' * 70)
        print('Batch #', i)
        print('=' * 70)

      out1 = out0[i]

      if verbose:
        print('Sample INTs', out1[:12])
        print('=' * 70)

      if len(out1) != 0:

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

                  song_f.append(['note', time, dur, channel, pitch, vel ])

          patches = [0 if x==-1 else x for x in patches]

          data = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Giant Music Transformer',
                                                          output_file_name = output_dir + 'Giant-Music-Transformer-Music-Composition_'+str(i+(gc*number_of_batches_to_generate)).zfill(5),
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=patches,
                                                          verbose=verbose)

          if verbose:
            print('=' * 70)

except KeyboardInterrupt:
  print('STOP!!!')
  print('=' * 70)
  print('Stopping generation...')

print('Done!')
print('=' * 70)

"""# (BULK CUSTOM MIDI CONTINUATIONS)"""

#@title Load Seed MIDI

#@markdown Press play button to to upload your own seed MIDI or to load one of the provided sample seed MIDIs from the dropdown list below

select_seed_MIDI = "Upload your own custom MIDI" # @param ["Upload your own custom MIDI", "Giant-Music-Transformer-Piano-Seed-1", "Giant-Music-Transformer-Piano-Seed-2", "Giant-Music-Transformer-Piano-Seed-3", "Giant-Music-Transformer-Piano-Seed-4", "Giant-Music-Transformer-Piano-Seed-5", "Giant-Music-Transformer-Piano-Seed-6", "Giant-Music-Transformer-MI-Seed-1", "Giant-Music-Transformer-MI-Seed-2", "Giant-Music-Transformer-MI-Seed-3", "Giant-Music-Transformer-MI-Seed-4", "Giant-Music-Transformer-MI-Seed-5", "Giant-Music-Transformer-MI-Seed-6"]
render_MIDI_to_audio = False # @param {type:"boolean"}

print('=' * 70)
print('Giant Music Transformer Seed MIDI Loader')
print('=' * 70)

f = ''

if select_seed_MIDI != "Upload your own custom MIDI":
  print('Loading seed MIDI...')
  f = '/content/Giant-Music-Transformer/Seeds/'+select_seed_MIDI+'.mid'
  score = TMIDIX.midi2single_track_ms_score(open(f, 'rb').read(), recalculate_channels=False)

else:
  print('Upload your own custom MIDI...')
  print('=' * 70)
  uploaded_MIDI = files.upload()
  if list(uploaded_MIDI.keys()):
    score = TMIDIX.midi2single_track_ms_score(list(uploaded_MIDI.values())[0], recalculate_channels=False)
    f = list(uploaded_MIDI.keys())[0]

if f != '':

  fn = f.split('/')[-1]
  custom_midi_file_name = fn.split('.mid')[0]

  print('=' * 70)
  print('File:', f)
  print('=' * 70)

  #=======================================================
  # START PROCESSING

  # Convering MIDI to ms score with MIDI.py module
  score = TMIDIX.midi2single_track_ms_score(open(f, 'rb').read(), recalculate_channels=False)

  # INSTRUMENTS CONVERSION CYCLE
  events_matrix = []
  itrack = 1
  patches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  while itrack < len(score):
      for event in score[itrack]:
          if event[0] == 'note' or event[0] == 'patch_change':
              events_matrix.append(event)
      itrack += 1

  events_matrix.sort(key=lambda x: x[1])

  events_matrix1 = []

  for event in events_matrix:
          if event[0] == 'patch_change':
                patches[event[2]] = event[3]

          if event[0] == 'note':
                event.extend([patches[event[3]]])

                if events_matrix1:
                    if (event[1] == events_matrix1[-1][1]):
                        if ([event[3], event[4]] != events_matrix1[-1][3:5]):
                            events_matrix1.append(event)
                    else:
                        events_matrix1.append(event)

                else:
                    events_matrix1.append(event)

  if len(events_matrix1) > 0:
      if min([e[1] for e in events_matrix1]) >= 0 and min([e[2] for e in events_matrix1]) >= 0:

          #=======================================================
          # PRE-PROCESSING

          # checking number of instruments in a composition
          instruments_list_without_drums = list(set([y[3] for y in events_matrix1 if y[3] != 9]))
          instruments_list = list(set([y[3] for y in events_matrix1]))

          if len(events_matrix1) > 0 and len(instruments_list_without_drums) > 0:

              #======================================

              events_matrix2 = []

              # Recalculating timings
              for e in events_matrix1:

                  # Original timings
                  e[1] = int(e[1] / 16)
                  e[2] = int(e[2] / 16)

              #===================================
              # ORIGINAL COMPOSITION
              #===================================

              # Sorting by patch, pitch, then by start-time

              events_matrix1.sort(key=lambda x: x[6])
              events_matrix1.sort(key=lambda x: x[4], reverse=True)
              events_matrix1.sort(key=lambda x: x[1])

              #=======================================================
              # FINAL PROCESSING

              melody_chords = []
              melody_chords2 = []

              # Break between compositions / Intro seq

              if 9 in instruments_list:
                  drums_present = 19331 # Yes
              else:
                  drums_present = 19330 # No

              if events_matrix1[0][3] != 9:
                  pat = events_matrix1[0][6]
              else:
                  pat = 128

              melody_chords.extend([19461, drums_present, 19332+pat]) # Intro seq

              #=======================================================
              # MAIN PROCESSING CYCLE
              #=======================================================

              abs_time = 0

              pbar_time = 0

              pe = events_matrix1[0]

              chords_counter = 1

              comp_chords_len = len(list(set([y[1] for y in events_matrix1])))

              for e in events_matrix1:

                  #=======================================================
                  # Timings...

                  # Cliping all values...
                  delta_time = max(0, min(255, e[1]-pe[1]))

                  # Durations and channels

                  dur = max(0, min(255, e[2]))
                  cha = max(0, min(15, e[3]))

                  # Patches
                  if cha == 9: # Drums patch will be == 128
                      pat = 128

                  else:
                      pat = e[6]

                  # Pitches

                  ptc = max(1, min(127, e[4]))

                  # Velocities

                  # Calculating octo-velocity
                  vel = max(8, min(127, e[5]))
                  velocity = round(vel / 15)-1

                  #=======================================================
                  # Outro seq

                  # if ((comp_chords_len - chords_counter) == 50) and (delta_time != 0):
                  #    out_t = 18946+delta_time
                  #    out_p = 19202+ptc
                  #    melody_chords.extend([18945, out_t, out_p]) # outro seq


                  # if delta_time != 0:
                  #    chords_counter += 1

                  #=======================================================
                  # FINAL NOTE SEQ

                  # Writing final note asynchronously

                  dur_vel = (8 * dur) + velocity
                  pat_ptc = (129 * pat) + ptc

                  melody_chords.extend([delta_time, dur_vel+256, pat_ptc+2304])
                  melody_chords2.append([delta_time, dur_vel+256, pat_ptc+2304])

                  pe = e

                  #=======================================================

              # melody_chords.extend([19462, 19462, 19462]) # EOS

              #=======================================================

              # TOTAL DICTIONARY SIZE 19462+1=19463
              #=======================================================

  #=======================================================

  song = melody_chords

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
                                                            output_file_name = '/content/Giant-Music-Transformer-Seed-Composition',
                                                            track_name='Project Los Angeles',
                                                            list_of_MIDI_patches=patches
                                                            )

  #=======================================================

  print('=' * 70)
  print('Composition stats:')
  print('Composition has', len(melody_chords2), 'notes')
  print('Composition has', len(melody_chords), 'tokens')
  print('Composition MIDI patches:', sorted(list(set([((y-2304) // 129) for y in melody_chords if 2304 <= y < 18945]))))
  print('=' * 70)

  print('Displaying resulting composition...')
  print('=' * 70)

  fname = '/content/Giant-Music-Transformer-Seed-Composition'

  if render_MIDI_to_audio:
    midi_audio = midi_to_colab_audio(fname + '.mid')
    display(Audio(midi_audio, rate=16000, normalize=False))

  TMIDIX.plot_ms_SONG(song_f, plot_title=fname)

else:
  print('=' * 70)

"""# (BULK CONTINUATIONS)"""

#@title Bulk Continuations

#@markdown NOTE: You can stop bulk generation at any time

#@markdown Generation settings

try_to_generate_outro = False #@param {type:"boolean"}
number_of_prime_tokens = 1020 # @param {type:"slider", min:3, max:8190, step:3}
number_of_generation_cycles = 10 # @param {type:"slider", min:1, max:256, step:1}
number_of_tokens_to_generate = 1041 # @param {type:"slider", min:30, max:8190, step:3}
number_of_batches_to_generate = 4 #@param {type:"slider", min:1, max:16, step:1}
temperature = 0.9 # @param {type:"slider", min:0.1, max:1, step:0.05}

#@markdown Other settings
include_prime_tokens_in_generated_output = True #@param {type:"boolean"}
allow_model_to_stop_generation_if_needed = False #@param {type:"boolean"}
verbose = False # @param {type:"boolean"}

if allow_model_to_stop_generation_if_needed:
  min_stop_token = 19462
else:
  min_stop_token = None

outy = melody_chords[:number_of_prime_tokens]

if try_to_generate_outro:
  outy.extend([18945])

num_gen_toks = min(8190, number_of_tokens_to_generate)

date_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d_%H-%M-%S')

output_dir = '/content/Output/Continuations/'+custom_midi_file_name+'_'+str(date_time) + '/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('=' * 70)
print('Giant Music Transformer Bulk Continuations Model Generator')
print('=' * 70)
print('Generating', number_of_generation_cycles * number_of_batches_to_generate, 'continuations blocks', number_of_tokens_to_generate, 'tokens each')
print('=' * 70)
print('Total number of tokens to generate:', number_of_generation_cycles * number_of_batches_to_generate * number_of_tokens_to_generate)
print('=' * 70)
print('Output directory:', output_dir)
print('=' * 70)
print('Generating...Please wait...')
print('=' * 70)

try:

  for gc in tqdm.tqdm(range(number_of_generation_cycles)):

    torch.cuda.empty_cache()

    inp = [outy] * number_of_batches_to_generate

    inp = torch.LongTensor(inp).cuda()

    with ctx:
      out = model.generate(inp,
                            num_gen_toks,
                            temperature=temperature,
                            return_prime=include_prime_tokens_in_generated_output,
                            eos_token=min_stop_token,
                            verbose=verbose)

    out0 = out.tolist()

    torch.cuda.empty_cache()

    if verbose:
      print('=' * 70)
      print('Done!')
      print('=' * 70)

    #======================================================================
    if verbose:
      print('Rendering results...')

    for i in range(number_of_batches_to_generate):

      if verbose:
        print('=' * 70)
        print('Batch #', i)
        print('=' * 70)

      out1 = out0[i]

      if verbose:
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

                  song_f.append(['note', time, dur, channel, pitch, vel ])

          patches = [0 if x==-1 else x for x in patches]

          detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                                    output_signature = 'Giant Music Transformer',
                                                                    output_file_name = output_dir + 'Giant-Music-Transformer-Music-Composition_'+str(i + (gc * number_of_batches_to_generate)).zfill(5),
                                                                    track_name='Project Los Angeles',
                                                                    list_of_MIDI_patches=patches,
                                                                    verbose=verbose
                                                                    )
      if verbose:
          print('=' * 70)

except KeyboardInterrupt:
  print('STOP!!!')
  print('=' * 70)
  print('Stopping generation...')

print('Done!')
print('=' * 70)

"""# (ZIP AND DOWNLOAD RESULTS)"""

# Commented out IPython magic to ensure Python compatibility.
#@title Zip and download all bulk generations results

print('=' * 70)

try:
    os.remove('Giant_Music_Transoformer_Bulk_Generations.zip')
except OSError:
    pass

print('Zipping... Please wait...')
print('=' * 70)

# %cd /content/Output/
!zip -r Giant_Music_Transoformer_Bulk_Generator_Output.zip *
# %cd /content/

print('=' * 70)
print('Done!')
print('=' * 70)

print('Downloading final zip file...')
print('=' * 70)

files.download('/content/Output/Giant_Music_Transoformer_Bulk_Generator_Output.zip')

print('Done!')
print('=' * 70)

"""# Congrats! You did it! :)"""