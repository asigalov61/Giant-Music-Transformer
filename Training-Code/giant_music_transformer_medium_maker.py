# -*- coding: utf-8 -*-
"""Giant_Music_Transformer_Medium_Maker.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14ThDuqaLjzedkxLBlGzzK1okVTHdhmGP

# Giant Music Transformer Medium Maker (ver. 1.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/

***

#### Project Los Angeles

#### Tegridy Code 2024

***

# GPU check
"""

!nvidia-smi

"""# Setup environment"""

!git clone --depth 1 https://github.com/asigalov61/tegridy-tools

!sudo pip3 install torch torchvision torchaudio
!pip3 install -U torch torchvision torchaudio
!sudo pip install einops
!sudo pip install torch-summary
!sudo pip3 install -U tqdm
#!pip install matplotlib

# Commented out IPython magic to ensure Python compatibility.
# Load modules and make data dir

print('Loading modules...')

import os
import pickle
import random
import secrets
import tqdm
import math

import gc

!set USE_FLASH_ATTENTION=1
os.environ['USE_FLASH_ATTENTION'] = '1'

import torch
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

# %cd /home/ubuntu/tegridy-tools/tegridy-tools/

import TMIDIX

# %cd /home/ubuntu/tegridy-tools/tegridy-tools/X-Transformer

from x_transformer_1_23_2 import *

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(False)

!set USE_FLASH_ATTENTION=1

# %cd /home/ubuntu/

if not os.path.exists('/home/ubuntu/INTS'):
    os.makedirs('/home/ubuntu/INTS')

import random

print('Done')

print('Torch version:', torch.__version__)

"""# Load training data"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /home/ubuntu/INTS/
!unzip DATA.zip
!rm DATA.zip
# %cd /home/ubuntu/

"""# Files List"""

dataset_addr = "/home/ubuntu/INTS"

#==========================================================================

filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames if file.endswith('.pickle')]
print('=' * 70)

random.shuffle(filez)

print('Loaded', len(filez), 'data files')
print('=' * 70)

"""# Load Training Data"""

SEQ_LEN = 8192
PAD_IDX = 19463 # Model pad index

#==========================================================================

print('=' * 70)
print('Loading data files...')
print('Please wait...')
print('=' * 70)

train_data = []
outros = []

chunks_counter = 0
outros_counter = 0
middle_counter = 0

gc.disable()

for lfa in tqdm.tqdm(filez):

    train_d = pickle.load(open(lfa, 'rb'))

    for t in train_d:

        if 0 <= max(t) < PAD_IDX: # final data integrity check

            pseq = t[:(SEQ_LEN+1)] + [PAD_IDX] * ((SEQ_LEN+1) - len(t[:(SEQ_LEN+1)]))

            train_data.append(pseq)

            chunks_counter += 1

            if len(t) > (SEQ_LEN * 2):
                train_data.append(t[-(SEQ_LEN+1):])
                outros_counter += 1

            if len(t) > (SEQ_LEN * 3):
                train_data.append(t[(len(t) // 2)-4096:(len(t) // 2)+4098][:SEQ_LEN+1])
                middle_counter += 1

        else:
            print('Bad data!!!')
            break

gc.enable()

#==========================================================================

print('Done!')
print('=' * 70)
print('Total number of main chunks:', chunks_counter)
print('Total number of outro chunks:', outros_counter)
print('Total number of middle chunks:', middle_counter)
print('All data is good:', len(max(train_data, key=len)) == len(min(train_data, key=len)))
print('=' * 70)
print('Randomizing train data...')
random.shuffle(train_data)
print('Done!')
print('=' * 70)
print('Total length of train data:', len(train_data))
print('=' * 70)

len(train_data[0])

train_data[0][:15]

"""# Setup model"""

# Setup model

# constants

VALIDATE_EVERY  = 100
SAVE_EVERY = 500
GENERATE_EVERY  = 250
GENERATE_LENGTH = 512
PRINT_STATS_EVERY = 10

NUM_EPOCHS = 5

BATCH_SIZE = 8
GRADIENT_ACCUMULATE_EVERY = 8

LEARNING_RATE = 1e-4
GRAD_CLIP = 1.5

# instantiate the model

model = TransformerWrapper(
    num_tokens = PAD_IDX+1,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 2048,
                          depth = 8,
                          heads = 32,
                          rotary_pos_emb = True,
                          attn_flash = True
                         )
    )

model = AutoregressiveWrapper(model, ignore_index = PAD_IDX, pad_value=PAD_IDX)

model.cuda()

print('Done!')

summary(model)

# Dataloader

def get_train_data_batch(tdata, index, seq_len, batch_size, pad_idx):

    batch = tdata[(index*batch_size):(index*batch_size)+batch_size]

    return torch.LongTensor(batch).cuda()

# precision/optimizer/scaler

dtype = torch.bfloat16

ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = torch.amp.GradScaler('cuda')

"""# Train"""

# Train the model

train_losses = []
val_losses = []

train_accs = []
val_accs = []

nsteps = 0

for ep in range(NUM_EPOCHS):

        print('=' * 70)
        print('Randomizing train data...')
        random.shuffle(train_data)
        print('=' * 70)

        print('=' * 70)
        print('Epoch #', ep)
        print('=' * 70)

        NUM_BATCHES = len(train_data) // BATCH_SIZE // GRADIENT_ACCUMULATE_EVERY

        model.train()

        for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='Training'):

            optim.zero_grad()

            for j in range(GRADIENT_ACCUMULATE_EVERY):
                with ctx:
                    loss, acc = model(get_train_data_batch(train_data, (i*GRADIENT_ACCUMULATE_EVERY)+j, SEQ_LEN, BATCH_SIZE, PAD_IDX))
                    loss = loss / GRADIENT_ACCUMULATE_EVERY
                scaler.scale(loss).backward()

            if i % PRINT_STATS_EVERY == 0:
                print(f'Training loss: {loss.item() * GRADIENT_ACCUMULATE_EVERY}')
                print(f'Training acc: {acc.item()}')

            train_losses.append(loss.item() * GRADIENT_ACCUMULATE_EVERY)
            train_accs.append(acc.item())

            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optim)
            scaler.update()

            nsteps += 1

            if i % VALIDATE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    with ctx:
                        val_loss, val_acc = model(get_train_data_batch(train_data, i, SEQ_LEN, BATCH_SIZE, PAD_IDX))

                        print(f'Validation loss: {val_loss.item()}')
                        print(f'Validation acc: {val_acc.item()}')

                        val_losses.append(val_loss.item())
                        val_accs.append(val_acc.item())

                        print('Plotting training loss graph...')

                        tr_loss_list = train_losses
                        plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                        plt.show()
                        plt.close()
                        print('Done!')

                        print('Plotting training acc graph...')

                        tr_loss_list = train_accs
                        plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                        plt.show()
                        plt.close()
                        print('Done!')

                        print('Plotting validation loss graph...')
                        tr_loss_list = val_losses
                        plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                        plt.show()
                        plt.close()
                        print('Done!')

                        print('Plotting validation acc graph...')
                        tr_loss_list = val_accs
                        plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                        plt.show()
                        plt.close()
                        print('Done!')

                model.train()

            if i % GENERATE_EVERY == 0:
                model.eval()

                inp = random.choice(get_train_data_batch(train_data, i, SEQ_LEN, BATCH_SIZE, PAD_IDX))[:GENERATE_LENGTH]

                print(inp)

                with ctx:
                    sample = model.generate(inp[None, ...], GENERATE_LENGTH, return_prime=True)

                print(sample)

                data = sample.tolist()[0][GENERATE_LENGTH:]

                print('Sample INTs', data[:15])

                if len(data) != 0:

                    song = data
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
                                                                          output_file_name = '/home/ubuntu/Giant-Music-Transformer-Composition',
                                                                          track_name='Project Los Angeles',
                                                                          list_of_MIDI_patches=patches
                                                                          )

                print('Done!')

                model.train()

            if i % SAVE_EVERY == 0:

                print('Saving model progress. Please wait...')
                print('model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

                fname = '/home/ubuntu/model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

                torch.save(model.state_dict(), fname)

                data = [train_losses, train_accs, val_losses, val_accs]

                TMIDIX.Tegridy_Any_Pickle_File_Writer(data, '/home/ubuntu/losses_accs')

                print('Done!')

"""# Final Save"""

print('Saving model progress. Please wait...')
print('model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

fname = '/home/ubuntu/model_checkpoint_' + str(nsteps) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

torch.save(model.state_dict(), fname)
#torch.save(optim.state_dict(), fname+'_opt')

print('Done!')

data = [train_losses, train_accs, val_losses, val_accs]

TMIDIX.Tegridy_Any_Pickle_File_Writer(data, '/home/ubuntu/losses_accuracies')

# Save training loss graph

plt.plot([i for i in range(len(train_losses))] ,train_losses, 'b')
plt.savefig('/home/ubuntu/training_loss_graph.png')
plt.close()
print('Done!')

# Save training acc graph

plt.plot([i for i in range(len(train_accs))] ,train_accs, 'b')
plt.savefig('/home/ubuntu/training_acc_graph.png')
plt.close()
print('Done!')

# Save validation loss graph

plt.plot([i for i in range(len(val_losses))] ,val_losses, 'b')
plt.savefig('/home/ubuntu/validation_loss_graph.png')
plt.close()
print('Done!')

# Save validation acc graph

plt.plot([i for i in range(len(val_accs))] ,val_accs, 'b')
plt.savefig('/home/ubuntu/validation_acc_graph.png')
plt.close()
print('Done!')

"""# Eval"""

!sudo pip install huggingface_hub

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id='asigalov61/Giant-Music-Transformer',
                filename='Giant_Music_Transformer_Medium_Trained_Model_10446_steps_0.7202_loss_0.8233_acc.pth',
                local_dir='/home/ubuntu/Models/',
                )

SEQ_LEN = 8192
PAD_IDX = 19463

model = TransformerWrapper(
    num_tokens = PAD_IDX+1,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 2048,
                          depth = 8,
                          heads = 32,
                          rotary_pos_emb = True,
                          attn_flash = True
                         )
    )

model = AutoregressiveWrapper(model, ignore_index = PAD_IDX, pad_value=PAD_IDX)

print('=' * 70)
print('Loading model checkpoint...')

model_path = 'Models/Giant_Music_Transformer_Medium_Trained_Model_10446_steps_0.7202_loss_0.8233_acc.pth'

model.load_state_dict(torch.load(model_path))

print('=' * 70)

model.cuda()
model.eval()

print('Done!')

summary(model)

dtype = torch.bfloat16

ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)

midi_file = 'Giant-Music-Transformer-Piano-Seed-2.mid'

f = midi_file

if f != '':

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

print('Done!')
print('=' * 70)
print(len(melody_chords))
print('=' * 70)

x = (torch.tensor(melody_chords[:1000], dtype=torch.long, device='cuda')[None, ...])
#x = torch.tensor([19461, 19330+1, 19332+128] * 4, dtype=torch.long, device='cuda')
#x = torch.tensor([[0]] * 4, dtype=torch.long, device='cuda')
# run generation

with ctx:
    out = model.generate(x,
                         700,
                         temperature=0.9,
                         filter_logits_fn=top_k,
                         filter_kwargs={'k': 15},
                         return_prime=True,
                         verbose=True)

y = out.tolist()

print('---------------')

print(y)

#@title Test INTs

data = y[0]

print('Sample INTs', data[:15])

if len(data) != 0:

    song = data
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
                                                          output_file_name = '/home/ubuntu/Giant-Music-Transformer-Composition',
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=patches
                                                          )

print('Done!')

tok_emb = model.net.token_emb.emb.weight.detach().cpu().tolist()

cos_sim = metrics.pairwise_distances(
  tok_emb, metric='cosine'
)
plt.figure(figsize=(7, 7))
plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
plt.xlabel("Position")
plt.ylabel("Position")
plt.tight_layout()
plt.plot()
plt.savefig("/home/ubuntu/Giant-Music-Transformer-Tokens-Embeddings-Plot.png", bbox_inches="tight")

"""# Congrats! You did it! :)"""