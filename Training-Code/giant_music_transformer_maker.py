# -*- coding: utf-8 -*-
"""Giant_Music_Transformer_Maker.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gYTkuSjJNi13cm7TXry_xxiMmRuh3v8T

# Giant Music Transformer Maker (ver. 1.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/

***

#### Project Los Angeles

#### Tegridy Code 2023

***

# GPU check
"""

!nvidia-smi

"""# Setup environment"""

!git clone https://github.com/asigalov61/tegridy-tools

!pip install --upgrade pip
!pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
!pip install einops
!pip install torch-summary
!pip install tqdm
!pip install matplotlib

# Commented out IPython magic to ensure Python compatibility.
# Load modules and make data dir

print('Loading modules...')

import os
import pickle
import random
import secrets
import tqdm
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

# %cd /content/tegridy-tools/tegridy-tools/

import TMIDIX

# %cd /content/tegridy-tools/tegridy-tools/X-Transformer

from x_transformer_1_23_2 import *

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# %cd /content/

if not os.path.exists('/content/INTS'):
    os.makedirs('/content/INTS')

import random

print('Done')

"""# Load training data"""

dataset_addr = "/content/INTS"

SEQ_LEN = 8192 # Models seq len
PAD_IDX = 19847 # Models pad index

CHUNKS_LENGTH = SEQ_LEN+1
MIN_NUMBER_OF_CHUNK_EVENTS = 512 # min number of tokens per chunk

#==========================================================================

filez = list()
for (dirpath, dirnames, filenames) in os.walk(dataset_addr):
    filez += [os.path.join(dirpath, file) for file in filenames]
print('=' * 70)

filez.sort()

#==========================================================================

print('Loading training data... Please wait...')
print('=' * 70)

train_data = []

chunks_counter = 0
discarted_chunks_counter = 0

for f in tqdm.tqdm(filez):
  train_d = pickle.load(open(f, 'rb'))
  random.shuffle(train_d)
  for t in train_d:
    for i in range(0, len(t), int((CHUNKS_LENGTH * 3) / 4)):

      #=========================================================================
      # collecting all possible chunks of chunks length

      if 0 <= max(t[i:i+CHUNKS_LENGTH]) < PAD_IDX: # final data integrity check
        if len(t[i:i+CHUNKS_LENGTH]) == CHUNKS_LENGTH:
          train_data.append(t[i:i+CHUNKS_LENGTH])

        else:
          if len(t[i:i+CHUNKS_LENGTH]) > MIN_NUMBER_OF_CHUNK_EVENTS:
            td = t[i:i+CHUNKS_LENGTH] + [PAD_IDX] * (CHUNKS_LENGTH-len(t[i:i+CHUNKS_LENGTH])) # padding with pad index
            train_data.append(td)
          else:
            discarted_chunks_counter += 1

        chunks_counter += 1

    #=========================================================================
    # Collecting middle chunk if it larger than chunks length

    if 0 <= max(t) < PAD_IDX: # final data integrity check
        if len(t) > CHUNKS_LENGTH+2:
            sidx = int((len(t) / 2)-(CHUNKS_LENGTH / 2))
            train_data.append(t[sidx:sidx+CHUNKS_LENGTH])

        else:
            discarted_chunks_counter += 1

        chunks_counter += 1
#==========================================================================

print('Done!')
print('=' * 70)
print('Total number of imput chunks:', chunks_counter)
print('Total number of good chunks:', len(train_data))
print('Total number of discarted chunks:', discarted_chunks_counter, '/', round(100 * discarted_chunks_counter/chunks_counter, 3), '%')
print('All data is good:', len(max(train_data, key=len)) == len(min(train_data, key=len)))
print('=' * 70)
print('Final data randomization...')
random.shuffle(train_data)
print('Done!')
print('=' * 70)

len(train_data)

train_data[0][:10], train_data[0][-10:]

len(train_data) // 16

"""# Setup model"""

# Setup model

# constants

NUM_EPOCHS = 1

BATCH_SIZE = 2
GRADIENT_ACCUMULATE_EVERY = 8

NUM_BATCHES = (len(train_data) // BATCH_SIZE // GRADIENT_ACCUMULATE_EVERY) * NUM_EPOCHS

LEARNING_RATE = 2e-4

VALIDATE_EVERY  = 100
SAVE_EVERY = 500
GENERATE_EVERY  = 100
PRINT_STATS_EVERY = 20

GENERATE_LENGTH = 32

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

# instantiate the model

model = TransformerWrapper(
    num_tokens = PAD_IDX+1,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = 64, heads = 32, attn_flash = True)
    )

model = AutoregressiveWrapper(model, ignore_index = PAD_IDX)

model.cuda()

print('Done!')

summary(model)

# Dataloader

class MusicDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):

        # consequtive sampling

        full_seq = torch.Tensor(self.data[index][:self.seq_len+1]).long()

        return full_seq.cuda()

    def __len__(self):
        return (len(self.data) // BATCH_SIZE) * BATCH_SIZE

train_dataset = MusicDataset(train_data, SEQ_LEN)
val_dataset   = MusicDataset(train_data, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

# precision/optimizer/scaler

dtype = torch.float16

ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler(enabled=True)

"""# Train"""

# Train the model

train_losses = []
val_losses = []

train_accs = []
val_accs = []

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='Training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        with ctx:
            loss, acc = model(next(train_loader))
            loss = loss / GRADIENT_ACCUMULATE_EVERY
            scaler.scale(loss).backward()

    if i % PRINT_STATS_EVERY == 0:
        print(f'Training loss: {loss.item() * GRADIENT_ACCUMULATE_EVERY}')
        print(f'Training acc: {acc.item()}')

    train_losses.append(loss.item() * GRADIENT_ACCUMULATE_EVERY)
    train_accs.append(acc.item())

    scaler.unscale_(optim)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optim)
    scaler.update()
    optim.zero_grad(set_to_none=True)

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            with ctx:
                val_loss, val_acc = model(next(val_loader))

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

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:-1]

        print(inp)

        with ctx:
            sample = model.generate(inp[None, ...], GENERATE_LENGTH)

        print(sample)

    if i % SAVE_EVERY == 0:

        print('Saving model progress. Please wait...')
        print('model_checkpoint_' + str(i) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

        fname = '/content/model_checkpoint_' + str(i) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

        torch.save(model.state_dict(), fname)

        data = [train_losses, train_accs, val_losses, val_accs]

        TMIDIX.Tegridy_Any_Pickle_File_Writer(data, '/content/losses_accs')

        print('Done!')

#======================================================================================================

print('Saving model progress. Please wait...')
print('model_checkpoint_' + str(i) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

fname = '/content/model_checkpoint_' + str(i) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

torch.save(model.state_dict(), fname)

print('Done!')

data = [train_losses, train_accs, val_losses, val_accs]

TMIDIX.Tegridy_Any_Pickle_File_Writer(data, '/content/losses_accuracies')

# Save training loss graph

plt.plot([i for i in range(len(train_losses))] ,train_losses, 'b')
plt.savefig('/content/training_loss_graph.png')
plt.close()
print('Done!')

# Save training acc graph

plt.plot([i for i in range(len(train_accs))] ,train_accs, 'b')
plt.savefig('/content/training_acc_graph.png')
plt.close()
print('Done!')

# Save validation loss graph

plt.plot([i for i in range(len(val_losses))] ,val_losses, 'b')
plt.savefig('/content/validation_loss_graph.png')
plt.close()
print('Done!')

# Save validation acc graph

plt.plot([i for i in range(len(val_accs))] ,val_accs, 'b')
plt.savefig('/content/validation_acc_graph.png')
plt.close()
print('Done!')

"""# Final Save"""

print('Saving model progress. Please wait...')
print('model_checkpoint_' + str(i) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth')

fname = '/content/model_checkpoint_' + str(i) + '_steps_' + str(round(float(train_losses[-1]), 4)) + '_loss_' + str(round(float(train_accs[-1]), 4)) + '_acc.pth'

torch.save(model.state_dict(), fname)

print('Done!')

data = [train_losses, train_accs, val_losses, val_accs]

TMIDIX.Tegridy_Any_Pickle_File_Writer(data, '/content/losses_accuracies')

# Save training loss graph

plt.plot([i for i in range(len(train_losses))] ,train_losses, 'b')
plt.savefig('/content/training_loss_graph.png')
plt.close()
print('Done!')

# Save training acc graph

plt.plot([i for i in range(len(train_accs))] ,train_accs, 'b')
plt.savefig('/content/training_acc_graph.png')
plt.close()
print('Done!')

# Save validation loss graph

plt.plot([i for i in range(len(val_losses))] ,val_losses, 'b')
plt.savefig('/content/validation_loss_graph.png')
plt.close()
print('Done!')

# Save validation acc graph

plt.plot([i for i in range(len(val_accs))] ,val_accs, 'b')
plt.savefig('/content/validation_acc_graph.png')
plt.close()
print('Done!')

"""# Eval"""

model.eval()

#x = (torch.tensor(random.choice(train_data)[:1000], dtype=torch.long, device=device_type)[None, ...])
x = torch.tensor([[19845, 19714+0, 19716+0]] * 4, dtype=torch.long, device=device_type)

# run generation

with ctx:
  out = model.generate(x,
                        500,
                        temperature=0.9,
                        return_prime=True,
                        verbose=True)

y = out.tolist()

print('---------------')

#@title Test INTs

data = y[0]

print('Sample INTs', data[:15])

out = data[:200000]

if len(out) != 0:

    song = out
    song_f = []

    time = 0
    dur = 0
    vel = 0
    pitch = 0
    channel = 0

    patches = [-1] * 16

    channels = [0] * 16
    channels[9] = 1

    for ss in song:

        if 0 <= ss < 512:

            time += (ss * 8)

        if 512 <= ss < 2560:

            dur = ((ss-512) // 8) * 16
            vel = (((ss-512) % 8)+1) * 15

        if 2560 <= ss < 19201:
            patch = (ss-2560) // 129

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


            pitch = (ss-2560) % 129

            song_f.append(['note', time, dur, channel, pitch, vel ])

patches = [0 if x==-1 else x for x in patches]

detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                    output_signature = 'Giant Music Transformer',
                                                    output_file_name = '/content/Giant-Music-Transformer-Composition',
                                                    track_name='Project Los Angeles',
                                                    list_of_MIDI_patches=patches
                                                    )

print('Done!')

patches

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
plt.savefig("/content/Giant-Music-Transformer-Tokens-Embeddings-Plot.png", bbox_inches="tight")

"""# Congrats! You did it! :)"""