#==================================================================================
# https://huggingface.co/spaces/asigalov61/Giant-Music-Transformer
#==================================================================================

print('=' * 70)
print('Loading core Giant Music Transformer modules...')

import os

import time as reqtime
import datetime
from pytz import timezone

print('=' * 70)
print('Loading main Giant Music Transformer modules...')

os.environ['USE_FLASH_ATTENTION'] = '1'

import torch

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

import TMIDIX

from midi_to_colab_audio import midi_to_colab_audio

from x_transformer_1_23_2 import *

import random

print('=' * 70)
print('Loading aux Giant Music Transformer modules...')

import matplotlib.pyplot as plt

import gradio as gr
import spaces

print('=' * 70)
print('PyTorch version:', torch.__version__)
print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

#==================================================================================

SOUDFONT_PATH = 'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2'

NUM_OUT_BATCHES = 8

PREVIEW_LENGTH = 120

#==================================================================================

print('=' * 70)
print('Instantiating model...')

device_type = 'cuda'
dtype = 'bfloat16'

ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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

model = AutoregressiveWrapper(model, ignore_index=PAD_IDX, pad_value=PAD_IDX)

print('=' * 70)
print('Loading model checkpoint...')

model_path = 'Giant_Music_Transformer_Medium_Trained_Model_10446_steps_0.7202_loss_0.8233_acc.pth'

model.load_state_dict(torch.load(model_path, map_location='cpu'))

print('=' * 70)
print('Done!')
print('=' * 70)
print('Model will use', dtype, 'precision...')
print('=' * 70)

#==================================================================================

def load_midi(input_midi):

    raw_score = TMIDIX.midi2single_track_ms_score(input_midi.name)
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)
    
    escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes[0], timings_divider=16)

    instruments_list = list(set([y[6] for y in escore_notes]))

    #=======================================================
    # FINAL PROCESSING
    #=======================================================
    
    melody_chords = []

    # Break between compositions / Intro seq
    
    if 128 in instruments_list:
      drums_present = 19331 # Yes
    else:
      drums_present = 19330 # No
    
    pat = escore_notes[0][6]
    
    melody_chords.extend([19461, drums_present, 19332+pat]) # Intro seq
    
    #=======================================================
    # MAIN PROCESSING CYCLE
    #=======================================================
    
    pe = escore_notes[0]
    
    for e in escore_notes:
    
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
        # FINAL NOTE SEQ
        #=======================================================
        
        # Writing final note asynchronously
        
        dur_vel = (8 * dur) + velocity
        pat_ptc = (129 * pat) + ptc
        
        melody_chords.extend([delta_time, dur_vel+256, pat_ptc+2304])
        
        pe = e

    return melody_chords

#==================================================================================

def save_midi(tokens, batch_number=None):

    song = tokens
    song_f = []
    
    time = 0
    dur = 0
    vel = 90
    pitch = 0
    channel = 0
    
    patches = [-1] * 16
    patches[9] = 9
    
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

    if batch_number == None:
        fname = 'Giant-Music-Transformer-Music-Composition'
        
    else:
        fname = 'Giant-Music-Transformer-Music-Composition_'+str(batch_number)
    
    data = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                  output_signature = 'Giant Music Transformer',
                                                  output_file_name = fname,
                                                  track_name='Project Los Angeles',
                                                  list_of_MIDI_patches=patches,
                                                  verbose=False
                                                  )

    return song_f

#==================================================================================

@spaces.GPU
def generate_music(prime, 
                    num_gen_tokens, 
                    num_gen_batches,
                    gen_outro,
                    gen_drums,
                    model_temperature,
                    model_sampling_top_p
                  ):

    model.cuda()
    model.eval()

    print('Generating...')

    if not prime:
        inputs = [19461]

    else:
        inputs = prime

    if gen_outro:
      inputs.extend([18945])
    
    if gen_drums:
        drums = [36, 38]
        drum_pitch = random.choice(drums)
        inputs.extend([0, ((8*8)+6)+256, ((128*129)+drum_pitch)+2304])
        
    torch.cuda.empty_cache()
    
    inp = [inputs] * num_gen_batches
    
    inp = torch.LongTensor(inp).cuda()
    
    with ctx:
      with torch.inference_mode():
        out = model.generate(inp,
                              num_gen_tokens,
                              filter_logits_fn=top_p,
                              filter_kwargs={'thres': model_sampling_top_p},
                              temperature=model_temperature,
                              return_prime=False,
                              verbose=False)
    
    output = out.tolist()
    
    print('Done!')
    print('=' * 70)    
    
    return output
    
#==================================================================================

final_composition = []
generated_batches = []

#==================================================================================

def generate_callback(input_midi, 
                        num_prime_tokens, 
                        num_gen_tokens,
                        gen_outro,
                        gen_drums,
                        model_temperature,
                        model_sampling_top_p
                     ):

    global generated_batches
    generated_batches = []

    if not final_composition and input_midi is not None:
        final_composition.extend(load_midi(input_midi)[:num_prime_tokens])

    batched_gen_tokens = generate_music(final_composition, 
                                        num_gen_tokens, 
                                        NUM_OUT_BATCHES,
                                        gen_outro,
                                        gen_drums,
                                        model_temperature,
                                        model_sampling_top_p
                                       )
    
    outputs = []
    
    for i in range(len(batched_gen_tokens)):

        tokens = batched_gen_tokens[i]
        
        # Preview
        tokens_preview = final_composition[-PREVIEW_LENGTH:]
        
        # Save MIDI to a temporary file
        midi_score = save_midi(tokens_preview + tokens, i)

        # MIDI plot

        if len(final_composition) > PREVIEW_LENGTH:
            midi_plot = TMIDIX.plot_ms_SONG(midi_score, 
                                            plot_title='Batch # ' + str(i),
                                            preview_length_in_notes=int(PREVIEW_LENGTH / 3),
                                            return_plt=True
                                           )

        else:
            midi_plot = TMIDIX.plot_ms_SONG(midi_score, 
                                            plot_title='Batch # ' + str(i), 
                                            return_plt=True
                                           )

        # File name
        fname = 'Giant-Music-Transformer-Music-Composition_'+str(i)
        
        # Save audio to a temporary file
        midi_audio = midi_to_colab_audio(fname + '.mid', 
                                        soundfont_path=SOUDFONT_PATH,
                                        sample_rate=16000,
                                        output_for_gradio=True
                                        )

        outputs.append(((16000, midi_audio), midi_plot, tokens))
        
    return outputs

#==================================================================================

def generate_callback_wrapper(input_midi, 
                                num_prime_tokens, 
                                num_gen_tokens,
                                gen_outro,
                                gen_drums,
                                model_temperature,
                                model_sampling_top_p
                             ):

    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()

    print('=' * 70)
    if input_midi is not None:
            fn = os.path.basename(input_midi.name)
            fn1 = fn.split('.')[0]
            print('Input file name:', fn)
    print('Num prime tokens:', num_prime_tokens)
    print('Num gen tokens:', num_gen_tokens)
    print('Gen outro:', gen_outro)
    print('Gen drums:', gen_drums)
    print('Model temp:', model_temperature)
    print('Model top_p:', model_sampling_top_p)
    print('=' * 70)
    
    result = generate_callback(input_midi, 
                                num_prime_tokens, 
                                num_gen_tokens,
                                gen_outro,
                                gen_drums,
                                model_temperature,
                                model_sampling_top_p
                             )
    
    generated_batches.extend([sublist[2] for sublist in result])

    print('=' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)
    print('Req execution time:', (reqtime.time() - start_time), 'sec')
    print('*' * 70)
    
    return tuple(item for sublist in result for item in sublist[:2])

#==================================================================================

def add_batch(batch_number):
    
    final_composition.extend(generated_batches[batch_number])

    # Save MIDI to a temporary file
    midi_score = save_midi(final_composition)

    # MIDI plot
    midi_plot = TMIDIX.plot_ms_SONG(midi_score, plot_title='Giant Music Transformer Composition', return_plt=True)

    # File name
    fname = 'Giant-Music-Transformer-Music-Composition'
    
    # Save audio to a temporary file
    midi_audio = midi_to_colab_audio(fname + '.mid', 
                                    soundfont_path=SOUDFONT_PATH,
                                    sample_rate=16000,
                                    output_for_gradio=True
                                    )

    return (16000, midi_audio), midi_plot, fname+'.mid'

#==================================================================================

def remove_batch(batch_number, num_tokens):

    global final_composition

    if len(final_composition) > num_tokens:
        final_composition = final_composition[:-num_tokens]

    # Save MIDI to a temporary file
    midi_score = save_midi(final_composition)

    # MIDI plot
    midi_plot = TMIDIX.plot_ms_SONG(midi_score, plot_title='Giant Music Transformer Composition', return_plt=True)

    # File name
    fname = 'Giant-Music-Transformer-Music-Composition'
    
    # Save audio to a temporary file
    midi_audio = midi_to_colab_audio(fname + '.mid', 
                                    soundfont_path=SOUDFONT_PATH,
                                    sample_rate=16000,
                                    output_for_gradio=True
                                    )

    return (16000, midi_audio), midi_plot, fname+'.mid'

#==================================================================================

def reset():
    global final_composition
    final_composition = []
    
#==================================================================================

PDT = timezone('US/Pacific')

print('=' * 70)
print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
print('=' * 70)

with gr.Blocks() as demo:

    demo.load(reset)

    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Giant Music Transformer</h1>")
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Fast multi-instrumental music transformer with true full MIDI instruments range, efficient encoding, octo-velocity and outro tokens</h1>")
    gr.Markdown(
        "Check out [Giant Music Transformer](https://github.com/asigalov61/Giant-Music-Transformer) on GitHub!\n\n"
        "[Open In Colab]"
        "(https://colab.research.google.com/github/asigalov61/Giant-Music-Transformer/blob/main/Giant_Music_Transformer.ipynb)"
        " for faster execution and endless generation"
    )
    
    gr.Markdown("## Upload seed MIDI or click 'Generate' button for random output")
    
    input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"])
    input_midi.upload(reset)
    
    gr.Markdown("## Generate")
    
    num_prime_tokens = gr.Slider(15, 6999, value=600, step=3, label="Number of prime tokens")
    num_gen_tokens = gr.Slider(15, 1200, value=600, step=3, label="Number of tokens to generate")
    gen_outro = gr.Checkbox(value=False, label="Try to generate an outro")
    gen_drums = gr.Checkbox(value=False, label="Try to introduce drums")
    model_temperature = gr.Slider(0.1, 1, value=0.9, step=0.01, label="Model temperature")
    model_sampling_top_p = gr.Slider(0.1, 1, value=0.96, step=0.01, label="Model sampling top p value")
    
    generate_btn = gr.Button("Generate", variant="primary")

    gr.Markdown("## Select batch")
    
    outputs = []
    
    for i in range(NUM_OUT_BATCHES):
        with gr.Tab(f"Batch # {i}") as tab:
            
            audio_output = gr.Audio(label=f"Batch # {i} MIDI Audio", format="mp3", elem_id="midi_audio")
            plot_output = gr.Plot(label=f"Batch # {i} MIDI Plot")
            
            outputs.extend([audio_output, plot_output])

    generate_btn.click(generate_callback_wrapper, 
                       [input_midi, 
                        num_prime_tokens, 
                        num_gen_tokens,
                        gen_outro,
                        gen_drums,
                        model_temperature,
                        model_sampling_top_p
                       ], 
                       outputs
                      )
    
    gr.Markdown("## Add/Remove batch")
    
    batch_number = gr.Slider(0, NUM_OUT_BATCHES-1, value=0, step=1, label="Batch number to add/remove")
    
    add_btn = gr.Button("Add batch", variant="primary")
    remove_btn = gr.Button("Remove batch", variant="stop")
    
    final_audio_output = gr.Audio(label="Final MIDI audio", format="mp3", elem_id="midi_audio")
    final_plot_output = gr.Plot(label="Final MIDI plot")
    final_file_output = gr.File(label="Final MIDI file")

    add_btn.click(add_batch, inputs=[batch_number],
                  outputs=[final_audio_output, final_plot_output, final_file_output]                  
                 )
    
    remove_btn.click(remove_batch, inputs=[batch_number, num_gen_tokens], 
                     outputs=[final_audio_output, final_plot_output, final_file_output]                     
                    )

    demo.unload(reset)

#==================================================================================

demo.launch()

#==================================================================================