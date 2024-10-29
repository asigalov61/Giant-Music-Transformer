#=====================================================================
# https://huggingface.co/spaces/asigalov61/Intelligent-MIDI-Comparator
#=====================================================================

import os.path

import time as reqtime
import datetime
from pytz import timezone

import torch

import spaces
import gradio as gr

from x_transformer_1_23_2 import *
import random
import tqdm

from midi_to_colab_audio import midi_to_colab_audio
import TMIDIX

import matplotlib.pyplot as plt

from sklearn.metrics import pairwise
    
# =================================================================================================

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h*6.0)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]

def generate_colors(n):
    return [hsv_to_rgb(i/n, 1, 1) for i in range(n)]

def add_arrays(a, b):
    return [sum(pair) for pair in zip(a, b)]

def plot_ms_SONG(ms_song,
                  preview_length_in_notes=0,
                  block_lines_times_list = None,
                  plot_title='ms Song',
                  max_num_colors=129,
                  drums_color_num=128,
                  plot_size=(11,4),
                  note_height = 0.75,
                  show_grid_lines=False,
                  return_plt = False,
                  timings_multiplier=1,
                  plot_curve_values=None,
                  plot_curve_notes_step=200,
                  save_plot=''
                  ):

  '''Tegridy ms SONG plotter/vizualizer'''

  notes = [s for s in ms_song if s[0] == 'note']

  if (len(max(notes, key=len)) != 7) and (len(min(notes, key=len)) != 7):
    print('The song notes do not have patches information')
    print('Please add patches to the notes in the song')

  else:

    start_times = [(s[1] * timings_multiplier) / 1000 for s in notes]
    durations = [(s[2]  * timings_multiplier) / 1000 for s in notes]
    pitches = [s[4] for s in notes]
    patches = [s[6] for s in notes]

    colors = generate_colors(max_num_colors)
    colors[drums_color_num] = (1, 1, 1)

    pbl = (notes[preview_length_in_notes][1] * timings_multiplier) / 1000

    fig, ax = plt.subplots(figsize=plot_size)

    # Create a rectangle for each note with color based on patch number
    for start, duration, pitch, patch in zip(start_times, durations, pitches, patches):
        rect = plt.Rectangle((start, pitch), duration, note_height, facecolor=colors[patch])
        ax.add_patch(rect)

    if plot_curve_values is not None:

        stimes = start_times[plot_curve_notes_step // 2::plot_curve_notes_step]
        
        min_val = min(plot_curve_values)
        max_val = max(plot_curve_values)
        spcva = [((value - min_val) / (max(max_val - min_val, 0.00001))) * 100 for value in plot_curve_values]

        ax.plot(stimes, spcva, marker='o', linestyle='-', color='w')

    # Set the limits of the plot
    ax.set_xlim([min(start_times), max(add_arrays(start_times, durations))])
    ax.set_ylim([min(spcva), max(spcva)])

    # Set the background color to black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('white')

    if preview_length_in_notes > 0:
      ax.axvline(x=pbl, c='white')

    if block_lines_times_list:
      for bl in block_lines_times_list:
        ax.axvline(x=bl, c='white')

    if show_grid_lines:
      ax.grid(color='white')

    plt.xlabel('Time (s)', c='black')
    plt.ylabel('MIDI Pitch', c='black')

    plt.title(plot_title)

    if return_plt:
      return fig

    if save_plot == '':
      plt.show()

    else:
      plt.savefig(save_plot)

# =================================================================================================

def read_MIDI(input_midi):

    #===============================================================================
    raw_score = TMIDIX.midi2single_track_ms_score(input_midi)
    
    #===============================================================================
    # Enhanced score notes
    
    events_matrix1 = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    
    #=======================================================
    # PRE-PROCESSING
    
    instruments_list = list(set([y[3] for y in events_matrix1]))
    
    #======================================

    events_matrix1 = TMIDIX.augment_enhanced_score_notes(events_matrix1, timings_divider=16)
    
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
      # FINAL NOTE SEQ
    
      # Writing final note asynchronously
    
      dur_vel = (8 * dur) + velocity
      pat_ptc = (129 * pat) + ptc
    
      melody_chords.extend([delta_time, dur_vel+256, pat_ptc+2304])
      melody_chords2.append([delta_time, dur_vel+256, pat_ptc+2304])
    
      pe = e

    return melody_chords, melody_chords2

# =================================================================================================

def tokens_to_MIDI(tokens, MIDI_name):

    print('Rendering results...')
    
    print('=' * 70)
    print('Sample INTs', tokens[:12])
    print('=' * 70)
    
    if len(tokens) != 0:
    
        song = tokens
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
                                                              output_signature = 'Intelligent MIDI Comparator',
                                                              output_file_name = MIDI_name,
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=patches
                                                              )
    
    new_fn = MIDI_name+'.mid'
            
    
    audio = midi_to_colab_audio(new_fn, 
                        soundfont_path=soundfont,
                        sample_rate=16000,
                        volume_scale=10,
                        output_for_gradio=True
                        )
    
    print('Done!')
    print('=' * 70)

    return new_fn, song_f, audio

# =================================================================================================
                       
@spaces.GPU
def CompareMIDIs(input_src_midi, input_trg_midi, input_sampling_resolution, input_sampling_overlap):
    
    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()

    print('Loading model...')

    SEQ_LEN = 8192 # Models seq len
    PAD_IDX = 19463 # Models pad index
    DEVICE = 'cuda' # 'cuda'

    # instantiate the model

    model = TransformerWrapper(
        num_tokens = PAD_IDX+1,
        max_seq_len = SEQ_LEN,
        attn_layers = Decoder(dim = 1024, depth = 32, heads = 32, attn_flash = True)
        )
    
    model = AutoregressiveWrapper(model, ignore_index = PAD_IDX)

    model.to(DEVICE)
    print('=' * 70)

    print('Loading model checkpoint...')

    model.load_state_dict(
        torch.load('Giant_Music_Transformer_Large_Trained_Model_36074_steps_0.3067_loss_0.927_acc.pth',
                   map_location=DEVICE))
    print('=' * 70)

    model.eval()

    if DEVICE == 'cpu':
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16

    ctx = torch.amp.autocast(device_type=DEVICE, dtype=dtype)

    print('Done!')
    print('=' * 70)

    sfn = os.path.basename(input_src_midi.name)
    sfn1 = sfn.split('.')[0]

    tfn = os.path.basename(input_trg_midi.name)
    tfn1 = tfn.split('.')[0]
    
    print('-' * 70)
    print('Input src MIDI name:', sfn)
    print('Input trg MIDI name:', tfn)
    print('Req sampling resolution:', input_sampling_resolution)
    print('Req sampling overlap:', input_sampling_overlap)
    print('-' * 70)

    #===============================================================================

    print('Loading MIDIs...')

    src_tokens, src_notes = read_MIDI(input_src_midi.name)
    trg_tokens, trg_notes = read_MIDI(input_trg_midi.name)
    
    #==================================================================

    print('=' * 70)
    print('Number of src tokens:', len(src_tokens))
    print('Number of src notes:', len(src_notes))
    print('Number of trg tokens:', len(trg_tokens))
    print('Number of trg notes:', len(trg_notes))

    #==========================================================================
    
    print('=' * 70)
    print('Comparing...')    
    print('=' * 70)
    print('Giant Music Transformer MIDI Comparator')
    print('=' * 70)
  
    sampling_resolution = max(40, min(1000, input_sampling_resolution)) * 3
    sampling_overlap = max(0, min(500, input_sampling_overlap)) * 3
    
    comp_length = (min(len(src_tokens), len(trg_tokens)) // sampling_resolution) * sampling_resolution

    input_src_tokens = src_tokens[:comp_length]
    input_trg_tokens = trg_tokens[:comp_length]
    
    comp_cos_sims = []

    # torch.cuda.empty_cache()
    
    for i in range(0, comp_length, max(1, sampling_resolution-sampling_overlap)):
    
      inp = [input_src_tokens[i:i+sampling_resolution]]
    
      inp = torch.LongTensor(inp).to(DEVICE)
    
      with ctx:
        with torch.no_grad():
            out = model(inp)
          
      cache = out[2]
      src_embedings = cache.layer_hiddens[-1]
    
      inp = [input_trg_tokens[i:i+sampling_resolution]]
    
      inp = torch.LongTensor(inp).to(DEVICE)
    
      with ctx:
        with torch.no_grad():
            out = model(inp)
            
      cache = out[2]
      trg_embedings = cache.layer_hiddens[-1]
    
      cos_sim = pairwise.cosine_similarity([src_embedings.cpu().detach().numpy()[0].flatten()],
                                          [trg_embedings.cpu().detach().numpy()[0].flatten()]
                                          ).tolist()[0][0]
    
      comp_cos_sims.append(cos_sim)
    
    output_min_sim = min(comp_cos_sims)
    output_avg_sim = sum(comp_cos_sims) / len(comp_cos_sims)
    output_max_sim = max(comp_cos_sims)

    print('Min sim:', output_min_sim)
    print('Avg sim:', output_avg_sim)
    print('max sim:', output_max_sim)
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    #===============================================================================

    print('Rendering results...')

    sname, ssong_f, saudio = tokens_to_MIDI(src_tokens[:comp_length], sfn1)
    tname, tsong_f, taudio = tokens_to_MIDI(trg_tokens[:comp_length], tfn1)

    #========================================================

    output_src_audio = (16000, saudio)
    output_src_plot = plot_ms_SONG(ssong_f, 
                                   plot_title=sfn1, 
                                   plot_curve_values=comp_cos_sims, 
                                   plot_curve_notes_step=max(1, sampling_resolution-sampling_overlap) // 3, 
                                   return_plt=True
                                  )

    output_trg_audio = (16000, taudio)
    output_trg_plot = plot_ms_SONG(tsong_f, 
                                   plot_title=tfn1, 
                                   plot_curve_values=comp_cos_sims, 
                                   plot_curve_notes_step=max(1, sampling_resolution-sampling_overlap) // 3, 
                                   return_plt=True
                                  )
    
    print('Done!')
    print('=' * 70)
    
    #========================================================
    
    print('-' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('-' * 70)
    print('Req execution time:', (reqtime.time() - start_time), 'sec')

    return output_src_audio, output_src_plot, output_trg_audio, output_trg_plot, output_min_sim, output_avg_sim, output_max_sim

# =================================================================================================

if __name__ == "__main__":
    
    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)

    soundfont = "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"
   
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Intelligent MIDI Comparator</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Intelligent comparison of any pair of MIDIs</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Intelligent-MIDI-Comparator&style=flat)\n\n"
            "This is a demo for the Giant Music Transformer\n\n"
            "Check out [Giant Music Transformer](https://github.com/asigalov61/Giant-Music-Transformer) on GitHub!\n\n"
            "[Open In Colab]"
            "(https://colab.research.google.com/github/asigalov61/Giant-Music-Transformer/blob/main/Giant_Music_Transformer.ipynb)"
            " for all features, faster execution and endless generation"
        )

        
        gr.Markdown("## Upload your MIDIs or select a sample example below")
        
        gr.Markdown("## Upload source MIDI")
        
        input_src_midi = gr.File(label="Source MIDI", file_types=[".midi", ".mid", ".kar"])

        gr.Markdown("## Upload target MIDI")

        input_trg_midi = gr.File(label="Target MIDI", file_types=[".midi", ".mid", ".kar"])

        gr.Markdown("### Make sure that the MIDI has at least sampling resolution number of notes")
        
        input_sampling_resolution = gr.Slider(50, 2000, value=50, step=10, label="Sampling resolution in notes")

        gr.Markdown("### Make sure that the sampling overlap value is less than sampling resolution value")
        
        input_sampling_overlap = gr.Slider(0, 1000, value=0, step=10, label="Sampling overlap in notes")
        
        run_btn = gr.Button("compare", variant="primary")

        gr.Markdown("## MIDI comparison results")

        output_min_sim = gr.Number(label="Minimum similarity")
        output_avg_sim = gr.Number(label="Average similarity")
        output_max_sim = gr.Number(label="Maximum similarity")

        output_src_audio = gr.Audio(label="Source MIDI audio", format="mp3", elem_id="midi_audio")
        output_src_plot = gr.Plot(label="Source MIDI plot")

        output_trg_audio = gr.Audio(label="Target MIDI audio", format="mp3", elem_id="midi_audio")
        output_trg_plot = gr.Plot(label="Target MIDI plot")
        
        run_event = run_btn.click(CompareMIDIs, [input_src_midi, input_trg_midi, input_sampling_resolution, input_sampling_overlap],
                                  [output_src_audio, output_src_plot, output_trg_audio, output_trg_plot, output_min_sim, output_avg_sim, output_max_sim])

        gr.Examples(
            [
            ["Honesty.kar", "Hotel California.mid", 200, 0],
            ["House Of The Rising Sun.mid", "Nothing Else Matters.kar", 200, 0],
            ["Deep Relaxation Melody #6.mid", "Deep Relaxation Melody #8.mid", 200, 0],
            ["I Just Called To Say I Love You.mid", "Sharing The Night Together.kar", 200, 0],    
            ],
            [input_src_midi, input_trg_midi, input_sampling_resolution, input_sampling_overlap],
            [output_src_audio, output_src_plot, output_trg_audio, output_trg_plot, output_min_sim, output_avg_sim, output_max_sim],
            CompareMIDIs,
            cache_examples=True,
            cache_mode='eager'
        )
        
        app.queue().launch()