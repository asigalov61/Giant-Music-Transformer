# https://huggingface.co/spaces/asigalov61/Inpaint-Music-Transformer

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
    
# =================================================================================================
                       
@spaces.GPU
def InpaintPitches(input_midi, input_num_of_notes, input_patch_number):
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

    fn = os.path.basename(input_midi.name)
    fn1 = fn.split('.')[0]

    input_num_of_notes = max(8, min(2048, input_num_of_notes))

    print('-' * 70)
    print('Input file name:', fn)
    print('Req num of notes:', input_num_of_notes)
    print('Req patch number:', input_patch_number)
    print('-' * 70)

    #===============================================================================
    raw_score = TMIDIX.midi2single_track_ms_score(input_midi.name)
    
    #===============================================================================
    # Enhanced score notes
    
    events_matrix1 = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    
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
          # FINAL NOTE SEQ
    
          # Writing final note asynchronously
    
          dur_vel = (8 * dur) + velocity
          pat_ptc = (129 * pat) + ptc
    
          melody_chords.extend([delta_time, dur_vel+256, pat_ptc+2304])
          melody_chords2.append([delta_time, dur_vel+256, pat_ptc+2304])
    
          pe = e
    
        
    #==================================================================

    print('=' * 70)
    print('Number of tokens:', len(melody_chords))
    print('Number of notes:', len(melody_chords2))
    print('Sample output events', melody_chords[:5])
    print('=' * 70)
    print('Generating...')

    #@title Pitches/Instruments Inpainting
    
    #@markdown You can stop the inpainting at any time to render partial results
    
    #@markdown Inpainting settings
    
    #@markdown Select MIDI patch present in the composition to inpaint
    
    inpaint_MIDI_patch = input_patch_number
    
    #@markdown Generation settings
    number_of_prime_notes = 24
    number_of_memory_tokens = 1024 # @param {type:"slider", min:3, max:8190, step:3}
    number_of_samples_per_inpainted_note = 1 #@param {type:"slider", min:1, max:16, step:1}
    temperature = 0.85
    
    print('=' * 70)
    print('Giant Music Transformer Inpainting Model Generator')
    print('=' * 70)

    #==========================================================================

    nidx = 0
    first_inote = True
    fidx = 0

    number_of_prime_tokens = number_of_prime_notes * 3
    
    for i, m in enumerate(melody_chords):
    
        if 2304 <= melody_chords[i] < 18945:
            
            cpatch = (melody_chords[i]-2304) // 129
            
            if cpatch == inpaint_MIDI_patch:
                nidx += 1
                if first_inote:
                    fidx += 1
    
                if first_inote and fidx == number_of_prime_notes:
                    number_of_prime_tokens = i
                    first_inote = False

        if nidx == input_num_of_notes:
            break
            
    nidx = i

    #==========================================================================
    
    out2 = []
    
    for m in melody_chords[:number_of_prime_tokens]:
      out2.append(m)
    
    for i in range(number_of_prime_tokens, len(melody_chords[:nidx])):
    
        cpatch = (melody_chords[i]-2304) // 129
    
        if 2304 <= melody_chords[i] < 18945 and (cpatch) == inpaint_MIDI_patch:
    
            samples = []
    
            for j in range(number_of_samples_per_inpainted_note):
    
              inp = torch.LongTensor(out2[-number_of_memory_tokens:]).cuda()
    
              with ctx:
                out1 = model.generate(inp,
                                      1,
                                      temperature=temperature,
                                      return_prime=True,
                                      verbose=False)
    
                with torch.no_grad():
                  test_loss, test_acc = model(out1)
    
              samples.append([out1.tolist()[0][-1], test_acc.tolist()])
    
            accs = [y[1] for y in samples]
            max_acc = max(accs)
            max_acc_sample = samples[accs.index(max_acc)][0]
    
            cpitch = (max_acc_sample-2304) % 129
    
            out2.extend([((cpatch * 129) + cpitch)+2304])
    
        else:
            out2.append(melody_chords[i])

    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    #===============================================================================
    print('Rendering results...')
    
    print('=' * 70)
    print('Sample INTs', out2[:12])
    print('=' * 70)
    
    if len(out2) != 0:
    
        song = out2
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
                                                              output_file_name = fn1,
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=patches
                                                              )
    
    new_fn = fn1+'.mid'
            
    
    audio = midi_to_colab_audio(new_fn, 
                        soundfont_path=soundfont,
                        sample_rate=16000,
                        volume_scale=10,
                        output_for_gradio=True
                        )
    
    print('Done!')
    print('=' * 70)

    #========================================================

    output_midi_title = str(fn1)
    output_midi_summary = str(song_f[:3])
    output_midi = str(new_fn)
    output_audio = (16000, audio)
    
    output_plot = TMIDIX.plot_ms_SONG(song_f, plot_title=output_midi, return_plt=True)

    print('Output MIDI file name:', output_midi)
    print('Output MIDI title:', output_midi_title)
    print('Output MIDI summary:', output_midi_summary)
    print('=' * 70) 
    

    #========================================================
    
    print('-' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('-' * 70)
    print('Req execution time:', (reqtime.time() - start_time), 'sec')

    return output_midi_title, output_midi_summary, output_midi, output_audio, output_plot

# =================================================================================================

if __name__ == "__main__":
    
    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)

    soundfont = "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"
   
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Inpaint Music Transformer</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Inpaint pitches in any MIDI</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Inpaint-Music-Transformer&style=flat)\n\n"
            "This is a demo of the Giant Music Transformer pitches inpainting feature\n\n"
            "Check out [Giant Music Transformer](https://github.com/asigalov61/Giant-Music-Transformer) on GitHub!\n\n"
            "[Open In Colab]"
            "(https://colab.research.google.com/github/asigalov61/Giant-Music-Transformer/blob/main/Giant_Music_Transformer.ipynb)"
            " for all features, faster execution and endless generation"
        )
        gr.Markdown("## Upload your MIDI or select a sample example MIDI below")
        
        input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"])
        input_num_of_notes = gr.Slider(8, 2048, value=128, step=8, label="Number of composition notes to inpaint")
        input_patch_number = gr.Slider(0, 127, value=0, step=1, label="Composition MIDI patch to inpaint")
        
        run_btn = gr.Button("inpaint", variant="primary")

        gr.Markdown("## Inpainting results")

        output_midi_title = gr.Textbox(label="Output MIDI title")
        output_midi_summary = gr.Textbox(label="Output MIDI summary")
        output_audio = gr.Audio(label="Output MIDI audio", format="wav", elem_id="midi_audio")
        output_plot = gr.Plot(label="Output MIDI score plot")
        output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])

        run_event = run_btn.click(InpaintPitches, [input_midi, input_num_of_notes, input_patch_number],
                                  [output_midi_title, output_midi_summary, output_midi, output_audio, output_plot])

        gr.Examples(
            [["Giant-Music-Transformer-Piano-Seed-1.mid", 128, 0], 
             ["Giant-Music-Transformer-Piano-Seed-2.mid", 128, 0], 
             ["Giant-Music-Transformer-Piano-Seed-3.mid", 128, 0],
             ["Giant-Music-Transformer-Piano-Seed-4.mid", 128, 0],
             ["Giant-Music-Transformer-Piano-Seed-5.mid", 128, 2],
             ["Giant-Music-Transformer-Piano-Seed-6.mid", 128, 0],
             ["Giant-Music-Transformer-MI-Seed-1.mid", 128, 71],
             ["Giant-Music-Transformer-MI-Seed-2.mid", 128, 40],
             ["Giant-Music-Transformer-MI-Seed-3.mid", 128, 40],
             ["Giant-Music-Transformer-MI-Seed-4.mid", 128, 40],
             ["Giant-Music-Transformer-MI-Seed-5.mid", 128, 40],
             ["Giant-Music-Transformer-MI-Seed-6.mid", 128, 0]
            ],
            [input_midi, input_num_of_notes, input_patch_number],
            [output_midi_title, output_midi_summary, output_midi, output_audio, output_plot],
            InpaintPitches,
            cache_examples=True,
        )
        
        app.queue().launch()