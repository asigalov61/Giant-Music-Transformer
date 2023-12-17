# Giant Music Transformer Fine-Tune

***

## Fine-Tune Version

[![Open In Colab][colab-badge]][colab-notebook1]

[colab-notebook1]: <https://colab.research.google.com/github/asigalov61/Giant-Music-Transformer/blob/main/Fine-Tune/Giant_Music_Transformer_Fine_Tune.ipynb>
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>

### This notebook will allow you to fine-tune Giant Music Transformer pre-trained models on your MIDI dataset

***

## How to use:

### 1) Process your MIDI dataset as normal with [Giant Music Transformer Training Dataset Maker](https://github.com/asigalov61/Giant-Music-Transformer/blob/main/Training-Data/Giant_Music_Transformer_L_XL_Training_Dataset_Maker.ipynb)
### 2) Load resulting pickle files and fine-tune desired pre-trained model with the provided fine-tune code/colab
### 3) Use any of the generator colabs for inference and evaluation as usual. All you have to do is to load your fine-tuned model checkpoint with the model loader

***

## Fine-tune pro-tip:

### For best results fine-tune on quality homogenous MIDI datasets like [Mono Melodies Piano Violin Drums MIDI Dataset](https://github.com/asigalov61/Tegridy-MIDI-Dataset/blob/master/Mono-Melodies/Piano-Violin-Drums/Mono-Melodies-Piano-Violin-Drums-CC-BY-NC-SA.zip)

***

### Sample fine-tuned model checkpoint is available on [Hugging Face](https://huggingface.co/asigalov61/Giant-Music-Transformer/blob/main/Giant_Music_Transformer_Large_Fine_Tuned_Model_MMPVD_dataset_29501_steps_0.4661_loss_0.8679_acc.pth)

### This is a Giant Music Transformer Large Pre-Trained Model which was fine-tuned on the Mono Melodies Piano Violin Drums MIDI dataset

***

### Project Los Angeles
### Tegridy Code 2023
