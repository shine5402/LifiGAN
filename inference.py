from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
import time
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
from stft import TorchSTFT


h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)
    stft = TorchSTFT(filter_length=h.gen_istft_n_fft, hop_length=h.gen_istft_hop_size, win_length=h.gen_istft_n_fft,
                     window_device=a.device).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        ratio_sum = 0.
        ratio_count = 0
        for i, filename in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filename))
            wav_time = wav.size / sr
            start_time = time.time()

            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0))
            spec, phase = generator(x)
            y_g_hat = stft.inverse(spec, phase)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            end_time = time.time()
            inference_time = end_time - start_time
            ratio = wav_time / inference_time
            ratio_sum += ratio
            ratio_count += 1

            print('{}: wav_time={}, infer_time={}, ratio={}'.format(filename, wav_time,
                                                                    inference_time, ratio))

            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print('Saved to: {}'.format(output_file))

        print('Final ratio: {}x realtime.'.format(ratio_sum / ratio_count))


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--device', default='cuda')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available() and a.device == 'cuda':
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
        print('Using GPU to inference.')
    else:
        device = torch.device('cpu')
        torch.manual_seed(h.seed)
        print('Using CPU to inference.')

    inference(a)


if __name__ == '__main__':
    main()

