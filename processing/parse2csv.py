#!#/usr/bin/env python3
import pickle
import numpy as np
import os
import sys

import wave
import copy
import math

from helper import *
from scipy.io import wavfile
import speech_recognition as sr

import csv
import time


def parse_msp():
    root_path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code/data/msp"
    r = sr.Recognizer()
    since = time.time()
    with open(f'metadata.csv', 'w', encoding='utf-8') as metadata_f:
        with open(f'{root_path}/labels/labels_concensus.csv', 'r', encoding='utf-8') as labels_f:
            writer_meta = csv.writer(metadata_f)
            reader_label = csv.reader(labels_f, delimiter=',')

            # Csv headline
            writer_meta.writerow(['filename', 'emotion', 'valence', 'arousal', 'dominance', 'transcription'])

            line_count = 0
            for row in reader_label:
                if line_count > 0:
                    name, emo, a, v, d = row[0][:-4], row[1], row[2], row[3], row[4]
                    
                    audio_path = f'{root_path}/Audios/{name}.wav'
                    txt = audio2txt(audio_path, r, sr)

                    writer_meta.writerow([name, emo, v, a, d, txt])
                    print(f'{line_count}\t-\t{name}\t{txt}')
                line_count += 1

            time_elapsed = time.time() - since
            print(f'Processed {line_count} audio files from {root_path}\nin \
                 {time_elapsed // 60: .0f}m {time_elapsed % 60: .0f}s')


def parse_iemocap():
    code_path = os.path.dirname(os.path.realpath(os.getcwd()))
    emotions_used = np.array(
        ['ang', 'exc', 'neu', 'sad', 'xxx', 'fru', 'hap', 'sur', 'dis', 'fea', 'oth'])
    data_path = "../data/iemocap/"
    sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    framerate = 16000
    data = []
    ids = {}
    # Title for csv
    with open(f'metadata.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Csv headline
        writer.writerow(['id', 'start', 'end', 'emotion',
                         'valence', 'arousal', 'dominance', 'transcription'])
        # Csv headline

        for session in sessions:
            path_to_sentences = data_path + session + '/sentences/wav/'
            path_to_wav = data_path + session + '/dialog/wav/'
            path_to_emotions = data_path + session + '/dialog/EmoEvaluation/'
            path_to_transcriptions = data_path + session + '/dialog/transcriptions/'
    #        path_to_mocap_hand = data_path + session + '/dialog/MOCAP_hand/'
    #        path_to_mocap_rot = data_path + session + '/dialog/MOCAP_rotated/'
    #        path_to_mocap_head = data_path + session + '/dialog/MOCAP_head/'

            files2 = os.listdir(path_to_wav)
            files = []
            for f in files2:
                if f.endswith(".wav"):
                    if f[0] == '.':
                        files.append(f[2:-4])
                    else:
                        files.append(f[:-4])

            for f in files:
                print(f)
                mocap_f = f
                if (f == 'Ses05M_script01_1b'):
                    mocap_f = 'Ses05M_script01_1'

                wav = get_audio(path_to_wav, f + '.wav')
                transcriptions = get_transcriptions(
                    path_to_transcriptions, f + '.txt')
                emotions = get_emotions(path_to_emotions, f + '.txt')
                sample = split_wav(wav, emotions)

                for ie, e in enumerate(emotions):
                    if 'F' in e['id']:
                        e['signal'] = sample[ie]['left']
                    else:
                        e['signal'] = sample[ie]['right']

                    #e['signal'] = get_audio(path_to_sentences + f[:-5] + '/', f + '.wav')
                    #_, e['signal'] = wavfile.read(path_to_sentences + f[:-5] + '/' + f + '.wav')
                    e['transcription'] = transcriptions[e['id']]
                    #e['mocap_hand'] = get_mocap_hand(path_to_mocap_hand, mocap_f + '.txt', e['start'], e['end'])
                    #e['mocap_rot'] = get_mocap_rot(path_to_mocap_rot, mocap_f + '.txt', e['start'], e['end'])
                    #e['mocap_head'] = get_mocap_head(path_to_mocap_head, mocap_f + '.txt', e['start'], e['end'])
                    if e['emotion'] in emotions_used:
                        if e['id'] not in ids:
                            data.append(e)
                            ids[e['id']] = 1
                            # Write to csv
                            writer.writerow(
                                [e['id'], e['start'], e['end'], e['emotion'], e['v'], e['a'], e['d'], e['transcription']])

    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]


# parse_iemocap()
parse_msp()
