import pandas as pd
import csv
import multiprocessing
import numpy as np


def duration_info():
    with open('metadata.csv') as metadata_f:
        csv_reader = csv.reader(metadata_f, delimiter=',')
        line_count = 0
        durs = []
        for row in csv_reader:
            if line_count > 0:
                currdur = float(row[2]) - float(row[1])
                durs.append(currdur)
            line_count += 1

        print(
            f'maxdur: {max(durs)}\n meandur: {np.mean(np.asarray(durs))}\n stddev:{np.std(np.asarray(durs))}')


def get_path(filename):
    path_to_data = '../data/iemocap'
    sessions = {
        '1': 'Session1',
        '2': 'Session2',
        '3': 'Session3',
        '4': 'Session4',
        '5': 'Session5',
    }
    # Getting session
    sess_id = filename[4]  # Get the session id number (1, 2, 3, 4 or 5)
    session_name = sessions[sess_id]
    # Getting sentence folder
    sentence_folder = filename[:14]
    # Getting sentence wav name
    wav_name = filename[:19]

    if filename[7:18] == 'script01_1b':
        sentence_folder = filename[:18]
        wav_name = filename[:23]

    elif sentence_folder[7:13] == 'script':
        sentence_folder = filename[:17]
        wav_name = filename[:22]

    elif str(filename[7:15]) in ['impro05a', 'impro05b', 'impro08a', 'impro08b']:
        sentence_folder = filename[:15]
        wav_name = filename[:20]

    path_to_wav = f'{path_to_data}/{session_name}/sentences/wav/{sentence_folder}/{wav_name}.wav'
    return path_to_wav


def extract_paa_lld(path_to_wav):
    from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
    maxdur = 8

    [sampling_rate, signal] = audioBasicIO.read_audio_file(path_to_wav)
    signal = audioBasicIO.stereo_to_mono(signal)

    # Pad signal with 0s to maxlen of all seqncs
    signal.resize(maxdur * sampling_rate, refcheck=False)
    # every 160 samples a feature set is calculated -> 128000/0.080*samplingrate = num of features obtained
    features, f_names = ShortTermFeatures.feature_extraction(signal, sampling_rate, 0.025*sampling_rate,
                                                             0.080*sampling_rate)
    features = np.delete(features, np.s_[34:], 0)
    f_names = np.delete(f_names, np.s_[34:], 0)
    print(path_to_wav)

    return pd.DataFrame(data=features.T, columns=f_names)


def extract_paa_hsf(path_to_wav):
    from pyAudioAnalysis import audioBasicIO, ShortTermFeatures

    [sampling_rate, signal] = audioBasicIO.read_audio_file(path_to_wav)
    signal = audioBasicIO.stereo_to_mono(signal)

    # every 160 samples a feature set is calculated -> 128000/160 = num of features obtained
    features, f_names = ShortTermFeatures.feature_extraction(signal, sampling_rate, 0.025*sampling_rate,
                                                             0.010*sampling_rate)
    features_hsf = {}
    for index, name in enumerate(f_names):
        if index > 33:  # we dont want the whole 68 features that include the deltas
            break
        features_hsf[f'{name}-mean'] = np.mean(features[index])
        features_hsf[f'{name}-std'] = np.std(features[index])

    print(path_to_wav)

    return features_hsf


def extract_egemaps_hsf(path_to_wav):
    import audiofile
    from pyAudioAnalysis import audioBasicIO
    import opensmile

    [signal, sampling_rate] = audiofile.read(path_to_wav)
    # signal = audioBasicIO.stereo_to_mono(signal)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_channels=1,
    )
    features = smile.process_signal(
        signal,
        sampling_rate
    )

    # Calculate HSF of LLD features and keep only those (mean, std)
    features_hsf = {}
    for name in features.columns:
        features_hsf[f'{name}-mean'] = features[name].mean()
        features_hsf[f'{name}-std'] = features[name].std()
        features.drop(columns=[name], inplace=True)

    print(path_to_wav)

    return features_hsf

def extract_egemaps_lld(path_to_wav):
    import audiofile
    from pyAudioAnalysis import audioBasicIO
    import opensmile

    [signal, sampling_rate] = audiofile.read(path_to_wav)
    # signal = audioBasicIO.stereo_to_mono(signal)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_channels=1,
    )
    features = smile.process_signal(
        signal,
        sampling_rate
    )

    print(path_to_wav)

    return features


def extract_compare_lld(path_to_wav):
    import audiofile
    from pyAudioAnalysis import audioBasicIO
    import opensmile

    [signal, sampling_rate] = audiofile.read(path_to_wav)
    # signal = audioBasicIO.stereo_to_mono(signal)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_channels=1,
    )
    features = smile.process_signal(
        signal,
        sampling_rate
    )

    print(path_to_wav)

    return features

def extract_compare_hsf(path_to_wav):
    import audiofile
    from pyAudioAnalysis import audioBasicIO
    import opensmile

    [signal, sampling_rate] = audiofile.read(path_to_wav)
    # signal = audioBasicIO.stereo_to_mono(signal)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        num_channels=1,
    )
    try:
        features = smile.process_signal(
            signal,
            sampling_rate
        )
    except Exception:
        return {}

    # Calculate HSF of LLD features and keep only those (mean, std)
    features_hsf = {}
    for name in features.columns:
        features_hsf[f'{name}-mean'] = features[name].mean()
        features_hsf[f'{name}-std'] = features[name].std()
        features.drop(columns=[name], inplace=True)

    print(path_to_wav)

    return features_hsf

# def extract_gaussian_triad()

def extract_batch(rows, namespace, lock):
    import audiofile
    from pyAudioAnalysis import audioBasicIO
    for row in rows:
        name = row[0]
        # features = {'id':name}
        path_to_wav = get_path(name)

        signal, sampling_rate = audiofile.read(path_to_wav)
        signal = audioBasicIO.stereo_to_mono(signal)

        # f_gemaps = gemaps_features(signal, sampling_rate)
        # features.update(f_gemaps)
        f_paa = paa_features(signal, sampling_rate)
        # features.update(f_paa)

        lock.acquire()
        # namespace.df = namespace.df.append(features, ignore_index=True)
        namespace.df = namespace.df.append(f_paa, ignore_index=True)
        lock.release()

        print(name)
    exit(0)
