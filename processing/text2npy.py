from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import nltk
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import csv
import contractions

wordstokeep = {'can', 'again', 'each', 'too', 'our', 'any', 'nor',
        'only', 'why', 'was', 'out', 'other', 'now', 'doing', 'just',
        'ours', 'did', 'down', 'they', 'be', 'up', 'most', 'off', 'does',
        'are', 'were', 'having', 'do', 'has', 'not', 'until',  'before', 
        'yourself', 'both', 'you', 'here', 'than', 'will', 'more',
        'because', 'once', 'where', 'when', 'your', 'how', 'same', 'few',
        'there', 'them', 'all', 'who', 'him', 'but', 'we', 'after',
        'me', 'my', 'against', 'myself', 'no'}


def normalize(text):
    nltk.download("punkt")
    punctuation = string.punctuation
    nltk.download("wordnet")
    lemmatizer = WordNetLemmatizer()
    snowball = SnowballStemmer("english")
    nltk.download("stopwords")
    stopwords = set(nltk.corpus.stopwords.words("english"))
    for word in wordstokeep:
        stopwords.remove(word)
    normalized_text = []
    for sequence in text:
        normalized_sequence = []
        sequence = contractions.fix(sequence)
        for t in nltk.word_tokenize(sequence):
            t = t.lower()
            t = lemmatizer.lemmatize(t)
            if t not in stopwords and t not in punctuation and t.isalnum():
                normalized_sequence.append(t)
        normalized_text.append(normalized_sequence)
    return normalized_text


def tokenize(root_path, csv_col):
    with open(f"{root_path}/metadata.csv") as metadata_f:
        csv_reader = csv.reader(metadata_f, delimiter=",")
        line_count = 0
        text = []
        for row in csv_reader:
            if line_count > 0:
                text.append(row[csv_col])
            line_count += 1

    text = normalize(text)

    lengths = [len(seq) for seq in text]

    print(
        f"Max sequence length: {max(lengths)}, mean length: {np.mean(lengths)}, std dev: {np.std(lengths)}"
    )

    # Tokenize the sentences
    tokenizer = Tokenizer()

    # preparing vocabulary
    tokenizer.fit_on_texts(list(text))

    # converting text into integer sequences
    text_seq = tokenizer.texts_to_sequences(text)

    # padding to prepare sequences of same length
    text_seq = pad_sequences(text_seq, maxlen=np.max(lengths))

    np.save(f"text_seq_{np.max(lengths)}_lemmas.npy", text_seq)
    size_of_vocab = len(tokenizer.word_index) + 1  # +1 for padding
    print(f"Size of vocab: {size_of_vocab}")
    return size_of_vocab, tokenizer


def proc_embed(size_of_vocab, tokenizer):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open("/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/data/glove.840B.300d.txt", encoding="utf8")

    for line in f:
        values = line.split()
        word = "".join(values[:-300])
        coefs = np.asarray(values[-300:], dtype="float32")
        embeddings_index[word] = coefs

    f.close()
    print("Loaded %s word vectors." % len(embeddings_index))
    check_coverage(tokenizer.word_index, embeddings_index)
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((size_of_vocab, 300))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    np.save(f"embeddings_lemmas.npy", embedding_matrix)



def check_coverage(vocab, embeddings_index):
    import operator

    a = {}
    oov = {}
    k = 0
    j = 0
    for word, i in vocab.items():
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            j += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + j)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    print(sorted_x[:10])


def iemocap_cov_msp():
    root_path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code/data"

    with open(f"{root_path}/iemocap/metadata.csv") as metadata_f:
        csv_reader = csv.reader(metadata_f, delimiter=",")
        line_count = 0
        iemo_text = []
        for row in csv_reader:
            if line_count > 0:
                iemo_text.append(row[7])
            line_count += 1

    with open(f"{root_path}/msp-podcast/metadata.csv") as metadata_f:
        csv_reader = csv.reader(metadata_f, delimiter=",")
        line_count = 0
        msp_text = []
        for row in csv_reader:
            if line_count > 0:
                msp_text.append(row[5])
            line_count += 1


    iemo_text = normalize(iemo_text)
    msp_text = normalize(msp_text)

    iemo_len = [len(seq) for seq in iemo_text]
    msp_len = [len(seq) for seq in msp_text]

    print(f"IEMOCAP:\tMax length: {max(iemo_len)}, mean length: {np.mean(iemo_len)}, std dev: {np.std(iemo_len)}")
    print(f"MSP:\tMax length: {max(msp_len)}, mean length: {np.mean(msp_len)}, std dev: {np.std(msp_len)}")

    # Tokenize the sentences
    iemo_tokenizer = Tokenizer()
    msp_tokenizer = Tokenizer()

    # preparing vocabulary
    iemo_tokenizer.fit_on_texts(list(iemo_text))
    msp_tokenizer.fit_on_texts(list(msp_text))

    # converting text into integer sequences
    iemo_text_seq = iemo_tokenizer.texts_to_sequences(iemo_text)
    msp_text_seq = msp_tokenizer.texts_to_sequences(msp_text)

    # padding to prepare sequences of same length
    iemo_text_seq = pad_sequences(iemo_text_seq, maxlen=np.max(iemo_len))
    msp_text_seq = pad_sequences(msp_text_seq, maxlen=np.max(msp_len))

    iemo_vocab = len(iemo_tokenizer.word_index) + 1  # +1 for padding
    msp_vocab = len(msp_tokenizer.word_index) + 1  # +1 for padding
    print(f"Size of vocab:\t IEMOCAP:{iemo_vocab}\t MSP:{msp_vocab}")

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open("/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/data/glove.840B.300d.txt", encoding="utf8")

    for line in f:
        values = line.split()
        word = "".join(values[:-300])
        coefs = np.asarray(values[-300:], dtype="float32")
        embeddings_index[word] = coefs

    f.close()
    print("Loaded %s word vectors." % len(embeddings_index))
    # create a weight matrix for words in training docs
    msp_embed = {}
    iemo_embed = {}
    for word, i in iemo_tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            iemo_embed[word] = embedding_vector

    iemo_words_in_msp_embed = 0
    for word, token in msp_tokenizer.word_index.items():
        if word in iemo_embed.keys():
           iemo_words_in_msp_embed += 1

    print(f'There are {iemo_words_in_msp_embed} words from MSP vocab out of {len(iemo_embed.keys())} words in IEMOCAP embeddingss')

if __name__ == "__main__":
    dataset = "iemocap"
    root_path = f"/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code/data/{dataset}"
    if dataset == "iemocap":
        csv_col = 7
    else:
        csv_col = 5
    size_of_vocab, tokenizer = tokenize(root_path, csv_col)
    # proc_embed(size_of_vocab, tokenizer)
    # iemocap_cov_msp()