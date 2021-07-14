from torch import nn
import torch
import numpy as np


class AudioLSTM(nn.Module):
    def __init__(self, batch_size, timesteps, feature_size, hidden_size, dropout):
        super(AudioLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.feature_size = feature_size

	## create hidden ##
        # weight = next(self.parameters())
	# self.hidden = [(weight.new_zeros(1, self.timesteps, self.hidden_size[0]), weight.new_zeros(1, self.timesteps, self.hidden_size[0])),
        #      (weight.new_zeros(1, self.timesteps, self.hidden_size[1]), weight.new_zeros(1, self.timesteps, self.hidden_size[1])),
        #      (weight.new_zeros(1, self.timesteps, self.hidden_size[2]), weight.new_zeros(1, self.timesteps, self.hidden_size[2]))
	#      ]

        self.batch_norm = nn.BatchNorm1d(self.timesteps)
        self.layer1 = nn.LSTM(feature_size, hidden_size[0])
        self.layer2 = nn.LSTM(hidden_size[0], hidden_size[1])
        self.layer3 = nn.LSTM(hidden_size[1], hidden_size[2])
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self):
        weight = next(self.parameters())
        h_t0 = weight.new_zeros(1, self.timesteps, self.hidden_size[0])
        c_t0 = weight.new_zeros(1, self.timesteps, self.hidden_size[0])

        h_t1 = weight.new_zeros(1, self.timesteps, self.hidden_size[1])
        c_t1 = weight.new_zeros(1, self.timesteps, self.hidden_size[1])

        h_t2 = weight.new_zeros(1, self.timesteps, self.hidden_size[2])
        c_t2 = weight.new_zeros(1, self.timesteps, self.hidden_size[2])

        self.hidden = [(h_t0, c_t0), (h_t1, c_t1), (h_t2, c_t2)]

    def repackage_hidden(self):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        This is for doing truncated BPTT. The dependency graph of RNN can be simply viewed as this.
        If we did not truncate the history of hidden states (c, h), the back-propagated gradients would
        flow from the loss towards the beginning, which may result in gradient vanishing or exploding.
        Detailed explanation can be found here:
        https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
        """
        self.hidden = [(h.detach_(), c.detach_()) for h, c in self.hidden]

    def forward(self, input):
        input = self.batch_norm(input)

        # hidden[0] = (h_t0, c_t0)

        out, self.hidden[0] = self.layer1(input, self.hidden[0])
        out, self.hidden[1] = self.layer2(out, self.hidden[1])
        out, self.hidden[2] = self.layer3(out, self.hidden[2])

        out = out.view(input.size(0), self.timesteps *
                       self.hidden_size[2])  # flatten
        out = self.dropout(out)

        return out


class TextLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim,
        batch_size,
        timesteps,
        hidden_size,
        output_size,
        dropout,
        vocab_size,
        embeddings,
    ):
        super(TextLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden = None
        self.vocab_size = vocab_size
        self.embedding_matrix = torch.from_numpy(np.load(embeddings))

        self.embedding_dim = embedding_dim
        self.layer1 = nn.LSTM(embedding_dim, hidden_size[0])
        self.layer2 = nn.LSTM(hidden_size[0], hidden_size[1])
        self.layer3 = nn.LSTM(hidden_size[1], hidden_size[2])
        self.linear = nn.Linear(hidden_size[2], output_size)
        self.dropout = nn.Dropout(dropout)

        self.init_embeddings()

    def init_embeddings(self):
        self.embedding_matrix  # .cuda()
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim)  # .cuda()
        self.embedding.from_pretrained(self.embedding_matrix, freeze=False)

    def repackage_hidden(self):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        This is for doing truncated BPTT. The dependency graph of RNN can be simply viewed as this.
        If we did not truncate the history of hidden states (c, h), the back-propagated gradients would
        flow from the loss towards the beginning, which may result in gradient vanishing or exploding.
        Detailed explanation can be found here:
        https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
        """
        self.hidden = [(h.detach_(), c.detach_()) for h, c in self.hidden]

    def init_hidden(self):
        weight = next(self.parameters())
        h_t0 = weight.new_zeros(1, self.timesteps, self.hidden_size[0])
        c_t0 = weight.new_zeros(1, self.timesteps, self.hidden_size[0])

        h_t1 = weight.new_zeros(1, self.timesteps, self.hidden_size[1])
        c_t1 = weight.new_zeros(1, self.timesteps, self.hidden_size[1])

        h_t2 = weight.new_zeros(1, self.timesteps, self.hidden_size[2])
        c_t2 = weight.new_zeros(1, self.timesteps, self.hidden_size[2])

        self.hidden = [(h_t0, c_t0), (h_t1, c_t1), (h_t2, c_t2)]

    def forward(self, input):
        # hidden[0] = (h_t0, c_t0)
        embed_input = self.embedding(input)

        # Reshape from (batchSize, timesteps, num_words, len_word_embedding) -> (bS, ts, len_word_embedding)
        embed_input = embed_input.view(
            input.size(0), self.timesteps, self.embedding_dim
        )
        # print(f'Embedded input:  {embed_input.size()}')

        out, self.hidden[0] = self.layer1(embed_input, self.hidden[0])
        out, self.hidden[1] = self.layer2(out, self.hidden[1])
        out, self.hidden[2] = self.layer3(out, self.hidden[2])

        # we want only the out from last timestep
        out = out[:, self.timesteps - 1, :]
        # print(f'Lstm output: {out.shape}')

        out = self.linear(out)
        out = self.dropout(out)

        return out


class DenseVAD(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(DenseVAD, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])

        self.v_linear = nn.Linear(hidden_size[1], 1)
        self.a_linear = nn.Linear(hidden_size[1], 1)
        self.d_linear = nn.Linear(hidden_size[1], 1)

    def forward(self, concat_in):
        # out = self.relu(self.linear1(out))
        # out = self.relu(self.linear2(out))
        out_vad = self.linear1(concat_in)
        out_vad = self.linear2(out_vad)
        out_vad = self.dropout(out_vad)

        out_v = self.v_linear(out_vad)
        out_a = self.a_linear(out_vad)
        out_d = self.d_linear(out_vad)

        out_vad = torch.cat((out_v, out_a, out_d), dim=1)

        return out_vad


class DenseEMO(nn.Module):
    def __init__(self, input_size, hidden_size, n_categories, dropout):
        super(DenseEMO, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_size, hidden_size[0])
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.onehot_linear = nn.Linear(hidden_size[1], n_categories)

    def forward(self, concat_in):
        # out = self.relu(self.linear1(out))
        # out = self.relu(self.linear2(out))
        out_cat = self.linear1(concat_in)
        out_cat = self.linear2(out_cat)
        out_cat = self.dropout(out_cat)

        out_cat = self.onehot_linear(out_cat)
        return out_cat


class Network(nn.Module):
    def __init__(
        self, input_size, audio_net, text_net, hidden_size, dropout, n_categories
    ):
        super(Network, self).__init__()
        self.audio_net = audio_net
        self.text_net = text_net
        self.dense_vad = DenseVAD(input_size, hidden_size, dropout)
        self.dense_emo = DenseEMO(
            input_size, hidden_size, n_categories, dropout)

    def init_hidden(self):
        self.text_net.init_hidden()
        self.audio_net.init_hidden()

    def repackage_hidden(self):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        self.audio_net.repackage_hidden()
        self.text_net.repackage_hidden()

    def forward(self, audio_input, text_input):
        audio_out = self.audio_net(audio_input)
        text_out = self.text_net(text_input)
        out = torch.cat((audio_out, text_out), dim=1)

        out_dim = self.dense_vad(out)
        out_cat = self.dense_emo(out)
        return out_cat, out_dim


def create_network(
    train_bsz,
    vocab_size,
    embed_path,
    audio_params,
    text_params,
    net_params,
    n_cat,
    device,
):

    audio_net = AudioLSTM(
        batch_size=train_bsz,
        timesteps=audio_params["timesteps"],
        feature_size=audio_params["feature_size"],
        hidden_size=audio_params["hidden_size"],
        dropout=audio_params["dropout"],
    ).to(device)

    text_net = TextLSTM(
        embedding_dim=text_params["embed_dim"],
        batch_size=train_bsz,
        timesteps=text_params["timesteps"],
        hidden_size=text_params["hidden_size"],
        output_size=text_params["output_size"],
        dropout=text_params["dropout"],
        vocab_size=vocab_size,
        embeddings=embed_path,
    ).to(device)

    input_size = text_net.output_size + \
        (audio_net.timesteps * audio_net.hidden_size[2])
    net = Network(
        input_size=input_size,
        audio_net=audio_net,
        text_net=text_net,
        hidden_size=net_params["hidden_size"],
        dropout=net_params["dropout"],
        n_categories=n_cat,
    ).to(device)

    return net


net = create_network(train_bsz=64, vocab_size=2913,
                     embed_path='/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/data/iemocap/npy/embeddings_lemmas.npy',
                     audio_params={
                         "timesteps": 1,
                         "feature_size": 198,
                         "hidden_size": (256, 256, 256),
                         "dropout": 0.3,
                     },
                     text_params={
                         "embed_dim": 300,
                         "timesteps": 73,
                         "hidden_size": (256, 256, 256),
                         "output_size": 64,
                         "dropout": 0.3,
                     },
                     net_params={"hidden_size": (64, 32), "dropout": 0.4},
                     n_cat=6,
                     device='cpu')

net.init_hidden()
dummy_audio_input = torch.randn(64, 1, 198)
dummy_text_input = torch.FloatTensor(64, 73).uniform_(0, 2913).type(torch.LongTensor)
input = (dummy_audio_input, dummy_text_input)
path = '/Users/martin/Documents/UNIVERSIDAD/CLASES/4º/2o Cuatri/TFG/code'
torch.onnx.export(net, input, f'{path}/parallel-split.onnx',
		  input_names = ['acoustic input', 'text input'],
                  output_names=['Discrete Emotion', 'Valence Arousal Dominance'],
                  dynamic_axes={'acoustic input': {0: 'batch_size'},
                                'text input': {0: 'batch_size'},
                                'Valence Arousal Dominance': {0: 'batch_size'},
                                'Discrete Emotion': {0: 'batch_size'}},
		  verbose=True)
