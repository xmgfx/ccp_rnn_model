# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from data.loader import Loader


class PlayCardActionEmbedding(nn.Module):
    def __init__(self, num_action, embedding_dim):
        super(PlayCardActionEmbedding, self).__init__()

        self.embed = nn.Embedding(num_embeddings=num_action,
                                  embedding_dim=embedding_dim)

    def forward(self, inputs):
        return self.embed(inputs)


class CardVecEncode(nn.Module):
    def __init__(self, encode_dim):
        super(CardVecEncode, self).__init__()
        self.card_vec_dim = 69
        self.encode = nn.Linear(in_features=self.card_vec_dim,
                                out_features=encode_dim)

    def forward(self, inputs):
        return self.encode(inputs)


class CardAllMcardActionEncode(nn.Module):
    def __init__(self):
        super(CardAllMcardActionEncode, self).__init__()
        self.num_mcard_action = 309
        self.all_mcard_action_embedding_dim = 100

        self.encode = nn.Linear(in_features=self.num_mcard_action,
                                out_features=self.all_mcard_action_embedding_dim)

    def forward(self, inputs):
        return self.encode(inputs)


class NumCardEmbedding(nn.Module):
    def __init__(self):
        super(NumCardEmbedding, self).__init__()
        self.max_num_card = 54
        self.num_card_embedding_dim = 10

        self.encode = nn.Linear(in_features=self.max_num_card,
                                out_features=self.num_card_embedding_dim)

    def forward(self, inuts):
        return self.encode(inuts)


class LSTMNetWork(nn.Module):
    num_mcard_action = 309
    num_record_mcard_action = 312
    num_record_ktype_action = 5
    num_record_klen_actoin = 8

    mcard_action_embedding_dim = 100
    record_mcard_action_embedding_dim = 100
    record_ktype_action_embedding_dim = 10
    record_klen_action_embedding_dim = 30

    def __init__(self, dropout_rate, output_dim, num_layers):
        super(LSTMNetWork, self).__init__()

        self.dropout_rate = dropout_rate
        self.output_dim = output_dim

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.mcard_action_embed = PlayCardActionEmbedding(num_action=self.num_mcard_action,
                                                          embedding_dim=self.mcard_action_embedding_dim)

        self.record_mcard_action_embed = PlayCardActionEmbedding(num_action=self.num_record_mcard_action,
                                                                 embedding_dim=self.record_mcard_action_embedding_dim)

        self.record_ktype_action_embed = PlayCardActionEmbedding(num_action=self.num_record_ktype_action,
                                                                 embedding_dim=self.record_ktype_action_embedding_dim)

        self.record_klen_actoin_embed = PlayCardActionEmbedding(num_action=self.num_record_klen_actoin,
                                                                embedding_dim=self.record_klen_action_embedding_dim)

        record_concat_dim = self.record_mcard_action_embedding_dim + \
                            self.record_ktype_action_embedding_dim + \
                            self.record_klen_action_embedding_dim
        self.lstm = nn.LSTM(
            input_size=record_concat_dim,
            hidden_size=self.output_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate)

    def forward(self, record_mcard_action, record_ktype_action, record_klen_action):
        seq_true_len = torch.sum(torch.sign(record_mcard_action), dim=-1)
        torch.nn.utils.rnn.pack_sequence()
        record_mcard_action = torch.nn.utils.rnn.pack_sequence(input=record_mcard_action, lengths=seq_true_len,
                                                                      batch_first=True)
        record_ktype_action = torch.nn.utils.rnn.pack_padded_sequence(input=record_ktype_action, lengths=seq_true_len,
                                                                      batch_first=True)
        record_klen_action = torch.nn.utils.rnn.pack_padded_sequence(input=record_klen_action, lengths=seq_true_len,
                                                                     batch_first=True)

        record_mcard_action_vec = self.record_mcard_action_embed(record_mcard_action)
        record_mcard_action_vec = self.dropout(record_mcard_action_vec)

        record_ktype_action_vec = self.record_ktype_action_embed(record_ktype_action)
        record_ktype_action_vec = self.dropout(record_ktype_action_vec)

        record_klen_action_vec = self.record_klen_actoin_embed(record_klen_action)
        record_klen_action_vec = self.dropout(record_klen_action_vec)

        record_action_vec = torch.cat(
            tensors=(record_mcard_action_vec, record_ktype_action_vec, record_klen_action_vec), dim=-1)

        output, (_, _) = self.lstm(record_action_vec)

        final_output = output[:, -1, :]

        return final_output


if __name__ == "__main__":
    path = "../data/sample100w"

    record_lstm_network = LSTMNetWork(dropout_rate=0.5, output_dim=100, num_layers=2)
    loader = Loader(path=path)

    for batch_data in loader.read_batch(batch_size=5):
        print(batch_data.keys())
        record_mcard_action = torch.tensor(batch_data['mcard_action_record'], dtype=torch.int64)
        record_mcard_action = Variable(record_mcard_action)
        record_ktype_action = torch.tensor(batch_data['ktype_action_record'], dtype=torch.int64)
        record_ktype_action = Variable(record_ktype_action)

        record_klen_action = torch.tensor(batch_data['klen_action_record'], dtype=torch.int64)
        record_klen_action = Variable(record_klen_action)

        record_lstm_output = record_lstm_network(record_mcard_action, record_ktype_action, record_klen_action)
        print(record_lstm_output.shape)

        break
