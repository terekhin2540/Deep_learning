import os
import math
import time
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)


print(torch.__version__)


train_x = '/kaggle/input/numbers-place-values/numbers__place_value/train.x'
train_y = '/kaggle/input/numbers-place-values/numbers__place_value/train.y'
interpolate_x = '/kaggle/input/numbers-place-values/numbers__place_value/interpolate.x'
interpolate_y = '/kaggle/input/numbers-place-values/numbers__place_value/interpolate.y'


from statistics import mean

def analitics(path):
    number_of_lines = 0
    avarage_characters_in_line = []
    number_of_characters_in_line = []
    number_of_characters = []
    with open(path, 'r') as text:
        for line in text:
            number_of_lines += 1
            for i in line:
                if i != '\n':
                    number_of_characters_in_line.append(i)
                    number_of_characters.append(i)
                else:
                    pass
            avarage_characters_in_line.append(len(number_of_characters_in_line))
            number_of_characters_in_line = []
        print(f'Number of sentences/lines: {number_of_lines}')
        print(f'Number characters: {len(number_of_characters)}')
        print(f'Average lengh of the line: {mean(avarage_characters_in_line)}')


## I deleted sign \n everywhere because we dont use it in the training model



print(interpolate_x.split('/')[-2])
print()
print(interpolate_x.split('/')[-1])
analitics(interpolate_x)
print()
print(interpolate_y.split('/')[-1])
analitics(interpolate_y)
print()
print(train_x.split('/')[-1])
analitics(train_x)
print()
print(train_y.split('/')[-1])
analitics(train_y)




class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>', sos_token='<sos>'):
        self.id_to_string = {}
        self.string_to_id = {}

        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0

        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1

        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt file and generate a 1D pytorch tensor
# containing the whole text mapped to sequence of token ID,
# and a vocab file
class ParallelTextDataset(Dataset):

    def __init__(self, src_file_path, trg_file_path, src_vocab=None,
                 trg_vocab=None, extend_vocab=False, device='cuda'):
        (self.data, self.src_vocab, self.trg_vocab,
         self.src_max_seq_length, self.tgt_max_seq_length) = self.parallel_text_to_data(
            src_file_path, trg_file_path, src_vocab, trg_vocab, extend_vocab, device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def parallel_text_to_data(self, src_file, tgt_file, src_vocab=None, tgt_vocab=None,
                              extend_vocab=False, device='cuda'):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        if src_vocab is None:
            src_vocab = Vocabulary()

        if tgt_vocab is None:
            tgt_vocab = Vocabulary()

        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = list(line)
                length = len(tokens)
                if src_max < length:
                    src_max = length

        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = list(line)
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
        tgt_max += 2  # add for begin/end tokens

        src_pad_idx = src_vocab.pad_id
        tgt_pad_idx = tgt_vocab.pad_id

        tgt_eos_idx = tgt_vocab.eos_id
        tgt_sos_idx = tgt_vocab.sos_id

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)
                for token in tokens:
                    seq.append(src_vocab.get_idx(token, extend_vocab=extend_vocab))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
                new_seq[:var_len] = var_seq
                src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)
                # append a start token
                seq.append(tgt_sos_idx)
                for token in tokens:
                    seq.append(tgt_vocab.get_idx(token, extend_vocab=extend_vocab))
                # append an end token
                seq.append(tgt_eos_idx)

                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

                # padding
                new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
                new_seq[:var_len] = var_seq
                tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        print("Done.")

        return data_list, src_vocab, tgt_vocab, src_max, tgt_max





# `DATASET_DIR` should be modified to the directory where you downloaded the dataset.
DATASET_DIR = "/kaggle/input/numbers-place-values/"

TRAIN_FILE_NAME = "train"
VALID_FILE_NAME = "interpolate"

INPUTS_FILE_ENDING = ".x"
TARGETS_FILE_ENDING = ".y"

TASK = "numbers__place_value"
# TASK = "comparison__sort"
# TASK = "algebra__linear_1d"

# Adapt the paths!

src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
trg_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

train_set = ParallelTextDataset(src_file_path, trg_file_path, extend_vocab=True)

# get the vocab
src_vocab = train_set.src_vocab
trg_vocab = train_set.trg_vocab

src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
trg_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"

valid_set = ParallelTextDataset(
    src_file_path, trg_file_path, src_vocab=src_vocab, trg_vocab=trg_vocab,
    extend_vocab=False)



########
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        assert x.size(0) < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class TransformerModel(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size,
                 d_model=256, pad_id=0, encoder_layers=3, decoder_layers=2,
                 dim_feedforward=1024, num_heads=8):
        # all arguments are (int)
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads

        self.embedding_src = nn.Embedding(
            source_vocabulary_size, d_model, padding_idx=pad_id)
        self.embedding_tgt = nn.Embedding(
            target_vocabulary_size, d_model, padding_idx=pad_id)

        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model, num_heads, encoder_layers, decoder_layers, dim_feedforward) #
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder
        self.linear = nn.Linear(d_model, target_vocabulary_size)

    def create_src_padding_mask(self, src):
        # input src of shape ()
        src_padding_mask = src.transpose(0, 1) == 0
        return src_padding_mask

    def create_tgt_padding_mask(self, tgt):
        # input tgt of shape ()
        tgt_padding_mask = tgt.transpose(0, 1) == 0
        return tgt_padding_mask




    def greedy_search(self, src, trg_vocab,trg, batch_size = 64 ):
        trg = trg.transpose(0,1)
        src_length = src.shape[0]
        trg_length = trg.shape[1]
        output_seq = []

        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)

        src_mask = torch.zeros((src_length,src_length)).type(torch.bool).to(DEVICE)

        memory_key_padding_mask = src_key_padding_mask.clone()

        src_embed = self.embedding_src(src)
        src_embed = self.pos_encoder(src_embed)

        encoder_hidden = self.transformer.encoder(src_embed, src_mask, src_key_padding_mask).to(DEVICE)

        input_tensor = torch.ones(src.shape[1], 1).fill_(trg_vocab.sos_id).type(torch.int).to(DEVICE)

        for i in range(trg_length):
            tgt_key_padding_mask = self.create_tgt_padding_mask(input_tensor).to(DEVICE)
            tgt_key_padding_mask = tgt_key_padding_mask.transpose(0,1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_tensor.shape[1]).to(DEVICE)

            tgt = self.embedding_tgt(input_tensor)
            tgt = self.pos_encoder(tgt)
            tgt = tgt.transpose(0,1)

            output = self.transformer.decoder(tgt, encoder_hidden,
                                              tgt_mask = tgt_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask)

            prediction = self.linear(output)
            prediction = torch.argmax(prediction[:, -1], keepdim=True)
            output_seq.append(prediction.item())
            if prediction.item() == trg_vocab.eos_id:
                break
            if len(output_seq) >= len(tgt):
                break
            input_tensor = torch.cat([input_tensor, prediction], dim=1)
        return output_seq


    # Implement me!
    def forward_separate(self, src, tgt):

        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.num_heads, self.dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.encoder_layers)


        decoder_layer = nn.TransformerDecoderLayer(self.d_model, self.nhead, self.dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.decoder_layers)


        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)

        out = self.encoder(out, src_key_padding_mask = src_key_padding_mask)

        out = self.decoder(tgt, out, tgt_mask = tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)


        out = self.linear(out)
        return out



    def forward(self, src, tgt):
        """Forward function.

        Parameters:
          src: tensor of shape (sequence_length, batch, data dim)
          tgt: tensor of shape (sequence_length, batch, data dim)
        Returns:
          tensor of shape (sequence_length, batch, data dim)
        """
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)


        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)

        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)


        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)

        out = self.transformer(
            out, tgt, src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        out = self.linear(out)
        return out



def accuracy(predicted, tgt):
    num_correct = sum([1 for p, t in zip(predicted, tgt) if (p == t).all()])
    return num_correct / len(predicted)



batch_size = 64

train_data_loader = DataLoader(
    dataset=train_set, batch_size=batch_size, shuffle=True)

valid_data_loader = DataLoader(
    dataset=valid_set, batch_size=batch_size, shuffle=False)

source_vocabulary_size = len(src_vocab.id_to_string)
target_vocabulary_size = len(trg_vocab.id_to_string)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=trg_vocab.pad_id)


model = TransformerModel(source_vocabulary_size, target_vocabulary_size).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



iteration = 0
valid_loss = []
training_accuracy = []
valid_acc = []
loss_iterations = []
loss_iterations_1000 = []


# while True:
for _ in range(10000):
    print(f"=== start iteration {iteration} ===")
    losses = 0
    gradient_accumulation_time = 1
    for batch in train_data_loader:
        model.train()
        src, trg = batch[0].to(DEVICE), batch[1].to(DEVICE)

        trg_input = trg[:, :-1]

        src = src.transpose(0,1)
        trg_input = trg_input.transpose(0,1)

        prediction = model(src, trg_input)
        pred_new = prediction.view(-1, pred.shape[-1])
        trg_new = trg[:, 1:].reshape(-1)
        loss = loss_fn(pred_new, trg_new)
        loss.backward()
        loss_iterations.append(loss.item())
        losses += loss.item()

        if iteration % 1000 == 0:
            model.eval()
            with torch.no_grad():
                loss_iterations_1000.append(np.mean(loss_iterations))
                loss_iterations = []
                #print('we were here')
                #                 src = src.flatten()
                #                 trg = trg.flatten()
                output = model.greedy_search(src, trg_vocab, trg)
                accur = accuracy(output, trg)
                #                 print(f"acc: {acc}")
                training_accuracy.append(accur)
                print("Iteration: ", iteration, " Train accuracy : ", accur, " Loss : ", loss.item())
                loss_eval = 0
                accur = 0

                for batch_eval in valid_data_loader:
                    src, trg = batch_eval[0].to(DEVICE), batch_eval[1].to(DEVICE)
                    trg_input = trg[:, :-1]
                    src = src.transpose(0,1)
                    trg_input = trg_input.transpose(0,1)
                    pred = model(src, trg_input)
                    output = model.greedy_search(src, trg_vocab, trg)
                    accur += accuracy(output, trg)

                valid_acc.append(accur / len(valid_data_loader))
                valid_loss.append(loss_eval / len(valid_data_loader))

            print("Iteration: ", iteration, " Valid accuracy : ", accur / len(valid_data_loader))
            print()

        if gradient_accumulation_time % 50 == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()


        iteration += 1
        gradient_accumulation_time += 1



print(len(loss_iterations_1000))
print(loss_iterations_1000)



epochs = np.arange(0,len(loss_iterations_1000))
print(f'epochs: - {len(epochs)}')
print(f'len(epochs) - {len(loss_iterations_1000)}')
print(f'len(training_accuracy) - {len(training_accuracy)}')
print(f'len(valid_acc) - {len(valid_acc)}')
# print(f'len(training_accuracy) - {len(training_accuracy)}')
# print(f'len(validation_accuracy) - {len(validation_accuracy)}')



fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Loss plot")
ax.plot(epochs * 1000, loss_iterations_1000, color='red')
#ax.plot(epochs, validation_loss, color='red')

ax.set_xlabel("epochs", fontsize =16)
ax.set_ylabel("Loss", fontsize =16)



# fig , ax = plt.subplots(figsize=(10, 8))
# plt.title("Accuracy plot")
# ax.plot(epochs, training_accuracy, color='red')
# ax.plot(epochs, validation_accuracy, color='red')
#
# ax.set_xlabel("epochs", fontsize =16)
# ax.set_ylabel("Accuracy", fontsize =16)













