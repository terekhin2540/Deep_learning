import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from statistics import mean 
import matplotlib.pyplot as plt
from torch.autograd import Variable



if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


text_path = '/kaggle/input/aesops-fables-dataset/Aesops_Fables.txt'



number_of_characters = []
number_of_lines = 0
with open(text_path, 'r') as text:
    for line in text:
        number_of_lines += 1
        for i in line:
            number_of_characters.append(i)

print(f'Number of characters: {len(number_of_characters)}')
print(f'Number of unique characters: {len(set(number_of_characters))}')
print(f'Number of lines: {number_of_lines}')




class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1        
        
        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        
    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt file and generate a 1D PyTorch tensor
# containing the whole text mapped to sequence of token IDs, and a vocab object.
class TextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        self.data, self.vocab = self.text_to_data(file_path, vocab, extend_vocab, device)
        
    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab, extend_vocab, device):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object
          extend_vocab: bool, if True extend the vocab
          device: device

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        data_list = []

        # Construct data
        full_text = []
        print(f"Reading text file from: {text_file}")
        with open(text_file, 'r') as text:
            for line in text:
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

        # convert to tensor
        data = torch.tensor(full_text, device=device, dtype=torch.int64)
        print("Done.")

        return data, vocab
    

# Since there is no need for schuffling the data, we just have to split
# the text data according to the batch size and bptt length.
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class DataBatches:

    def __init__(self, data, bsz, bptt_len, pad_id):
        self.batches = self.create_batch(data, bsz, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, bsz, bptt_len, pad_id):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.
          bsz: int, batch size
          bptt_len: int, bptt length
          pad_id: int, ID of the padding token

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)
        segment_len = text_len // bsz + 1

        # Question: Explain the next two lines!
        padded = input_data.data.new_full((segment_len * bsz,), pad_id)
        padded[:text_len] = input_data.data
        padded = padded.view(bsz, segment_len).t()
        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, bsz), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                batches.append(batch)
                #print(f'i == 0, shape is : {padded[i * bptt_len:(i + 1) * bptt_len].shape}')
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])
                x = padded[i * bptt_len - 1:(i + 1) * bptt_len]
                #print(f'i != 0, shape is : {x.shape}')

        return batches



DEVICE = 'cuda'

batch_size = 32
bptt_len = 64

my_data = TextData(text_path, device=DEVICE)
batches = DataBatches(my_data, batch_size, bptt_len, pad_id=0)


## Model

# RNN based language model
class RNNModel(nn.Module):

    def __init__(self, num_classes, emb_dim, hidden_dim, num_layers):
        """Parameters:
        
          num_classes (int): number of input/output classes
          emb_dim (int): token embedding size
          hidden_dim (int): hidden layer size of RNNs
          num_layers (int): number of RNN layers
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_layer = nn.Embedding(num_classes, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, input, state):
        emb = self.input_layer(input)
        output, state = self.rnn(emb, state)
        output = self.out_layer(output)
        output = output.view(-1, self.num_classes)
        return output, state

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, bsz, self.hidden_dim)


# To be modified for LSTM...
def custom_detach(h):
    return h.detach()



## Decoding
@torch.no_grad()
def complete(model, prompt, steps, sample=False):
    """Complete the prompt for as long as given steps using the model.
    
    Parameters:
      model: language model
      prompt (str): text segment to be completed
      steps (int): number of decoding steps.
      sample (bool): If True, sample from the model. Otherwise greedy.

    Returns:
      completed text (str)
    """
    model.eval()
    out_list = []
    
    # forward the prompt, compute prompt's ppl
    prompt_list = []
    char_prompt = list(prompt)
    for char in char_prompt:
        prompt_list.append(my_data.vocab.string_to_id[char])
    x = torch.tensor(prompt_list).to(DEVICE).unsqueeze(1)
    
    states = model.init_hidden(1)
    logits, states = model(x, states)
    probs = F.softmax(logits[-1], dim=-1)
        
    if sample:
        assert False, "Implement me!"
    else:
        max_p, ix = torch.topk(probs, k=1, dim=-1)

    out_list.append(my_data.vocab.id_to_string[int(ix)])
    x = ix.unsqueeze(1)
    
    # decode 
    for k in range(steps):
        logits, states = model(x, states)
        probs = F.softmax(logits, dim=-1)
        if sample:  # sample from the distribution or take the most likely
            assert False, "Implement me!"
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        out_list.append(my_data.vocab.id_to_string[int(ix)])
        x = ix
    return ''.join(out_list)





learning_rate = 0.0005
clipping = 1.0
embedding_size = 64
rnn_size = 2048
rnn_num_layers = 1

# vocab_size = len(module.vocab.itos)
vocab_size = len(my_data.vocab.id_to_string)
print(F"vocab size: {vocab_size}")

model = RNNModel(
    num_classes=vocab_size, emb_dim=embedding_size, hidden_dim=rnn_size,
    num_layers=rnn_num_layers)
model = model.to(DEVICE)
hidden = model.init_hidden(batch_size)

loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)



# Training

num_epochs = 30
report_every = 30
prompt = "Dogs like best to"
perplexity_loss_epochs = []
perplexity_loss_batchs = []
generation_quality = []

for ep in range(num_epochs):
    print(f"=== start epoch {ep} ===")
    state = model.init_hidden(batch_size)
    for idx in range(len(batches)):
        batch = batches[idx]
        model.train()
        optimizer.zero_grad()
        state = custom_detach(state)
        
        input = batch[:-1]
        target = batch[1:].reshape(-1)

        bsz = input.shape[1]
        prev_bsz = state.shape[1]
        if bsz != prev_bsz:
            state = state[:, :bsz, :]
        output, state = model(input, state)
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()
        
        
        with torch.no_grad():
            perplexity_loss_batchs.append(torch.exp(loss).item())
        if idx % report_every == 0:
            #print(f"train loss: {loss.item()}")  # replace me by the line below!
            print(f"train ppl: {torch.exp(loss).item()}")
            generated_text = complete(model, prompt, 128, sample=False)
            print(f'----------------- epoch/batch {ep}/{idx} -----------------')
            print(prompt)
            print(generated_text)
            print(f'----------------- end generated text -------------------')
            
    if ep == 3:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)
    elif ep == 15:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)   
    elif ep == 29:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)  
        
    perplexity_loss_epochs.append(np.mean(perplexity_loss_batchs))
    perplexity_loss_batchs = []



complete(model, "THE DONKEY IN THE LION’S SKIN", 512, sample=False)


epochs = np.arange(1,31,1)
print(f'len(epochs) - {len(epochs)}')
print(f'len(perplexity_loss) - {len(perplexity_loss_epochs)}')


fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Perplexity plot")
ax.plot(epochs, perplexity_loss_epochs, color='red')

ax.set_xlabel("epochs", fontsize =16)
ax.set_ylabel("Perplexity", fontsize =16)


print(f'Generation quality in the very beginning of the training: \n{generation_quality[0]}')
print('----------------')
print(f'Generation quality in the middle of the training: \n{generation_quality[1]}')
print('----------------')
print(f'Generation quality in the end of the training: \n{generation_quality[2]}')
print('----------------')


## A title of a fable which exists in the book
complete(model, "The Peasant and the Apple Tree", 512, sample=False)

## A title which you invent, which is not in the book, but similar in the style
complete(model, "The Cat and the Friend", 512, sample=False)

## Some texts in a similar style.
complete(model, "The Cat and the Friend", 512, sample=False)

## Anything I think might be interesting
complete(model, "The whale", 512, sample=False)


## 3 Extending the Initial Code

# RNN based language model
class RNNModel(nn.Module):

    def __init__(self, num_classes, emb_dim, hidden_dim, num_layers):
        """Parameters:
        
          num_classes (int): number of input/output classes
          emb_dim (int): token embedding size
          hidden_dim (int): hidden layer size of RNNs
          num_layers (int): number of RNN layers
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_layer = nn.Embedding(num_classes, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, input, state):
        emb = self.input_layer(input)
#         hidden = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).to(DEVICE)
#         cell = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).to(DEVICE)
        output, state = self.lstm(emb, state)
        output = self.out_layer(output)
        output = output.view(-1, self.num_classes)
        return output, state #(hidden_new, cell_new)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, bsz, self.hidden_dim)


# To be modified for LSTM...
def custom_detach(h, c):
    return h.detach(), c.detach()



learning_rate = 0.001
clipping = 1.0
embedding_size = 64
rnn_size = 2048
rnn_num_layers = 1

# vocab_size = len(module.vocab.itos)
vocab_size = len(my_data.vocab.id_to_string)
print(F"vocab size: {vocab_size}")

model = RNNModel(
    num_classes=vocab_size, emb_dim=embedding_size, hidden_dim=rnn_size,
    num_layers=rnn_num_layers)
model = model.to(DEVICE)
hidden = model.init_hidden(batch_size)

loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)



@torch.no_grad()
def complete(model, prompt, steps, sample=False):
    """Complete the prompt for as long as given steps using the model.
    
    Parameters:
      model: language model
      prompt (str): text segment to be completed
      steps (int): number of decoding steps.
      sample (bool): If True, sample from the model. Otherwise greedy.

    Returns:
      completed text (str)
    """
    model.eval()
    out_list = []
    
    # forward the prompt, compute prompt's ppl
    prompt_list = []
    char_prompt = list(prompt)
    for char in char_prompt:
        prompt_list.append(my_data.vocab.string_to_id[char])
    x = torch.tensor(prompt_list).to(DEVICE).unsqueeze(1)
    
    #states = model.init_hidden(1)
    state_hidden = model.init_hidden(1)
    state_cell = model.init_hidden(1)
    
    logits, (state_hidden, state_cell) = model(x, (state_hidden, state_cell))
    probs = F.softmax(logits[-1], dim=-1)
        
    if sample:
        assert False, "Implement me!"
    else:
        max_p, ix = torch.topk(probs, k=1, dim=-1)

    out_list.append(my_data.vocab.id_to_string[int(ix)])
    x = ix.unsqueeze(1)
    
    # decode 
    for k in range(steps):
        logits, (state_hidden, state_cell) = model(x, (state_hidden, state_cell))
        #logits, states = model(x,  states)  # logits, (states, states) = model(x, (states, states))
        probs = F.softmax(logits, dim=-1)
        if sample:  # sample from the distribution or take the most likely
            assert False, "Implement me!"
        else: 
            _, ix = torch.topk(probs, k=1, dim=-1)
        out_list.append(my_data.vocab.id_to_string[int(ix)])
        x = ix
    return ''.join(out_list)



# Training

num_epochs = 30
report_every = 30
prompt = "Dogs like best to"
perplexity_loss_epochs = []
perplexity_loss_batchs = []
generation_quality = []

for ep in range(num_epochs):
    print(f"=== start epoch {ep} ===")
    state = model.init_hidden(batch_size)
    state_hidden = model.init_hidden(batch_size)
    state_cell = model.init_hidden(batch_size)
    for idx in range(len(batches)):
        batch = batches[idx]
        model.train()
        optimizer.zero_grad()
        
#         state_hidden = model.init_hidden(batch_size)
#         state_cell = model.init_hidden(batch_size)
        #state = custom_detach(state)
        
        input = batch[:-1]
        target = batch[1:].reshape(-1)

        bsz = input.shape[1]
        prev_bsz = state.shape[1]
        if bsz != prev_bsz:
            #state = state[:, :bsz, :]
            state_hidden = state_hidden[:, :bsz, :]
            state_cell = state_cell[:, :bsz, :]
        output, (state_hidden, state_cell) = model(input, (state_hidden, state_cell))
        
        state_hidden, state_cell = custom_detach(state_hidden, state_cell)
        #state_cell = custom_detach(state_cell)
        
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()
        
        with torch.no_grad():
            perplexity_loss_batchs.append(torch.exp(loss).item())
            
        if idx % report_every == 0:
            #print(f"train loss: {loss.item()}")  # replace me by the line below!
            print(f"train ppl: {torch.exp(loss).item()}")
            generated_text = complete(model, prompt, 128, sample=False)
            print(f'----------------- epoch/batch {ep}/{idx} -----------------')
            print(prompt)
            print(generated_text)
            print(f'----------------- end generated text -------------------')
            
    if ep == 3:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)
    elif ep == 15:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)   
    elif ep == 29:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)  
        
    perplexity_loss_epochs.append(np.mean(perplexity_loss_batchs))
    perplexity_loss_batchs = []



epochs = np.arange(1,31,1)
print(f'len(epochs) - {len(epochs)}')
print(f'len(perplexity_loss) - {len(perplexity_loss_epochs)}')


print(f'Best best perplexity: {min(perplexity_loss_epochs)}')  # achived perplexity value below 1.03!!!!


fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Perplexity plot")
ax.plot(epochs, perplexity_loss_epochs, color='red')

ax.set_xlabel("epochs", fontsize =16)
ax.set_ylabel("Perplexity", fontsize =16)


complete(model, "THE DONKEY IN THE LION’S SKIN", 512, sample=False)


## Sampling 

@torch.no_grad()
def complete(model, prompt, steps, sample=False):
    """Complete the prompt for as long as given steps using the model.
    
    Parameters:
      model: language model
      prompt (str): text segment to be completed
      steps (int): number of decoding steps.
      sample (bool): If True, sample from the model. Otherwise greedy.

    Returns:
      completed text (str)
    """
    model.eval()
    out_list = []
    
    # forward the prompt, compute prompt's ppl
    prompt_list = []
    char_prompt = list(prompt)
    for char in char_prompt:
        prompt_list.append(my_data.vocab.string_to_id[char])
    x = torch.tensor(prompt_list).to(DEVICE).unsqueeze(1)
    
    #states = model.init_hidden(1)
    state_hidden = model.init_hidden(1)
    state_cell = model.init_hidden(1)
    
    logits, (state_hidden, state_cell) = model(x, (state_hidden, state_cell))
#     print('Logits')
#     print(logits)
#     print(len(logits))
    probs = F.softmax(logits[-1], dim=-1)
#     print('Probs')
#     print(probs)
#     print(len(probs))
        
    if sample:
        #assert False, "Implement me!"
        ix = torch.multinomial(probs, num_samples=1)
    else:
        max_p, ix = torch.topk(probs, k=1, dim=-1)
    
    #print(ix)
    out_list.append(my_data.vocab.id_to_string[int(ix)])
    x = ix.unsqueeze(1)
    
    # decode 
    for k in range(steps):
        logits, (state_hidden, state_cell) = model(x, (state_hidden, state_cell))
        #logits, states = model(x,  states)  # logits, (states, states) = model(x, (states, states))
        probs = F.softmax(logits, dim=-1)
        if sample:  # sample from the distribution or take the most likely
            #assert False, "Implement me!"
            ix = torch.multinomial(probs, num_samples=1)
        else: 
            _, ix = torch.topk(probs, k=1, dim=-1)
        out_list.append(my_data.vocab.id_to_string[int(ix)])
        x = ix
    return ''.join(out_list)


complete(model, "THE DONKEY IN THE LION’S SKIN", 512, sample=False)

complete(model, "THE DONKEY IN THE LION’S SKIN", 512, sample=True)


complete(model, "THE MOUSE, THE CAT, AND THE COCK", 512, sample=False)

complete(model, "THE MOUSE, THE CAT, AND THE COCK", 512, sample=True)


complete(model, "THE DOG, AND THE CHICKEN", 512, sample=False)

complete(model, "THE DOG, AND THE CHICKEN", 512, sample=True)



## Be Creative

text_path = '/kaggle/input/entomology/Entomology.txt'

number_of_characters = []
number_of_lines = 0
with open(text_path, 'r') as text:
    for line in text:
        number_of_lines += 1
        for i in line:
            number_of_characters.append(i)

print(f'Number of characters: {len(number_of_characters)}')
print(f'Number of unique characters: {len(set(number_of_characters))}')
print(f'Number of lines: {number_of_lines}')



class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1        
        
        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        
    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt file and generate a 1D PyTorch tensor
# containing the whole text mapped to sequence of token IDs, and a vocab object.
class TextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        self.data, self.vocab = self.text_to_data(file_path, vocab, extend_vocab, device)
        
    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab, extend_vocab, device):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object
          extend_vocab: bool, if True extend the vocab
          device: device

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        data_list = []

        # Construct data
        full_text = []
        print(f"Reading text file from: {text_file}")
        with open(text_file, 'r') as text:
            for line in text:
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

        # convert to tensor
        data = torch.tensor(full_text, device=device, dtype=torch.int64)
        print("Done.")

        return data, vocab
    

# Since there is no need for schuffling the data, we just have to split
# the text data according to the batch size and bptt length.
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class DataBatches:

    def __init__(self, data, bsz, bptt_len, pad_id):
        self.batches = self.create_batch(data, bsz, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, bsz, bptt_len, pad_id):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.
          bsz: int, batch size
          bptt_len: int, bptt length
          pad_id: int, ID of the padding token

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)
        segment_len = text_len // bsz + 1

        # Question: Explain the next two lines!
        padded = input_data.data.new_full((segment_len * bsz,), pad_id)
        padded[:text_len] = input_data.data
        padded = padded.view(bsz, segment_len).t()
        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, bsz), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                batches.append(batch)
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])

        return batches




DEVICE = 'cuda'

batch_size = 32
bptt_len = 64

my_data = TextData(text_path, device=DEVICE)
batches = DataBatches(my_data, batch_size, bptt_len, pad_id=0)



# RNN based language model
class RNNModel(nn.Module):

    def __init__(self, num_classes, emb_dim, hidden_dim, num_layers):
        """Parameters:
        
          num_classes (int): number of input/output classes
          emb_dim (int): token embedding size
          hidden_dim (int): hidden layer size of RNNs
          num_layers (int): number of RNN layers
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_layer = nn.Embedding(num_classes, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers)
        self.out_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, input, state):
        emb = self.input_layer(input)
#         hidden = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).to(DEVICE)
#         cell = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).to(DEVICE)
        output, state = self.lstm(emb, state)
        output = self.out_layer(output)
        output = output.view(-1, self.num_classes)
        return output, state #(hidden_new, cell_new)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, bsz, self.hidden_dim)


# To be modified for LSTM...
def custom_detach(h, c):
    return h.detach(), c.detach()




learning_rate = 0.001
clipping = 1.0
embedding_size = 64
rnn_size = 2048
rnn_num_layers = 2

# vocab_size = len(module.vocab.itos)
vocab_size = len(my_data.vocab.id_to_string)
print(F"vocab size: {vocab_size}")

model = RNNModel(
    num_classes=vocab_size, emb_dim=embedding_size, hidden_dim=rnn_size,
    num_layers=rnn_num_layers)
model = model.to(DEVICE)
hidden = model.init_hidden(batch_size)

loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)




@torch.no_grad()
def complete(model, prompt, steps, sample=False):
    """Complete the prompt for as long as given steps using the model.
    
    Parameters:
      model: language model
      prompt (str): text segment to be completed
      steps (int): number of decoding steps.
      sample (bool): If True, sample from the model. Otherwise greedy.

    Returns:
      completed text (str)
    """
    model.eval()
    out_list = []
    
    # forward the prompt, compute prompt's ppl
    prompt_list = []
    char_prompt = list(prompt)
    for char in char_prompt:
        prompt_list.append(my_data.vocab.string_to_id[char])
    x = torch.tensor(prompt_list).to(DEVICE).unsqueeze(1)
    
    #states = model.init_hidden(1)
    state_hidden = model.init_hidden(1)
    state_cell = model.init_hidden(1)
    
    logits, (state_hidden, state_cell) = model(x, (state_hidden, state_cell))
#     print('Logits')
#     print(logits)
#     print(len(logits))
    probs = F.softmax(logits[-1], dim=-1)
#     print('Probs')
#     print(probs)
#     print(len(probs))
        
    if sample:
        #assert False, "Implement me!"
        ix = torch.multinomial(probs, num_samples=1)
    else:
        max_p, ix = torch.topk(probs, k=1, dim=-1)
    
    #print(ix)
    out_list.append(my_data.vocab.id_to_string[int(ix)])
    x = ix.unsqueeze(1)
    
    # decode 
    for k in range(steps):
        logits, (state_hidden, state_cell) = model(x, (state_hidden, state_cell))
        #logits, states = model(x,  states)  # logits, (states, states) = model(x, (states, states))
        probs = F.softmax(logits, dim=-1)
        if sample:  # sample from the distribution or take the most likely
            #assert False, "Implement me!"
            ix = torch.multinomial(probs, num_samples=1)
        else: 
            _, ix = torch.topk(probs, k=1, dim=-1)
        out_list.append(my_data.vocab.id_to_string[int(ix)])
        x = ix
    return ''.join(out_list)






# Training

num_epochs = 45
report_every = 60
prompt = "The most detailed and careful experiments are"
perplexity_loss_epochs = []
perplexity_loss_batchs = []
generation_quality = []

for ep in range(num_epochs):
    print(f"=== start epoch {ep} ===")
    state = model.init_hidden(batch_size)
    state_hidden = model.init_hidden(batch_size)
    state_cell = model.init_hidden(batch_size)
    for idx in range(len(batches)):
        batch = batches[idx]
        model.train()
        optimizer.zero_grad()
        
#         state_hidden = model.init_hidden(batch_size)
#         state_cell = model.init_hidden(batch_size)
        #state = custom_detach(state)
        
        input = batch[:-1]
        target = batch[1:].reshape(-1)

        bsz = input.shape[1]
        prev_bsz = state.shape[1]
        if bsz != prev_bsz:
            #state = state[:, :bsz, :]
            state_hidden = state_hidden[:, :bsz, :]
            state_cell = state_cell[:, :bsz, :]
        output, (state_hidden, state_cell) = model(input, (state_hidden, state_cell))
        
        state_hidden, state_cell = custom_detach(state_hidden, state_cell)
        #state_cell = custom_detach(state_cell)
        
        loss = loss_fn(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()
        
        with torch.no_grad():
            perplexity_loss_batchs.append(torch.exp(loss).item())
            
        if idx % report_every == 0:
            #print(f"train loss: {loss.item()}")  # replace me by the line below!
            print(f"train ppl: {torch.exp(loss).item()}")
            generated_text = complete(model, prompt, 128, sample=False)
            print(f'----------------- epoch/batch {ep}/{idx} -----------------')
            print(prompt)
            print(generated_text)
            print(f'----------------- end generated text -------------------')
            
    if ep == 3:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)
    elif ep == 29:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)   
    elif ep == 44:
        full_phase = prompt + generated_text
        generation_quality.append(full_phase)  
        
    perplexity_loss_epochs.append(np.mean(perplexity_loss_batchs))
    perplexity_loss_batchs = []




complete(model, "the muscular system of insects", 512, sample=False)

complete(model, "the sensory organs", 512, sample=False) 

complete(model, "The organs of smell", 512, sample=False)

complete(model, "the digestive canal and its appendages", 512, sample=False) 

print(f'Best best perplexity: {min(perplexity_loss_epochs)}')  




epochs = np.arange(1,46,1)
print(f'len(epochs) - {len(epochs)}')

print(f'len(perplexity_loss) - {len(perplexity_loss_epochs)}')
fig , ax = plt.subplots(figsize=(10, 8))
plt.title("Perplexity plot")
ax.plot(epochs, perplexity_loss_epochs, color='red')

ax.set_xlabel("epochs", fontsize =16)
ax.set_ylabel("Perplexity", fontsize =16)







