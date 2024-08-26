from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import random
import torch
import torch.nn as nn
from torch.nn import Parameter
from enum import IntEnum
import time
import math
import unicodedata
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.optim as optim

def findFiles(path): return glob.glob(path)

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('./data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)



# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class MyLSTM(nn.Module):

    def __init__(self, input_size, hidden_size,num_classes):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.out = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

        # 输入门i_t
        self.W_i = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))
        # 遗忘门f_t
        self.W_f = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        # 候选内部状态g_t
        self.W_g = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_g = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))
        # 输出门o_t
        self.W_o = Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        # 初始化参数
        self._initialize_weights()
    
    def _initialize_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def initHidden(self, x):
        h_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        c_t = torch.zeros(1, x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        return h_t, c_t

    def forward(self, x, init_states=None):

        batch_size, seq_size, _ = x.size()
        hidden_seq = []

        # 状态初始化
        if init_states is None:
            h_t, c_t = self.initHidden(x)
        else:
            h_t, c_t = init_states

        for t in range(seq_size):
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t @ self.W_i + h_t @ self.U_i.T + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_f + h_t @ self.U_f.T + self.b_f)
            g_t = torch.tanh(x_t @ self.W_g + h_t @ self.U_g.T + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_o + h_t @ self.U_o.T + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t)

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()

        output = self.out(hidden_seq)
        output = self.softmax(output)
        return output, h_t, c_t
n_hidden = 64
lstm=MyLSTM(n_letters, n_hidden, n_categories)
print(lstm)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

criterion = nn.NLLLoss()
learning_rate = 0.0001 # If you set this too high, it might explode. If too low, it might not learn
optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)

# lstm
def train(category_tensor, line_tensor):
    h0, c0 = lstm.initHidden(line_tensor)

    lstm.zero_grad()
    optimizer.zero_grad()

    output, h, c = lstm(line_tensor, (h0, c0))

    loss = criterion(output[-1], category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.item()

n_iters = 300000
# print_every = 5000
print_every = 5000
plot_every = 1000



# Keep track of losses for plotting
current_loss = 0
all_losses = []

correct_count = 0
all_accuracy = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    guess, guess_i = categoryFromOutput(output[-1])
    # guess, guess_i = categoryFromOutput(output)
    correct = '✓' if guess == category else '✗ (%s)' % category
    if correct == '✓':
        correct_count += 1

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
        
        accuracy = correct_count / print_every * 100
        print('%d %d%% (%s) %.4f %s / %s %s Accuracy: %.2f%%' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct, accuracy))
        all_accuracy.append(accuracy)
        correct_count = 0  # 重置正确计数

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0



plt.figure()
plt.plot(all_losses)

plt.figure()
plt.plot(all_accuracy)

# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

# Just return an output given a line
def evaluate(line_tensor):
    h0, c0 = lstm.initHidden(line_tensor)

#     for i in range(line_tensor.size()[0]):
    output, h, c = lstm(line_tensor, (h0, c0))

    return output[-1]

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# Normalize by dividing every row by its sum
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()