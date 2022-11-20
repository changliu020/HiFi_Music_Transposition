from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import rnn
import torch
from dataset import NottinghamDataset


def collate_fn(data):
    data.sort(key=lambda d: len(d), reverse=True)
    data_length = [len(d) for d in data]
    data = rnn.pad_sequence(data, batch_first=True, padding_value=0)
    return data.unsqueeze(-1), data_length


dataset = NottinghamDataset('./data/nottingham-dataset/wav')
dataloader = DataLoader(dataset, batch_size=3, collate_fn=collate_fn)
for i, (data, length) in enumerate(dataloader):
    if i > 0:
        break
    print(rnn.pack_padded_sequence(data, length, batch_first=True).data.shape)

model = torch.nn.LSTM(1, 5, batch_first=True)
flag = 0
for i, (data, length) in enumerate(dataloader):
    if i > 0:
        break
    data = rnn.pack_padded_sequence(data, length, batch_first=True)
    output, hidden = model(data)
    if flag == 0:
        output, out_len = rnn.pad_packed_sequence(output, batch_first=True)
        print(output.shape)
        print(output)
        flag = 1
