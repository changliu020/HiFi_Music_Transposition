import os
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchaudio


class NottinghamDataset(Dataset):
    def __init__(self, data_dir, target_sample_rate, transformation=None):
        self.data_dir = data_dir
        self.name_list = os.listdir(self.data_dir)
        self.target_sample_rate = target_sample_rate
        # self.num_samples = num_samples
        if transformation == 'mel_spectrogram':
            self.transformation = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )
        else:
            self.transformation = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        signal, sr = torchaudio.load(os.path.join(
            self.data_dir, self.name_list[idx]))

        if sr != self.target_sample_rate:
            signal = torchaudio.functional.resample(
                signal, orig_freq=sr, new_freq=self.target_sample_rate)

        if signal.shape[0] > 1:                     # turn stereo to mono
            signal = signal.mean(0)

        # if signal.shape[1] > self.num_samples:      # cut
        #     signal = signal[:, :self.num_samples]
        # elif signal.shape[1] < self.num_samples:    # pad
        #     signal = F.pad(signal, (0, self.num_samples - signal.shape[1]))

        if self.transformation:
            signal = self.transformation(signal)
            signal = signal.transpose(-1, -2)

        return signal


if __name__ == '__main__':
    dataset = NottinghamDataset('./data/nottingham-dataset/wav', target_sample_rate=22050,
                                transformation='mel_spectrogram')
    print(dataset[0].shape)
    print(dataset[1].shape)
    print(dataset[2].shape)
