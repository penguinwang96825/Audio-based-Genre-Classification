import os
import torch
import joblib
import librosa
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn import preprocessing
warnings.filterwarnings("ignore")


class SliceableDict(dict):
    default = None
    def __getitem__(self, key):
        if isinstance(key, list):
            return {k: self.get(k, self.default) for k in key}
            return {k: self[k] for k in key}
            return {k: self[k] for k in key if k in self}
        return dict.get(self, key)


def save_song(song_dirs="./mp3", save_path="dict.bin"):
    song_dict = SliceableDict()
    # Set song ID
    idx = 0
    for root, dirs, files in os.walk(song_dirs):
        for i, file in enumerate(tqdm(files)):
            # Read files
            path = os.path.join(root, file)
            artist, song, genre = file.split("-")
            artist = artist.strip()
            song = song.strip()
            genre = genre.split(".mp3")[0].strip()
            # Instantiate SongInfo() to dictionary
            signal, sr = librosa.load(path)
            info = SongInfo(artist, song, genre, signal)
            song_dict[idx] = info
            # Move to next song ID
            idx += 1
            if i == 1:
                break
    joblib.dump(song_dict, save_path, compress=5)


class SongInfo(object):

    def __init__(self, artist, song, genre, signal):
        assert isinstance(artist, str), "Artist must be string!"
        assert isinstance(song, str), "Song must be string!"
        assert isinstance(genre, str), "Genre must be string!"

        if type(signal).__module__ == np.__name__:
            signal = signal.tolist()

        self.artist = artist
        self.song = song
        self.genre = genre
        self.signal = signal


def load_song(song_dirs="dict.bin"):
    song_dict = joblib.load("dict.bin")
    return song_dict


class SongGenreDataset(Dataset):

    def __init__(self, song_dict):
        self.song_dict = song_dict
        self.mapping = {
            "R&B": 1,
            "LoveSong": 2,
            "ountry": 3
        }

    def __len__(self):
        return len(self.song_dict)

    def __getitem__(self, idx):
        feature = np.array(self.song_dict[[idx]].signal, dtype=np.float)
        target = self.song_dict[[idx]].genre
        target = self.mapping[target]
        return {
            'feature': torch.tensor(feature, dtype=torch.float64),
            'target': torch.tensor(target, dtype=torch.long)}


def main():
    # save_song(save_path="small_dict.bin")
    song_dict = load_song("small_dict.bin")
    print(song_dict)
    dataset = SongGenreDataset(song_dict)
    print(dataset[0])
    print(dataset[:])


if __name__ == '__main__':
    main()
