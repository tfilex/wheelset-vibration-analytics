import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pywt


class RULDataset(Dataset):
    """
    Dataset for Remaining Useful Life (RUL) prediction using XJTU-SY bearing data.
    Reads CSV files chronologically, extracts a signal window, and applies CWT.
    """

    def __init__(self, data_dir, seq_length=5, window_size=1024, cwt_widths=None):
        """
        Args:
            data_dir (str): Path to the bearing directory (e.g. 'data/raw/XJTU-SY/35Hz12kN/Bearing1_1').
            seq_length (int): Number of consecutive files to form a sequence.
            window_size (int): Number of samples to take from each CSV file.
            cwt_widths (array-like): Widths for the Continuous Wavelet Transform.
        """
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.window_size = window_size

        if cwt_widths is None:
            # Default widths for CWT, yielding 32 frequency bands
            self.cwt_widths = np.arange(1, 33)
        else:
            self.cwt_widths = cwt_widths

        # Get all CSV files and sort them chronologically (natural sort)
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        files.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        self.file_paths = [os.path.join(data_dir, f) for f in files]

        self.total_files = len(self.file_paths)
        if self.total_files < self.seq_length:
            raise ValueError(f"Not enough files ({self.total_files}) for sequence length {
                             self.seq_length}.")

    def __len__(self):
        # We can form sequences starting from index 0 up to total_files - seq_length
        return self.total_files - self.seq_length + 1

    def _process_file(self, file_path):
        """Read a CSV file, extract a window, and compute CWT for both channels."""
        # Read the CSV file.
        # XJTU-SY format: Horizontal_vibration_signals, Vertical_vibration_signals
        df = pd.read_csv(file_path)

        # Take the first 'window_size' samples
        if len(df) < self.window_size:
            # Pad if for some reason it's smaller
            h_sig = np.pad(df.iloc[:, 0].values,
                           (0, self.window_size - len(df)))
            v_sig = np.pad(df.iloc[:, 1].values,
                           (0, self.window_size - len(df)))
        else:
            h_sig = df.iloc[:self.window_size, 0].values
            v_sig = df.iloc[:self.window_size, 1].values

        # Apply CWT using PyWavelets Mexican hat wavelet ('mexh' is equivalent to ricker)
        cwt_h, _ = pywt.cwt(h_sig, self.cwt_widths, 'mexh')
        cwt_v, _ = pywt.cwt(v_sig, self.cwt_widths, 'mexh')

        # Shape of cwt_h: (len(cwt_widths), window_size)
        # Stack channels to get (channels, height, width) -> (2, num_widths, window_size)
        scalogram = np.stack([cwt_h, cwt_v], axis=0)

        return scalogram

    def __getitem__(self, idx):
        # The sequence of files is from idx to idx + seq_length - 1
        seq_paths = self.file_paths[idx: idx + self.seq_length]

        scalograms = []
        for path in seq_paths:
            scalo = self._process_file(path)
            scalograms.append(scalo)

        # Shape: (seq_length, channels=2, num_widths, window_size)
        scalograms = np.stack(scalograms, axis=0)

        # Calculate normalized RUL in [0, 1].
        # RUL = 1.0 means the beginning of the bearing lifetime, and
        # RUL = 0.0 means the last available measurement before failure.
        # Target reflects the state at the *end* of the observed sequence.
        current_step = idx + self.seq_length - 1
        total_steps = self.total_files - 1
        rul = float(np.clip(1.0 - (current_step / total_steps), 0.0, 1.0))

        # Convert to torch tensors
        x = torch.tensor(scalograms, dtype=torch.float32)
        y = torch.tensor([rul], dtype=torch.float32)

        return x, y
