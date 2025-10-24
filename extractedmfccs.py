# audio_processor.py
import wave
import struct
import numpy as np
import math

class AudioProcessor:
    def __init__(self, audio_path: str = "", frame_length=1024, hop_length=512, num_mfcc=13, num_filters=26, alpha=0.97):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.num_mfcc = num_mfcc
        self.num_filters = num_filters
        self.alpha = alpha
        self.audio_path = audio_path
        self.signal = None
        self.sr = None
        self.duration = None

    def load_audio(self, path=None):
        if path is None:
            path = self.audio_path
        if not path:
            raise ValueError("No audio file path provided")

        with wave.open(path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        
        if sampwidth == 1:
            fmt = "{}B".format(n_frames * n_channels) 
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        elif sampwidth == 2:
            fmt = "{}h".format(n_frames * n_channels)  
            data = np.frombuffer(raw, dtype=np.int16)
        else:
            
            data = np.frombuffer(raw, dtype=np.int16)

        if n_channels > 1:
            data = data.reshape(-1, n_channels)
            data = data.mean(axis=1)  # simple mono mix

        # Normalize to float32 in range [-1, 1]
        max_val = float(2 ** (8 * sampwidth - 1))
        signal = data.astype(np.float32) / max_val

        self.signal = signal
        self.sr = framerate
        self.duration = len(signal) / float(framerate) if framerate > 0 else 0.0
        return signal, framerate

    # helper: hz <-> mel
    def hz_to_mel(self, f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(self, m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _frames(self, signal):
        N = self.frame_length
        H = self.hop_length
        if len(signal) < N:
            # pad
            pad_width = N - len(signal)
            signal = np.concatenate([signal, np.zeros(pad_width, dtype=signal.dtype)])
        frames = []
        for i in range(0, len(signal) - N + 1, H):
            frame = signal[i:i+N]
            window = np.hanning(N)
            frames.append(frame * window)
        frames = np.stack(frames).astype(np.float32)  # shape (num_frames, N)
        return frames

    def _rfft_power(self, frames):
        # compute power spectrum for each framed window
        N = self.frame_length
        fft = np.fft.rfft(frames, n=N)  # shape (num_frames, n_fft_bins)
        power = (np.abs(fft) ** 2) / N
        return power

    def extract_mfcc(self, signal, sr):
        # 1) basic pre-emphasis
        if signal.size == 0:
            return np.zeros((self.num_mfcc, 0), dtype=np.float32)
        sig = signal.astype(np.float32)
        emphasized = np.empty_like(sig)
        emphasized[0] = sig[0]
        emphasized[1:] = sig[1:] - self.alpha * sig[:-1]

        # 2) framing & windowing
        frames = self._frames(emphasized)  # (T, N)
        if frames.shape[0] == 0:
            return np.zeros((self.num_mfcc, 0), dtype=np.float32)

        # 3) power spectrum
        power_spec = self._rfft_power(frames)  # (T, n_fft_bins)
        n_fft_bins = power_spec.shape[1]

        # 4) mel filterbank
        min_freq, max_freq = 0, sr / 2.0
        mel_min, mel_max = self.hz_to_mel(min_freq), self.hz_to_mel(max_freq)
        mel_points = np.linspace(mel_min, mel_max, self.num_filters + 2)
        hz_points = self.mel_to_hz(mel_points)
        # bin indices
        bin_indices = np.floor((self.frame_length + 1) * hz_points / sr).astype(int)
        # clamp
        bin_indices = np.clip(bin_indices, 0, n_fft_bins - 1)

        filterbank = np.zeros((self.num_filters, n_fft_bins), dtype=np.float32)
        for i in range(1, self.num_filters + 1):
            left = bin_indices[i - 1]
            center = bin_indices[i]
            right = bin_indices[i + 1]
            if center - left > 0:
                rising = (np.arange(left, center) - left) / float(center - left)
                filterbank[i-1, left:center] = rising
            if right - center > 0:
                falling = (right - np.arange(center, right)) / float(right - center)
                filterbank[i-1, center:right] = falling

        # apply filterbank
        mel_energies = np.dot(power_spec, filterbank.T)  # shape (T, num_filters)
        # numerical stability
        mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
        log_mel = np.log(mel_energies)

        # 5) DCT 
        def dct_matrix(K, N):
            n = np.arange(N)
            k = np.arange(K)[:, None]
            basis = np.cos(np.pi * (2*n + 1) * k / (2.0 * N))
            scale = np.sqrt(2.0 / N) * ((k != 0).astype(np.float32) * 1.0 + (k == 0).astype(np.float32) * (1.0 / np.sqrt(2.0)))
            return scale * basis

        Nmel = self.num_filters
        K = self.num_mfcc
        D = dct_matrix(K, Nmel)  # (K, Nmel)
        # log_mel: (T, Nmel)
        mfcc = (D @ log_mel.T)  # (K, T)

        # 6) mean/var normalization per utterance
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        std[std == 0] = 1.0
        mfcc = (mfcc - mean) / std

        return mfcc.astype(np.float32)  # (num_mfcc, T)
