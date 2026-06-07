import librosa
import numpy as np
import scipy.signal
from typing import Optional


class Wav2LipAudio:
    MEL_BASIS = None
    _mel_basis_cache = {}

    @classmethod
    def _build_mel_basis(cls, sr: int = 16000, n_fft: int = 800, n_mels: int = 80, fmin: float = 55.0, fmax: Optional[float] = None):
        key = (sr, n_fft, n_mels, fmin, fmax)
        if key not in cls._mel_basis_cache:
            cls._mel_basis_cache[key] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        return cls._mel_basis_cache[key]

    @classmethod
    def melspectrogram(cls, y: np.ndarray, sr: int = 16000, n_fft: int = 800, hop_length: int = 160, win_length: int = 800, n_mels: int = 80, fmin: float = 55.0, fmax: Optional[float] = None) -> np.ndarray:
        if fmax is None:
            fmax = float(sr // 2)

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=scipy.signal.windows.hann(win_length, sym=False))
        mag = np.abs(D)
        mel_basis = cls._build_mel_basis(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel = mel_basis @ mag
        mel = np.log10(np.maximum(mel, 1e-10))
        return mel.T

    @classmethod
    def load_and_preprocess(cls, audio_path: str, sr: int = 16000) -> np.ndarray:
        wav, _ = librosa.load(audio_path, sr=sr, mono=True)
        return wav

    @classmethod
    def get_chunk_energy(cls, mel_chunk: np.ndarray) -> float:
        return float(np.mean(np.power(10.0, mel_chunk)))

    @classmethod
    def get_mel_chunks(cls, audio_path: str, fps: int = 25, mel_step_length: int = 16, mel_step_stride: int = None):
        wav = cls.load_and_preprocess(audio_path)
        mel = cls.melspectrogram(wav)
        if mel_step_stride is None:
            hop_time = 160 / 16000
            video_frame_time = 1.0 / fps
            mel_step_stride = max(1, int(round(video_frame_time / hop_time)))
        mel_chunks = []
        i = 0
        while True:
            start_idx = int(i * fps / 25 * mel_step_stride)
            if start_idx + mel_step_length > len(mel):
                break
            mel_chunks.append(mel[start_idx: start_idx + mel_step_length, :])
            i += 1
        return mel_chunks, mel
