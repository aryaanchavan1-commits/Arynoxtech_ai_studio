import cv2
import numpy as np
from scipy.fftpack import dct
from scipy.ndimage import gaussian_filter


def _dct_features(frame: np.ndarray, patch_size: int = 8) -> np.ndarray:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    feats = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = gray[y:min(y + patch_size, h), x:min(x + patch_size, w)]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            coeffs = dct(dct(patch, axis=0, norm='ortho'), axis=1, norm='ortho')
            feats.append(coeffs[:3, :3].flatten())
    return np.concatenate(feats)


class TemporalAttentionProcessor:
    def __init__(self, window_size: int = 15, temperature: float = 0.5):
        if window_size % 2 == 0:
            window_size += 1
        self.window_size = window_size
        self.temperature = temperature

    def process(self, frames):
        n = len(frames)
        half = self.window_size // 2
        features = [_dct_features(f) for f in frames]
        features = np.array(features)
        feat_norm = np.linalg.norm(features, axis=1, keepdims=True)
        feat_norm = np.where(feat_norm > 1e-8, feat_norm, 1.0)
        features = features / feat_norm

        result = []
        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window_feats = features[start:end]
            sims = window_feats @ features[i]
            sims = np.exp(sims / self.temperature)
            sims = sims / (sims.sum() + 1e-8)
            blended = np.zeros_like(frames[i], dtype=np.float32)
            for w, idx in zip(sims, range(start, end)):
                blended += w * frames[idx].astype(np.float32)
            result.append(np.clip(blended, 0, 255).astype(np.uint8))
        return result


class Latent3DProcessor:
    def __init__(self, latent_scale: int = 4, temporal_sigma: float = 1.0, spatial_sigma: float = 0.5):
        self.latent_scale = latent_scale
        self.temporal_sigma = temporal_sigma
        self.spatial_sigma = spatial_sigma

    def process(self, frames, chunk_size: int = 16):
        n = len(frames)
        h, w = frames[0].shape[:2]
        lh = max(1, h // self.latent_scale)
        lw = max(1, w // self.latent_scale)
        result = [None] * n

        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            chunk = frames[chunk_start:chunk_end]
            T = len(chunk)
            latent = np.zeros((T, lh, lw, 3), dtype=np.float32)
            for t in range(T):
                latent[t] = cv2.resize(chunk[t].astype(np.float32), (lw, lh))

            for c in range(3):
                latent_c = latent[:, :, :, c]
                for t in range(T):
                    latent_c[t] = gaussian_filter(latent_c[t], sigma=self.spatial_sigma)
                latent_c = gaussian_filter(latent_c, sigma=(self.temporal_sigma, 0, 0))

            for t in range(T):
                up = cv2.resize(latent[t], (w, h))
                up = np.clip(up, 0, 255).astype(np.uint8)
                result[chunk_start + t] = up

        return result


def temporal_refine(frames, window_size: int = 15, latent_scale: int = 4, temperature: float = 0.5):
    attn = TemporalAttentionProcessor(window_size=window_size, temperature=temperature)
    frames = attn.process(frames)
    latent = Latent3DProcessor(latent_scale=latent_scale)
    frames = latent.process(frames)
    return frames
