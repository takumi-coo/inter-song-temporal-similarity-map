from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from getFeatures import extract_frame_features


@dataclass
class AudioFeatureExtractorConfig:
    feature_dim: int
    frame_length_sec: float


@dataclass
class AudioFeature:
    frame_vectors: torch.Tensor  # (T, D)
    num_frames: int


class FrameFeatureExtractor:
    def __init__(self, config: AudioFeatureExtractorConfig, cache_dir: Optional[str] = None) -> None:
        self.config = config
        self.cache_dir = cache_dir

    def _cache_path(self, audio_file: str) -> str:
        basename_without_ext = os.path.splitext(os.path.basename(audio_file))[0]
        return os.path.join(self.cache_dir, basename_without_ext + ".pt")

    def extract(self, audio_file: str) -> AudioFeature:
        # cache load
        if self.cache_dir is not None:
            cp = self._cache_path(audio_file)
            if os.path.exists(cp):
                try:
                    cached = torch.load(cp, map_location="cpu")
                    if not isinstance(cached, torch.Tensor):
                        cached = torch.tensor(cached, dtype=torch.float)
                    cached = cached.float()
                    return AudioFeature(frame_vectors=cached, num_frames=cached.shape[0])
                except Exception as e:
                    print(f"Warning: Failed to load cache from {cp}: {e}")

        features = extract_frame_features(audio_file, frame_length_sec=self.config.frame_length_sec)
        fv = features["frame_vectors"]
        if not isinstance(fv, torch.Tensor):
            fv = torch.tensor(fv, dtype=torch.float)
        else:
            fv = fv.float()

        af = AudioFeature(frame_vectors=fv, num_frames=fv.shape[0])

        # cache save
        if self.cache_dir is not None:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(af.frame_vectors.cpu(), self._cache_path(audio_file))
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")

        return af



class ContrastiveFeatureDataset(Dataset):
    def __init__(
        self,
        wav_folder: str,
        frame_length_sec: float = 0.5,
        seq_len: int = 32,
        num_negatives: int = 5,
        cache_dir: Optional[str] = None,
        extensions: Tuple[str, ...] = (".wav", ".mp3", ".flac"),
    ) -> None:
        self.seq_len = int(seq_len)
        self.num_negatives = int(num_negatives)

        self.extractor = FrameFeatureExtractor(
            config=AudioFeatureExtractorConfig(feature_dim=66, frame_length_sec=float(frame_length_sec)),
            cache_dir=cache_dir,
        )

        # load all songs -> list of (T, D)
        p = Path(wav_folder)
        files = sorted(
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in set(ext.lower() for ext in extensions)
        )
        print(f"Loading {len(files)} songs from: {wav_folder}")

        self.feature_list: List[torch.Tensor] = []
        self.song_lengths: List[int] = []

        for file in files:
            af = self.extractor.extract(file)
            feat = af.frame_vectors
            if feat.ndim != 2:
                raise ValueError(f"Expected 2D frame_vectors (T,D). Got {feat.shape} from {file}")
            if feat.shape[1] != self.extractor.config.feature_dim:
                raise ValueError(
                    f"Feature dim mismatch: expected {self.extractor.config.feature_dim}, got {feat.shape[1]} ({file})"
                )
            if af.num_frames > 0:
                self.feature_list.append(feat.float())
                self.song_lengths.append(int(af.num_frames))
                print(f"{os.path.basename(file)}: {feat.shape}")

        self.num_songs = len(self.feature_list)
        if self.num_songs < 2:
            raise ValueError("Need at least 2 songs to sample negatives from other songs.")

        self.total_samples = int(sum(self.song_lengths))

    def __len__(self) -> int:
        return self.total_samples

    def _get_padded_slice(self, tensor: torch.Tensor, start_idx: int) -> torch.Tensor:

        max_len = tensor.shape[0]
        end_idx = start_idx + self.seq_len
        if end_idx <= max_len:
            return tensor[start_idx:end_idx]
        slice_data = tensor[start_idx:max_len]
        pad_len = self.seq_len - slice_data.shape[0]
        return F.pad(slice_data, (0, 0, 0, pad_len))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        song_idx = 0
        t = int(idx)
        for sl in self.song_lengths:
            if t < sl:
                break
            t -= sl
            song_idx += 1

        this_song = self.feature_list[song_idx]  # (T,D)

        anchor = self._get_padded_slice(this_song, t)       # (seq_len, D)
        positive = self._get_padded_slice(this_song, t + 1) # (seq_len, D)

        negatives = []
        for _ in range(self.num_negatives):
            neg_song_idx = random.randint(0, self.num_songs - 2)
            if neg_song_idx >= song_idx:
                neg_song_idx += 1

            neg_song = self.feature_list[neg_song_idx]
            neg_len = self.song_lengths[neg_song_idx]
            neg_t = random.randint(0, neg_len - 1)

            neg_seg = self._get_padded_slice(neg_song, neg_t)  # (seq_len, D)
            negatives.append(neg_seg)

        negatives = torch.stack(negatives, dim=0)  # (K, seq_len, D)

        return anchor, positive, negatives
