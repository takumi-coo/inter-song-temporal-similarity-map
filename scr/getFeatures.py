import librosa
import numpy as np

def extract_frame_features(
    audio,
    sr=44100,
    frame_length_sec=0.5,
    normalize='zscore',   # 'zscore' | 'minmax' | None
    eps=1e-8
):
    # === Load ===
    y, sr = librosa.load(audio, sr=sr)

    frame_length = int(frame_length_sec * sr)
    hop_length = frame_length

    # === Energy / RMS ===
    energy = np.array([
        np.sum(np.square(y[i:i + frame_length]))
        for i in range(0, len(y) - frame_length + 1, hop_length)
    ])
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    # === STFT ===
    stft = np.abs(librosa.stft(
        y,
        n_fft=frame_length,
        hop_length=hop_length
    ))

    # === MFCC 系 ===
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        hop_length=hop_length
    )
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # === Chroma ===
    chroma = librosa.feature.chroma_stft(
        S=stft,
        sr=sr
    )

    # === フレーム数調整 ===
    final_num_frames = min(
        energy.shape[0],
        rms.shape[0],
        mfcc.shape[1],
        chroma.shape[1]
    )

    # === Frame vectors ===
    frame_vectors = []
    for t in range(final_num_frames):
        vec = (
            list(mfcc[:, t]) +
            list(delta_mfcc[:, t]) +
            list(delta2_mfcc[:, t]) +
            list(chroma[:, t]) +
            [energy[t], rms[t]]
        )
        frame_vectors.append(vec)

    X = np.asarray(frame_vectors, dtype=np.float32)  # (T, 53)

    # === Normalization ===
    norm_stats = {}
    if normalize == 'zscore':
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        X = (X - mu) / (std + eps)
        norm_stats = {
            'type': 'zscore',
            'mean': mu.squeeze().tolist(),
            'std': std.squeeze().tolist(),
            'eps': eps
        }

    elif normalize == 'minmax':
        x_min = X.min(axis=0, keepdims=True)
        x_max = X.max(axis=0, keepdims=True)
        X = (X - x_min) / (x_max - x_min + eps)
        norm_stats = {
            'type': 'minmax',
            'min': x_min.squeeze().tolist(),
            'max': x_max.squeeze().tolist(),
            'eps': eps
        }

    features = {
        'frame_vectors': X.tolist(),
        'num_frames': final_num_frames,
        'norm_stats': norm_stats
    }
    return features
