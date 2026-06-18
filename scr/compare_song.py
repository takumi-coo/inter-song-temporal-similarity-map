
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from getFeatures import extract_frame_features

from model import LSTMEncoder

# ==== 設定 ====
INPUT_LENGTH = 4
HIDDEN_DIM = 128
OUTPUT_DIM = 64
MODEL_PATH = ""
AUDIO_DIR = ""
N_CLUSTERS = 4


def extract_feature_tensor(filepath):
    try:
        features = extract_frame_features(filepath)["frame_vectors"]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        print(f"[Error in {filepath}]: {e}")
        return None

def create_frame_sequences(tensor, length):
    tensor = tensor.squeeze(0)
    sequences = []
    for i in range(length - 1, len(tensor)):
        seq = tensor[i - length + 1:i + 1].unsqueeze(0)
        sequences.append(seq)
    return sequences

def get_segment_embeddings(model, device, file_path):
    feat = extract_feature_tensor(file_path)
    if feat is None:
        return None, None

    sequences = create_frame_sequences(feat, INPUT_LENGTH)
    embeddings = []
    for seq in sequences:
        with torch.no_grad():
            emb = model(seq.to(device)).cpu().squeeze().numpy()
            embeddings.append(emb)

    embeddings = np.array(embeddings)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    segment_means = []
    for i in range(N_CLUSTERS):
        seg_embs = embeddings[cluster_labels == i]
        if len(seg_embs) > 0:
            seg_mean = np.mean(seg_embs, axis=0)
            segment_means.append(seg_mean)
        else:
            segment_means.append(np.zeros((embeddings.shape[1],)))
    return segment_means, os.path.basename(file_path)


def visualize_similarity(sim_matrix, song_names):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(sim_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Cosine Similarity")
    plt.xticks(ticks=np.arange(len(song_names)), labels=song_names, rotation=45)
    plt.yticks(ticks=np.arange(len(song_names)), labels=song_names)
    plt.title("Segment-level Cosine Similarity Between Songs")
    plt.tight_layout()
    plt.show()

# ---- メイン処理 ----
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMEncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_segment_vectors = []
    song_names = []

    for file in sorted(os.listdir(AUDIO_DIR)):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(AUDIO_DIR, file)
        segment_means, name = get_segment_embeddings(model, device, path)
        if segment_means is not None:
            all_segment_vectors.append(segment_means)
            song_names.append(name)

    num_songs = len(all_segment_vectors)
    sim_matrix = np.zeros((num_songs, num_songs))

    for i in range(num_songs):
        for j in range(num_songs):
            sim = cosine_similarity(
                np.array(all_segment_vectors[i]),
                np.array(all_segment_vectors[j])
            )
            sim_matrix[i, j] = np.mean(sim)

    visualize_similarity(sim_matrix, song_names)

if __name__ == "__main__":
    main()
