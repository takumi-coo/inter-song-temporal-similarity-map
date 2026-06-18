import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LSTMEncoder
from dataset import ContrastiveFeatureDataset

# -------------------------------
# Checkpoint Helper Functions
# -------------------------------

def save_checkpoint(model, optimizer, epoch, folder="checkpoints", filename="checkpoint_latest.pth"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    torch.save(checkpoint, os.path.join(folder, f"checkpoint_epoch_{epoch}.pth"))
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, folder="checkpoints", checkpoint_id=None):
    if not os.path.exists(folder):
        print(f"No checkpoint folder found at '{folder}'. Starting from scratch.")
        return 0

    target_path = ""

    if isinstance(checkpoint_id, int):
        target_path = os.path.join(folder, f"checkpoint_epoch_{checkpoint_id}.pth")
        if not os.path.exists(target_path):
            print(f"Checkpoint id {checkpoint_id} not found. Starting from scratch.")
            return 0

    elif checkpoint_id is None or checkpoint_id == "latest":
        files = glob.glob(os.path.join(folder, "checkpoint_epoch_*.pth"))
        if not files:
            print("No checkpoint files found. Starting from scratch.")
            return 0
        try:
            target_path = max(
                files,
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]),
            )
        except ValueError:
            print("Could not parse checkpoint filenames. Starting from scratch.")
            return 0

    print(f"Loading checkpoint from: {target_path}")
    checkpoint = torch.load(target_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")
    return start_epoch



def info_nce_loss(anchor, positive, negatives, temperature=0.1):

    pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / temperature        # (B, 1)
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / temperature # (B, K)

    logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+K)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


def train_model(
    model,
    dataloader,
    optimizer,
    start_epoch=0,
    max_epochs=100,
    checkpoint_dir="checkpoints",
    temperature=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    print(f"Training started on {device}...")

    for epoch in range(start_epoch, max_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
        epoch_loss = 0.0

        for anchor, positive, negatives in loop:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negatives = negatives.to(device)

            batch_size, num_neg, seq_len, feat_dim = negatives.shape

            # Forward
            anchor_emb = model(anchor)  # (B, Dim)
            pos_emb = model(positive)   # (B, Dim)

            neg_flat = negatives.flatten(0, 1)   # (B*K, Seq, Feat)
            neg_emb_flat = model(neg_flat)       # (B*K, Dim)
            neg_emb = neg_emb_flat.unflatten(0, (batch_size, num_neg))  # (B, K, Dim)

            # Loss
            loss = info_nce_loss(anchor_emb, pos_emb, neg_emb, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

        save_checkpoint(model, optimizer, epoch=epoch+1, folder=checkpoint_dir)


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    path_list = ["train_test_data1"]

    CHECKPOINT_DIR = "checkpoints"
    SAVE_MODEL_PATH = "final_model.pth"

    MAX_EPOCHS = 100

    LOAD_CHECKPOINT_ID = None

    BATCH_SIZE = 32
    NUM_WORKERS = 0

    INPUT_DIM = 66
    HIDDEN_DIM = 128
    OUTPUT_DIM = 64
    LR = 1e-5

    TEMPERATURE = 0.1

    for data_path in path_list:
        dataset = ContrastiveFeatureDataset(wav_folder=data_path, frame_length_sec=0.25)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

        model = LSTMEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Checkpoint load
        start_epoch = 0
        if LOAD_CHECKPOINT_ID != -1:
            start_epoch = load_checkpoint(
                model,
                optimizer,
                folder=CHECKPOINT_DIR,
                checkpoint_id=LOAD_CHECKPOINT_ID,
            )

        # Train
        if start_epoch < MAX_EPOCHS:
            train_model(
                model,
                dataloader,
                optimizer,
                start_epoch=start_epoch,
                max_epochs=MAX_EPOCHS,
                checkpoint_dir=CHECKPOINT_DIR,
                temperature=TEMPERATURE,
            )
        else:
            print("Training already completed based on the loaded checkpoint.")

        # Save final model
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f"Final model saved to {SAVE_MODEL_PATH}")
