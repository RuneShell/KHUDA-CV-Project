import os
import gc
import numpy as np

from sklearn.metrics import f1_score
import torch

from utils.transforms import MyDataset
from models.model import VideoMAEMultiHead
from utils.logger import Logger


#######################
# 0. Custom Vars
#######################

BASE_DIR = "/data2/local_datasets/173_dataset/"
TRAIN_DIR = os.path.join(BASE_DIR, "train") # 1152 + 384
TEST_DIR = os.path.join(BASE_DIR, "val") # 144 + 48

LEARNING_RATE1 = 1e-3
LEARNING_RATE2 = 1e-4
LEARNING_RATE3 = 1e-5
EPOCHS = 100
BATCH_SIZE = 16
STAGE3_BATCH_SIZE = 4
DEVICE = "cuda"

CLIP_LEN = 48
IMAGE_SIZE = 224
CHECKPOINT_DIR = os.path.join("/data2/local_datasets/hjwork/work", "checkpoints")

logger = Logger()
logger.timestamp(f"Start Logging. epochs: {EPOCHS}, batch size: {BATCH_SIZE}, image size: {IMAGE_SIZE}, learning rate: {LEARNING_RATE1}/{LEARNING_RATE2}/{LEARNING_RATE3}, clip length: {CLIP_LEN}")

########################
# 1. methods
########################
def train_one_epoch(model, train_loader, optimizer, scaler, clip_loss, frame_loss, device=DEVICE):
    model.train()
    running_loss = 0.0
    all_clip_preds, all_clip_labels = [], []
    all_frame_preds, all_frame_labels = [], []

    for batch in train_loader:
        frames, clip_label, frame_labels = batch
        frames, clip_label, frame_labels = frames.to(device), clip_label.to(device), frame_labels.to(device)

        # forward
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            clip_logit, frame_logits = model(frames) # batched (B, 1), (B, CLIP_LEN // tubelet_size)
            # loss
            loss_clip = clip_loss(clip_logit, clip_label)
            loss_frame = frame_loss(frame_logits, frame_labels)
            loss = loss_clip + loss_frame # 1 : 1 weighted sum.
        running_loss += loss.item()
        # backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # prediction
        clip_pred = (clip_logit > 0).float()
        all_clip_preds.extend(clip_pred.cpu().numpy())
        all_clip_labels.extend(clip_label.cpu().numpy())

        frame_pred = (frame_logits > 0).float()
        all_frame_preds.extend(frame_pred.cpu().numpy())
        all_frame_labels.extend(frame_labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_clip_accuracy = (np.array(all_clip_preds) == np.array(all_clip_labels)).mean()
    epoch_clip_f1 = f1_score(all_clip_labels, all_clip_preds, average='macro')
    epoch_frame_f1 = f1_score(all_frame_labels, all_frame_preds, average='macro')
    return epoch_loss, epoch_clip_accuracy, epoch_clip_f1, epoch_frame_f1

def validate(model, val_loader, clip_loss, frame_loss, device=DEVICE):
    model.eval()
    running_loss = 0.0
    all_clip_preds, all_clip_labels = [], []
    all_frame_preds, all_frame_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            frames, clip_label, frame_labels = batch
            frames, clip_label, frame_labels = frames.to(device), clip_label.to(device), frame_labels.to(device)

            # forward
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                clip_logit, frame_logits = model(frames)
                # loss
                loss_clip = clip_loss(clip_logit, clip_label)
                loss_frame = frame_loss(frame_logits, frame_labels)
                loss = loss_clip + loss_frame
            # prediction
            clip_pred = (clip_logit > 0).float()
            all_clip_preds.extend(clip_pred.cpu().numpy())
            all_clip_labels.extend(clip_label.cpu().numpy())

            frame_pred = (frame_logits > 0).float()
            all_frame_preds.extend(frame_pred.cpu().numpy())
            all_frame_labels.extend(frame_labels.cpu().numpy())

            running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader)
    epoch_clip_accuracy = (np.array(all_clip_preds) == np.array(all_clip_labels)).mean()
    epoch_clip_f1 = f1_score(all_clip_labels, all_clip_preds, average='macro')
    epoch_frame_f1 = f1_score(all_frame_labels, all_frame_preds, average='macro')
    return epoch_loss, epoch_clip_accuracy, epoch_clip_f1, epoch_frame_f1


def train():
    model = VideoMAEMultiHead(pretrained="MCG-NJU/videomae-base", num_frames=CLIP_LEN).to(DEVICE)
    logger.timestamp(f"Model loaded: {model.__class__.__name__}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    mydataset = MyDataset('train', 
                        BASE_PATH=BASE_DIR,
                        BATCH_SIZE=BATCH_SIZE, image_size=IMAGE_SIZE, num_frames=CLIP_LEN)
    train_loader, val_loader = mydataset.get_train_dataset()

    #optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    clip_loss = torch.nn.BCEWithLogitsLoss()
    frame_loss = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


    # ==== training loop ====
    for epoch in range(EPOCHS):
        # Progressive Unfreezing
        if epoch == 0:
            logger.timestamp("Stage 1")
            model.unfreeze_heads()  # clip_head, frame_head만 학습
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        elif epoch == 3:
            logger.timestamp("Stage 2")
            model.unfreeze_last_layers(num_layers=1)  # backbone의 마지막 layer + heads 학습
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE2, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        elif epoch == 10:
            logger.timestamp("Stage 3")
            model.unfreeze_all()  # 전체 학습
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
            train_loader, val_loader = mydataset.get_train_dataset(
                train_batch_size=STAGE3_BATCH_SIZE,
                val_batch_size=STAGE3_BATCH_SIZE,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # train step
        train_loss, train_clip_accuracy, train_clip_f1, train_frame_f1 = train_one_epoch(model, train_loader, optimizer, scaler, clip_loss, frame_loss)
        logger.timestamp(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_clip_accuracy:.4f}, Train F1: {train_clip_f1:.4f}, Train Frame F1: {train_frame_f1:.4f}\n")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt"),
        )
        scheduler.step()

        # validation
        if epoch % 5 == 0:
            val_loss, val_accuracy, val_f1, val_frame_f1 = validate(model, val_loader, clip_loss, frame_loss)
            logger.timestamp(f"Epoch {epoch}/{EPOCHS} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val Frame F1: {val_frame_f1:.4f}\n")


########################
# 2. Train Loop
########################


if __name__ == "__main__":
    train()
    logger.timestamp("Training completed.")


