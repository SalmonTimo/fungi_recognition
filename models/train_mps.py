#!/usr/bin/env python3
# train_convnextv2_xl.py
"""
Train image recognition models given a zarr dataset.
Author: Tim Salmon  •  2025-05-01
"""
import argparse, math, os, time, json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm                                             # pip install timm>=0.9.10
from timm.data import create_transform
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2, NativeScaler
import zarr
from tqdm import tqdm

# ----------------------------  Data  ----------------------------------------- #
class ZarrDataset(Dataset):
    """Reads (C,H,W) images + label indices from a Zarr store."""
    def __init__(self, zarr_path: str, split: str, class_map: dict[str,int],
                 transform=None):
        import zarr
        import numpy as np

        root = zarr.open_group(zarr_path, mode="r")
        self.imgs    = root["images"]        # Zarr Array, shape = (N,3,256,256)
        self.genus   = root["genus"][:]      # in-memory numpy array of byte-strings
        N = self.imgs.shape[0]               # get length from shape
        img = root["images"][0]   # before you do anything
        print("first zarr image shape:", img.shape)          # is it (3,256,256) or (256,256,3)?

        # simple stratified 90/10 split
        rng = np.random.RandomState(42)
        indices = np.arange(N)
        rng.shuffle(indices)
        cut = int(0.9 * N)
        if split == "train":
            self.indices = indices[:cut]
        else:
            self.indices = indices[cut:]

        self.cls_map  = class_map
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = int(self.indices[idx])
        # read C,H,W uint8 and normalize to float32
        x = self.imgs[img_idx].astype("float32") / 255.0  
        # x is a numpy array; convert to tensor
        x = torch.from_numpy(x)
        # apply transforms (expects torch.Tensor in CHW)
        if self.transform:
            x = self.transform(x)
        # decode genus to string, strip nulls, map to int label
        genus_str = self.genus[img_idx].decode("utf-8").rstrip("\0")
        y = self.cls_map[genus_str]
        return x, y

# ----------------------------  Helpers  -------------------------------------- #
def build_class_map(zarr_path: str) -> dict[str,int]:
    import zarr, pandas as pd
    gens = zarr.open_group(zarr_path, "r")["genus"][:]
    uniq = pd.unique([g.decode("utf-8").rstrip("\0") for g in gens])
    return {g:i for i,g in enumerate(sorted(uniq))}

def create_loaders(zarr_path, batch_size, workers, img_size):
    class_map = build_class_map(zarr_path)
    mean,std  = (0.485,0.456,0.406), (0.229,0.224,0.225)
    train_tf  = create_transform(
        input_size=img_size,
        is_training=True,
        hflip=0.5, vflip=0.0,
        interpolation="bicubic",
        mean=mean, std=std,
        re_prob=0.25, re_mode="pixel", re_count=1,
    )
    val_tf    = create_transform(
        input_size=img_size,
        is_training=False,
        mean=mean, std=std
    )
    train_set = ZarrDataset(zarr_path,"train",class_map,train_tf)
    val_set   = ZarrDataset(zarr_path,"val",  class_map,val_tf)
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(
        train_set,
        batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
    ) 
    val_loader   = DataLoader(val_set,   batch_size, shuffle=False,
                              num_workers=workers, pin_memory=False)
    return train_loader,val_loader,len(class_map)

# ------------------------  Training / Validation  ---------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    device, epoch, ema=None, clip_norm=1.0):
    model.train()
    loss_meter, acc_meter = 0., 0.
    for step,(x,y) in enumerate(tqdm(loader)):
        print(x.dtype, x.min(), x.max())     # Expect float32 in roughly [-2, +2] after normalize
        print(x.shape)                       # Should be [B,3,H,W]
        print(f"step {step} lr = {optimizer.param_groups[0]['lr']}")
        x,y = x.to(device, non_blocking=True).to(memory_format=torch.channels_last), y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if ema: ema.update(model)
        scheduler.step(epoch + step/len(loader))

        loss_meter += loss.item()*x.size(0)
        acc_meter  += (out.argmax(1)==y).float().sum().item()
    n = len(loader.dataset)
    return loss_meter/n, acc_meter/n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = 0.0
    acc_meter  = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        # Pure FP32 inference
        out = model(x)
        loss = criterion(out, y)

        loss_meter += loss.item() * x.size(0)
        acc_meter  += (out.argmax(dim=1) == y).float().sum().item()

    n = len(loader.dataset)
    return loss_meter / n, acc_meter / n

# ------------------------------  Main  --------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zarr",          required=True)
    p.add_argument("--epochs",        type=int, default=30)
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--lr-base",       type=float, default=2e-4)
    p.add_argument("--img-size",      type=int, default=256)
    p.add_argument("--workers",       type=int, default=8)
    p.add_argument("--model-name",    default="mobilevitv2_200")
    p.add_argument("--output",        default="checkpoints")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_loader, val_loader, n_classes = create_loaders(
        args.zarr, args.batch_size, args.workers, args.img_size
    )

    # ---- model ----
    model = timm.create_model(
        args.model_name, pretrained=True, num_classes=n_classes, drop_path_rate=0.2
    )
    # model.set_grad_checkpointing(True)
    model = model.to(device, memory_format=torch.channels_last)

    # ---- Optimizer & LR ----
    global_batch = args.batch_size
    lr = args.lr_base * global_batch / 256
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9,0.999))
    steps_per_epoch = len(train_loader)
    sched = CosineLRScheduler(
        opt, t_initial=args.epochs*steps_per_epoch,
        lr_min=1e-6,
        warmup_t=5*steps_per_epoch,
        warmup_lr_init=lr*1e-3,
        k_decay=1.0
    )
    ema    = ModelEmaV2(model, decay=0.99996, device=device)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.output, exist_ok=True)
    best_acc=0; patience=10; bad_epochs=0
    for epoch in range(args.epochs):
        t0=time.time()
        tr_loss,tr_acc = train_one_epoch(model, train_loader, criterion,
                                         opt, sched, device, epoch, ema)
        va_loss,va_acc = evaluate(ema.module, val_loader, criterion, device)
        dt=time.time()-t0
        print(f"Epoch {epoch:02d}: "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3%} • "
              f"val loss {va_loss:.4f} acc {va_acc:.3%} • {dt/60:.1f} min")

        # checkpoint
        torch.save({
            "model": ema.module.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "acc":  va_acc,
        }, Path(args.output)/f"epoch{epoch:02d}.pt")

        # early stop
        if va_acc > best_acc:
            best_acc = va_acc
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping 🎉")
                break

if __name__ == "__main__":
    main()
