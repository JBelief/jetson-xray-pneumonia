import sys, os, yaml
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# train.py 재사용
sys.path.append("/workspace")
sys.path.append("/workspace/scripts")
from train import build_model

class XrayFolderDataset(Dataset):
    def __init__(self, cfg, split="val"):
        splits_dir = cfg["data"]["output_splits"]
        split_file = os.path.join(splits_dir, f"{split}.txt")
        items = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 마지막 토큰만 label, 나머지는 path로
                path_part, label_str = line.rsplit(maxsplit=1)
                items.append((path_part, int(label_str)))
        self.items = items
        self.img_size = cfg["data"].get("img_size", 224)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        import numpy as np
        x = (np.array(img, dtype="float32") / 255.0).transpose(2, 0, 1)
        x = torch.from_numpy(x)
        return x, label

def main(cfg_path):
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = XrayFolderDataset(cfg, split="val")
    dl = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = build_model(cfg["train"]["model"])
    ckpt = torch.load("models/best.ckpt", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(torch.softmax(logits, 1), 1)
            correct += (pred == y).sum().item()
            total   += y.size(0)

    acc = correct / total if total else 0.0
    print(f"val accuracy: {acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(cfg_path)
