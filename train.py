import csv
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam


from utils.config import load_config
from utils.metrics import evaluate
from utils.dataset import SelfDefineDataset
from models import build_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备", device)

    config = load_config()

    seed = config["training"]["seed"]
    data_config = config["dataset"]

    torch.manual_seed(seed)
    train_dataset = SelfDefineDataset(
        root=data_config["root"],
        split="train",
        seed=seed,
        image_size=data_config["image_size"],
        val_ratio=data_config["val_ratio"],
        train_dir=data_config["train_dir"],
        test_dir=data_config["test_dir"],
        in_channels=data_config["in_channels"],
    )
    val_dataset = SelfDefineDataset(
        root=data_config["root"],
        split="val",
        seed=seed,
        image_size=data_config["image_size"],
        val_ratio=data_config["val_ratio"],
        train_dir=data_config["train_dir"],
        test_dir=data_config["test_dir"],
        in_channels=data_config["in_channels"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    print("训练集数据数量：", len(train_dataset))
    print("验证集数据数量：", len(val_dataset))

    model = build_model(config["model"]).to(device)
    print(f"使用的模型为：{config['model']['name']}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_iou = -1.0
    epochs = config["training"]["epochs"]

    history = []
    for epoch in range(epochs):
        print(f"开始 Epoch {epoch+1}/{epochs}")
        model.train()
        train_loss_sum = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
        train_loss = train_loss_sum / len(train_dataset)

        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_iou = val_metrics["iou"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f}",
        )
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), output_dir / "best.pth")
            print("已保存最佳模型")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_metrics["iou"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
            }
        )
        with open(
            output_dir / "metrics.csv", "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    print("完毕")


if __name__ == "__main__":
    main()
