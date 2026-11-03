import torch
import torch.nn as nn

import segmentation_models_pytorch as smp


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    loss_sum = 0.0
    tp_list, fp_list, fn_list, tn_list = [], [], [], []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        loss_sum += loss.item() * images.size(0)

        predictions = logits.argmax(dim=1)

        stats = smp.metrics.get_stats(
            predictions, masks.long(), mode="multiclass", num_classes=4
        )

        stats = tuple(value[:, 1:] for value in stats)
        tp, fp, fn, tn = stats
        tp_list.append(tp.cpu())
        fp_list.append(fp.cpu())
        fn_list.append(fn.cpu())
        tn_list.append(tn.cpu())

    tp = torch.cat(tp_list, dim=0)
    fp = torch.cat(fp_list, dim=0)
    fn = torch.cat(fn_list, dim=0)
    tn = torch.cat(tn_list, dim=0)

    options = {"reduction": "macro", "zero_division": 0.0}

    return {
        "loss": loss_sum / len(loader.dataset),
        "iou": smp.metrics.iou_score(tp, fp, fn, tn, **options).item(),
        "f1": smp.metrics.f1_score(tp, fp, fn, tn, **options).item(),
        "precision": smp.metrics.precision(tp, fp, fn, tn, **options).item(),
        "recall": smp.metrics.recall(tp, fp, fn, tn, **options).item(),
    }
