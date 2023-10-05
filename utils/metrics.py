import torch
import numpy as np
from sklearn.metrics import f1_score
import random


class XEDiceLoss(torch.nn.Module):
    """
    Computes (0.5 * CrossEntropyLoss) + (0.5 * DiceLoss).
    """

    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred, true):
        pred = pred.squeeze(1)
        valid_pixel_mask = true.ne(255)

        temp_true = torch.where((true == 255), 0, true)
        bce_loss = self.bce(pred, temp_true.float())
        bce_loss = bce_loss.masked_select(valid_pixel_mask).mean()

        pred = pred.sigmoid()
        pred = pred.masked_select(valid_pixel_mask)
        true = true.masked_select(valid_pixel_mask)

        dice_loss = 1 - (2.0 * torch.sum(pred * true) + 1e-7) / (
            torch.sum(pred + true) + 1e-7
        )

        return (0.5 * bce_loss) + (0.5 * dice_loss)


def intersection_over_union(pred, true):
    """
    Calculates intersection and union for a batch of images.
    """
    valid_pixel_mask = true.ne(255)
    true = true.masked_select(valid_pixel_mask).to("cpu")
    pred = pred.masked_select(valid_pixel_mask).to("cpu")

    intersection = np.logical_and(true, pred)
    union = np.logical_or(true, pred)
    return intersection.sum() / union.sum()


def f1_score_fun(pred, true):
    """
    Calculates f1 score for a batch of images.
    """
    true_np = true.cpu().numpy().flatten().astype(int)
    pred_np = pred.cpu().numpy().flatten().astype(int)

    score = f1_score(pred_np, true_np)

    return score


def get_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
