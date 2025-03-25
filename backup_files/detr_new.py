import os
import argparse
import sys

import torch
from torch.utils.data import DataLoader
import torchvision
from transformers import DetrFeatureExtractor, DetrForObjectDetection
import pytorch_lightning as pl



# Custom Dataset for DETR Training

class DetrCocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        # Annotation file is assumed to be located in a subfolder "annotations"
        ann_file = os.path.join(img_folder, "annotations", "annotations_coco.json")
        super(DetrCocoDataset, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # Load the image and its annotations in COCO format.
        img, target = super(DetrCocoDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        # Remove the batch dimension from the returned tensors.
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target



# Collate Function for DataLoader

def collate_fn(batch, feature_extractor):
    pixel_values = [item[0] for item in batch]
    # Pad images to the same size.
    encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }



# LightningModule for DETR Training

class DetrLightning(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_labels, id2label):
        super().__init__()
        # Save hyperparameters for logging
        self.save_hyperparameters()
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.id2label = id2label  # (Optional: for inference visualization)

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        # Ensure each target tensor is on the same device as the model
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("train_" + k, v, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            self.log("val_" + k, v, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.hparams.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        return optimizer



# Training Function

def run_training(data_dir, epochs, batch_size, lr, lr_backbone, weight_decay, checkpoint_path=None):
    """
    Runs DETR training.

    Args:
        data_dir (str): Base directory for dataset (should contain 'train' and 'val' folders).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        lr_backbone (float): Learning rate for the backbone.
        weight_decay (float): Weight decay.
        checkpoint_path (str, optional): Path to a checkpoint to resume training from.
    """
    # Create feature extractor instance.
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    # Define paths to training and validation images.
    train_folder = os.path.join(data_dir, "train")
    val_folder = os.path.join(data_dir, "val")

    # Create datasets.
    train_dataset = DetrCocoDataset(img_folder=train_folder, feature_extractor=feature_extractor, train=True)
    val_dataset = DetrCocoDataset(img_folder=val_folder, feature_extractor=feature_extractor, train=False)

    # Build a mapping from category IDs to label names.
    categories = train_dataset.coco.cats
    id2label = {k: v['name'] for k, v in categories.items()}
    num_labels = len(id2label)

    # Create DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, feature_extractor)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, feature_extractor)
    )

    # Instantiate (or resume) the LightningModule.
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = DetrLightning.load_from_checkpoint(
            checkpoint_path,
            lr=lr,
            lr_backbone=lr_backbone,
            weight_decay=weight_decay,
            num_labels=num_labels,
            id2label=id2label
        )
    else:
        model = DetrLightning(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay,
                              num_labels=num_labels, id2label=id2label)

    # Create a PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=epochs,
        gradient_clip_val=0.1,
        log_every_n_steps=1,
    )

    # Begin training. Training logs will be printed to stdout.
    trainer.fit(model, train_loader, val_loader)



# Command-Line Interface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DETR on a custom dataset using PyTorch Lightning")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory for dataset (should contain 'train' and 'val' folders)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    run_training(args.data_dir, args.epochs, args.batch_size, args.lr, args.lr_backbone, args.weight_decay, args.checkpoint_path)
