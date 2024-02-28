import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger
from models.model import Model
from dataset import simAPDataset
import torchmetrics

class APDataloader(pl.LightningDataModule):
    def __init__(self, path, batch_size, shuffle, num_workers=0, do_transform=False):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.do_transform = do_transform


        self.train_dataset = simAPDataset(path, 'train', do_transform)
        self.val_dataset = simAPDataset(path, 'val', do_transform=False)
        self.test_dataset = simAPDataset(path, 'test', do_transform = False) # hard code this for test set
        # self.test_har_dataset = simAPDataset('./data/HAR/', 'test', do_transform = False) # hard code this for test set

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
    
    def test_dataloader(self):
        loader1 = DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                )
                    
        return loader1
    
class SimAP(pl.LightningModule):
    def __init__(self, input_dim, num_classes, num_channels, embed_dim, heads, depth, lr=1e-3, dropout=0.0, token_strat='channel'):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.heads = heads
        self.depth = depth
        self.dropout = dropout
        self.lr = lr
        self.token_strat = token_strat

        num_tokens = input_dim * num_channels // 8

        self.model = Model(seq_len=input_dim, num_classes=num_classes, num_channels=num_channels, embed_dim=embed_dim, 
                           heads=heads, depth=depth, dropout=dropout, num_tokens=num_tokens, token_strat=token_strat)
        self.criterion = nn.CrossEntropyLoss()
        #accuracy
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(num_classes=self.num_classes, average='macro', task='multiclass')


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, 0].long()
        # turn into one-hot
        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=-1), y)  # Accuracy expects class indices
        f1 = self.f1_score(torch.argmax(y_hat, dim=-1), y)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y[:, 0].long()
        # turn into one-hot

        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=-1), y)  # Accuracy expects class indices
        f1 = self.f1_score(torch.argmax(y_hat, dim=-1), y)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_f1', f1)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y = y[:, 0].long()

        y_hat, _ = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(torch.argmax(y_hat, dim=-1), y)
        f1 = self.f1_score(torch.argmax(y_hat, dim=-1), y)

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_f1', f1)

        return {'test_loss': loss, 'test_acc': acc, 'test_f1': f1}

    def configure_optimizers(self) :
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            div_factor=25,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }



if __name__ == '__main__':
    pl.seed_everything(40)

    config = {
        "dataset_path": "./data/combined/",
        "batch_size": 32,
        "shuffle": True,
        "epochs": 10,
        "lr": 1e-3,
        "num_classes": 6,
        "num_channels": 12,
        "input_dim": 128, # 128 is the sequence length of each channel
        "embed_dim": 512,
        "heads": 6,
        "depth": 6,
        "dropout": 0.0,
        "checkpoint_path": None,
        "do_transform": True,
        "token_strat": "channel", # or could be 'seq'
    }


    # Initialize wandb
    wandb_logger = WandbLogger(
        project="simAP",
        # id='2ju60nmw',
        config=config,
        log_model=False,
        mode="offline",
    )
    config = wandb_logger.experiment.config
    model_name = wandb_logger.experiment.name
    wandb_logger.experiment.log_code(".")

    # Load dataset
    dataset = APDataloader(
        path = config["dataset_path"],
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        do_transform=config["do_transform"],
    )

    # Initialize model
    model = SimAP(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        num_channels=config["num_channels"],
        embed_dim=config["embed_dim"],
        heads=config["heads"],
        depth=config["depth"],
        dropout=config["dropout"],
        lr=config["lr"],
        token_strat=config["token_strat"],
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
         dirpath='trained_models', filename=f"{model_name}_best", monitor="val_loss", mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)



    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="cpu",
        #devices=[0],
        #strategy="ddp_find_unused_parameters_true",
        precision='32',
        sync_batchnorm=True,
        # use_distributed_sampler=True,
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, lr_monitor, model_summary],
        gradient_clip_val=1.,
        logger=wandb_logger,
        # accumulate_grad_batches=2,
    )

    if config['checkpoint_path'] is not None:
        print("Loading pre-trained checkpoint")
        trainer.fit(model, dataset, ckpt_path=config['checkpoint_path'])
    else:
        trainer.fit(model, dataset)
    
    # Evaluate on test dataset
    test_results = trainer.test(model, datamodule=dataset)
    print(f"Test Results: {test_results}")

    # Finish wandb run
    wandb_logger.experiment.finish()

