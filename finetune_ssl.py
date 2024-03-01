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
from nih_dataset import NIHDataset
import torchmetrics
from train_ssl import  SimAPSSL

class APDataloader(pl.LightningDataModule):
    def __init__(self, path, batch_size, shuffle, num_workers=0, do_transform=False, max=None, min=None,):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.do_transform = do_transform


        self.train_dataset = NIHDataset(path, 'train', do_transform, max=max, min=min)
        self.val_dataset = NIHDataset(path, 'val', do_transform=False, max=max, min=min)
        self.test_dataset = NIHDataset(path, 'test', do_transform = False, max=max, min=min) # hard code this for test set
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
    def __init__(self, model_ckpt, num_classes, lr=1e-3, dropout=0.0, input_dim=128):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.input_dim = input_dim

        print(f"Loading model from {model_ckpt}")
        self.model = SimAPSSL.load_from_checkpoint(checkpoint_path=model_ckpt, num_classes=self.num_classes, task='finetune', 
                                                    input_dim=self.input_dim)

        self.criterion = nn.CrossEntropyLoss()
        #accuracy
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(num_classes=self.num_classes, average='macro', task='multiclass')


    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

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
        y = y.long()

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
        # ensure x is in [0, 1] range
        under_range = x < 0
        over_range = x > 1
        x[under_range] = 0.
        x[over_range] = 1.

        y = y.long()

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
        "checkpoint_path": None, 
        "dataset_path": "./data/processed_datasets/",
        "batch_size": 64,
        "shuffle": True,
        "do_transform": False,
        "max":np.array([3.14159274e+00, 1.56960034e+00, 3.14159250e+00, 9.43980408e+00,
        3.51681976e+01, 1.13550110e+01, 4.67008477e+04, 8.15972363e+03,
        1.16144785e+04, 1.12840000e+04, 4.40000000e+03, 2.69869995e+02]), 
        'min':np.array([-3.14159250e+00, -1.57079637e+00, -3.14159274e+00, -1.06317291e+01,
        -1.36996613e+01, -1.10269775e+01, -2.36590332e+03, -9.05344043e+03,
        -3.21341064e+03, -5.03000000e+02,  3.21600008e+00,  0.00000000e+00]),
        "epochs": 50,
        "num_classes":9, 
        "input_dim": 64, 
        "lr":1e-5, 
        "dropout":0.0,
        "model_ckpt": './trained_models/genial-water-8_best.ckpt'
    }


    # Initialize wandb
    wandb_logger = WandbLogger(
        project="simAP",
        # id='2ju60nmw',
        config=config,
        log_model=False,
        mode="online",
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
        max=config['max'],
        min=config['min']
    )

    # Initialize model
    model = SimAP(
        model_ckpt=config["model_ckpt"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        lr=config["lr"],
        input_dim=config['input_dim']
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
         dirpath='trained_models', filename=f"{model_name}_best", monitor="val_loss", mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_summary = ModelSummary(max_depth=2)



    # Set up trainer and fit
    trainer = pl.Trainer(
        accelerator="gpu",
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


