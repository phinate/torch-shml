import os

import pytorch_lightning as pl
import shml
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from dataset import EventDataset


class TrainingModule(pl.LightningModule):
    def __init__(self, model, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Store model
        self.model = model
        # Create loss module
        loss_fn = nn.BCELoss(reduction="none")

        def weighted_loss(y, y_hat, w):
            return (loss_fn(y, y_hat.unsqueeze(1).float()) * w).mean()

        self.loss_module = weighted_loss
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 38), dtype=torch.float32)

    def forward(self, events):
        # Forward function that is run when visualizing the graph
        return self.model(events)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        else:
            raise AssertionError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        features, labels, weights = batch
        scores = self.model(features)
        loss = self.loss_module(scores, labels, weights)

        # Calculate mean accuracy over training batch
        preds = (scores > 0.5).long()
        acc = (preds == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        features, labels, weights = batch
        scores = self.model(features)
        loss = self.loss_module(scores, labels, weights)
        # Calculate mean accuracy over training batch
        preds = (scores > 0.5).long()
        acc = (preds == labels).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        self.log("val_loss", loss)


#     def test_step(self, batch, batch_idx):
#         imgs, labels = batch
#         preds = self.model(imgs).argmax(dim=-1)
#         acc = (labels == preds).float().mean()
#         # By default logs it per epoch (weighted average over batches), and returns it afterwards
#         self.log("test_acc", acc)


def main():

    path = "/eos/user/n/nsimpson/shml_data/new"
    train_path = os.path.join(path, "train.parquet")
    test_path = os.path.join(path, "test.parquet")

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    train_dl = DataLoader(
        EventDataset(train_path, shml.ml_vars()),
        batch_size=64,
        num_workers=4,
        shuffle=True,
    )
    test_dl = DataLoader(
        EventDataset(test_path, shml.ml_vars()), batch_size=64, num_workers=4
    )

    def train_model(nn, save_name=None, **kwargs):
        """
        Inputs:
            model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
            save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
        """
        if save_name is None:
            save_name = "nn_test"

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = pl.Trainer(
            default_root_dir=os.path.join(path, save_name),  # Where to save models
            # We run on a single GPU (if possible)
            gpus=1 if str(device) == "cuda:0" else 0,
            # How many epochs to train for if no patience is set
            max_epochs=180,
            callbacks=[
                ModelCheckpoint(
                    save_weights_only=True, mode="max", monitor="val_acc"
                ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                LearningRateMonitor("epoch"),
            ],  # Log learning rate every epoch
            progress_bar_refresh_rate=1,
        )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
        trainer.logger._log_graph = (
            True  # If True, we plot the computation graph in tensorboard
        )
        trainer.logger._default_hp_metric = (
            None  # Optional logging argument that we don't need
        )

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = os.path.join(path, save_name + ".ckpt")
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {pretrained_filename}, loading...")
            # Automatically loads the model with the saved hyperparameters
            model = TrainingModule.load_from_checkpoint(pretrained_filename)
        else:
            pl.seed_everything(42)  # To be reproducable
            model = TrainingModule(model=nn, **kwargs)
            trainer.fit(model, train_dl, test_dl)
            model = TrainingModule.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )  # Load best checkpoint after training

        # Test best model on validation set
        val_result = trainer.test(model, test_dataloaders=test_dl, verbose=False)
        result = {"val": val_result[0]["test_acc"]}

        return model, result

    model = nn.Sequential(
        nn.Linear(38, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid(),
    )

    train_model(
        model,
        optimizer_name="Adam",
        optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
    )
