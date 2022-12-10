import time
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import Dropout, Linear, ReLU
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset, GitHub
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, Sequential, global_mean_pool
from pathlib import Path


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
LOG_DIR = "logs"
SEED = 14
pl.seed_everything(SEED)
# Ensure that all operations are deterministic on GPU for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class GCNClassifier(torch.nn.Module):

    def __init__(
            self,
            nbr_features: int = 3,
            nbr_classes: int = 2,
            hidden_layer_size: int = 256,
            batch_pool: bool = True
        ):
        """
        Sets up a 5-layer, Kipf & Welling, 2017 style GCN
        (https://arxiv.org/abs/1609.02907). Read the section on model depth.
        Dropout is performed here after every layer, rather than after the first and
        last only.

        :param nbr_features: the number of node features in the input
        :param nbr_classes: the number of node classes
        :param hidden_layer_size: the number of node features in the hidden layer
        :param batch_poo: if True, perform mean pooling by batch
        """
        # init for the super class (torch.nn.Module) allows GCN to use its methods
        super(GCNClassifier, self).__init__()
        self.nbr_features = nbr_features
        self.nbr_classes = nbr_classes
        self.hidden_layer_size = hidden_layer_size

        # read Pytorch Geometric's layer documentation here:
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

        # Some datasets pass everything through in 1 batch.
        # If this is the case, do not pool by batch but simply return the predictions.
        if batch_pool:
            self.model = Sequential(
                # define the input arguments of self.model
                "x, edge_index, batch_index",
                [
                    # GCNConv layer format: (GCNConv(in_channels, out_channels), "input1, input2 -> output")
                    (GCNConv(self.nbr_features, self.hidden_layer_size), "x, edge_index -> x1"),
                    # ReLU format: (ReLU(), "input -> output")
                    (ReLU(), "x1 -> x1a"),
                    (Dropout(p=0.5), "x1a -> x1d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x1d, edge_index -> x2"),
                    (ReLU(), "x2 -> x2a"),
                    (Dropout(p=0.5), "x2a -> x2d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x2d, edge_index -> x3"),
                    (ReLU(), "x3 -> x3a"),
                    (Dropout(p=0.5), "x3a -> x3d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x3d, edge_index -> x4"),
                    (ReLU(), "x4 -> x4a"),
                    (Dropout(p=0.5), "x4a -> x4d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x4d, edge_index -> x5"),
                    (ReLU(), "x5 -> x5a"),
                    (Dropout(p=0.5), "x5a -> x5d"),
                    # global_mean_pool returns batch-wise graph level outputs
                    # by averaging node features across the node dimension
                    (global_mean_pool, "x5d, batch_index -> x6"),
                    # linear transformation layer format: (Linear(in_channels, out_channels), "input -> output")
                    (Linear(self.hidden_layer_size, self.nbr_classes), "x6 -> x_out")
                ]
            )
        else:
            self.model = Sequential(
                # define the input arguments of self.model
                "x, edge_index, batch_index",
                [
                    # GCNConv layer format: (GCNConv(in_channels, out_channels), "input1, input2 -> output")
                    (GCNConv(self.nbr_features, self.hidden_layer_size), "x, edge_index -> x1"),
                    # ReLU format: (ReLU(), "input -> output")
                    (ReLU(), "x1 -> x1a"),
                    (Dropout(p=0.5), "x1a -> x1d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x1d, edge_index -> x2"),
                    (ReLU(), "x2 -> x2a"),
                    (Dropout(p=0.5), "x2a -> x2d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x2d, edge_index -> x3"),
                    (ReLU(), "x3 -> x3a"),
                    (Dropout(p=0.5), "x3a -> x3d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x3d, edge_index -> x4"),
                    (ReLU(), "x4 -> x4a"),
                    (Dropout(p=0.5), "x4a -> x4d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x4d, edge_index -> x5"),
                    (ReLU(), "x5 -> x5a"),
                    (Dropout(p=0.5), "x5a -> x5d"),
                    # linear transformation layer format: (Linear(in_channels, out_channels), "input -> output")
                    (Linear(self.hidden_layer_size, self.nbr_classes), "x5d -> x_out")
                ]
            )

    def forward(self, graph_data):
        """
        Parse the input, run it through the model, and return output.
        """
        node_features = graph_data.x
        edge_indices = graph_data.edge_index
        batch_indices = graph_data.batch
        x_out = self.model(node_features, edge_indices, batch_indices)
        return x_out


def evaluate_gcn_classifier(
        model: GCNClassifier,
        test_loader: DataLoader,
        loss_fxn: torch.nn.CrossEntropyLoss,
        tag: str = "_default",
        verbose: bool = False
    ):
    """
    An evaluation function for the GCN.

    :param model: an instance of GCNClassifier
    :param test_loader: data loader of the test set, or the set to be evaluated
    :param loss_fxn: cross entropy loss
    :param tag: string to identify the model being evaluated
    :param verbose: if True, prints results while training/evaluating
    :return:
    """
    correct_preds = 0.
    total_preds = 0.

    # shift model to eval mode
    model.eval()

    total_loss = 0
    total_batches = 0

    # for each batch in the test set...
    for batch in test_loader:
        try:
            temp = batch.x.shape
        except AttributeError:
            break
        # forward pass to get the predictions
        # do not use gradients, as this is for inference/validation
        with torch.no_grad():
            pred = model(batch.to(DEVICE))

        # calculate loss
        loss = loss_fxn(pred, batch.y.to(DEVICE))

        # increment numbers of correct predictions and total predictions
        correct_preds += (pred.argmax(dim=1) == batch.y).sum()
        total_preds += pred.shape[0]

        # add the batch level scores to the total scores
        total_loss += loss.detach()
        total_batches += batch.batch.max()

    test_loss = total_loss / total_batches if total_batches > 0 else total_loss
    test_accuracy = correct_preds / total_preds if total_preds > 0 else 0

    if verbose:
        print(f"accuracy = {test_accuracy:.4f}")

    results = {
        "accuracy": test_accuracy,
        "loss": test_loss,
        "tag": tag
    }

    return results


def train_gcn_classifier(
        model: GCNClassifier,
        train_loader: DataLoader,
        loss_fxn: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        epochs: int = 1000,
        print_progress_every_n_epochs: int = 1,
        verbose: bool = True,
        val_loader: DataLoader = None
):
    """
    A function to train a GCN classifier.

    :param model: an instance of GCNClassifier
    :param train_loader: data loader of the training set
    :param loss_fxn: cross entropy loss
    :param optimizer: optimization algorithm
    :param epochs: number of iterations over the training set
    :param print_progress_every_n_epochs: show progress at this interval of epochs
    :param verbose: if True, prints results while training/evaluating
    :param val_loader: data loader of the validation set

    :return: trained instance of GCNClassifier
    """
    model.to(DEVICE)

    # set up tensorboard logging for the model
    log_dir = LOG_DIR + f"/gcn_model_{str(int(time.time()))[-8:]}/"
    writer = SummaryWriter(log_dir=log_dir)

    # begin training loop
    for epoch in range(epochs):
        # shift model into training mode
        model.train()

        training_loss = 0.0
        batch_count = 0
        for batch in train_loader:
            # forward pass to get a prediction, calculate loss
            pred = model(batch.to(DEVICE))
            loss = loss_fxn(pred, batch.y.to(DEVICE))
            loss.backward()

            # update weights with the optimizer and zero out the gradients
            optimizer.step()
            optimizer.zero_grad()

            training_loss += loss.detach()
            batch_count += 1

        # tally and log the mean training loss for this epoch
        epoch_mean_training_loss = training_loss / batch_count
        writer.add_scalar("loss/train", epoch_mean_training_loss, epoch)

        # capture train/val loss and accuracy at checkpoint epochs
        if epoch % print_progress_every_n_epochs == 0:
            train_results = evaluate_gcn_classifier(
                model=model,
                test_loader=train_loader,
                loss_fxn=loss_fxn,
                tag=f"train_ckpt_{epoch}_",
                verbose=False
            )
            train_loss = train_results["loss"]
            train_accuracy = train_results["accuracy"]

            if verbose:
                print(
                    f"training loss & accuracy at epoch {epoch} = "
                    f"{train_loss:.4f} & {train_accuracy:.4f}"
                )

            # if a validation dataset is provided, evaluate it
            if val_loader is not None:
                val_results = evaluate_gcn_classifier(
                    model=model,
                    test_loader=val_loader,
                    loss_fxn=loss_fxn,
                    tag=f"val_ckpt_{epoch}_",
                    verbose=False
                )
                val_loss = val_results["loss"]
                val_accuracy = val_results["accuracy"]

                if verbose:
                    print(
                        f"val. loss & accuracy at epoch {epoch} = "
                        f"{val_loss:.4f} & {val_accuracy:.4f}"
                    )
            # if no validation set, set the scores to Inf and -Inf
            else:
                val_loss = float("Inf")
                val_accuracy = -float("Inf")

            # add train/val loss and accuracy to the tensorboard writer
            writer.add_scalar("loss/train_eval", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("accuracy/train", train_accuracy, epoch)
            writer.add_scalar("accuracy/val", val_accuracy, epoch)

    return model



class LightningGCNClassifier(pl.LightningModule):

    def __init__(
            self,
            nbr_features: int = 3,
            nbr_classes: int = 2,
            hidden_layer_size: int = 256,
            batch_pool: bool = True
    ):
        """
        Sets up a 5-layer, Kipf & Welling, 2017 style GCN (https://arxiv.org/abs/1609.02907).
        Read the section on model depth.  Dropout is performed here after every layer, rather than after
        the first and last only.

        :param nbr_features: the number of node features in the input
        :param nbr_classes: the number of node classes
        :param hidden_layer_size: the number of node features in the hidden layer
        :param batch_pool: if True, perform mean pooling by batch
        """
        # init for the super class (torch.nn.Module) allows GCN to use its methods
        super(LightningGCNClassifier, self).__init__()
        self.nbr_features = nbr_features
        self.nbr_classes = nbr_classes
        self.hidden_layer_size = hidden_layer_size

        # read Pytorch Geometric's layer documentation here:
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

        # Some datasets pass everything through in 1 batch.
        # If this is the case, do not pool by batch but simply return the predictions.
        if batch_pool:
            self.model = Sequential(
                # define the input arguments of self.model
                "x, edge_index, batch_index",
                [
                    # GCNConv layer format: (GCNConv(in_channels, out_channels), "input1, input2 -> output")
                    (GCNConv(self.nbr_features, self.hidden_layer_size), "x, edge_index -> x1"),
                    # ReLU format: (ReLU(), "input -> output")
                    (ReLU(), "x1 -> x1a"),
                    (Dropout(p=0.5), "x1a -> x1d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x1d, edge_index -> x2"),
                    (ReLU(), "x2 -> x2a"),
                    (Dropout(p=0.5), "x2a -> x2d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x2d, edge_index -> x3"),
                    (ReLU(), "x3 -> x3a"),
                    (Dropout(p=0.5), "x3a -> x3d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x3d, edge_index -> x4"),
                    (ReLU(), "x4 -> x4a"),
                    (Dropout(p=0.5), "x4a -> x4d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x4d, edge_index -> x5"),
                    (ReLU(), "x5 -> x5a"),
                    (Dropout(p=0.5), "x5a -> x5d"),
                    # global_mean_pool returns batch-wise graph level outputs
                    # by averaging node features across the node dimension
                    (global_mean_pool, "x5d, batch_index -> x6"),
                    # linear transformation layer format: (Linear(in_channels, out_channels), "input -> output")
                    (Linear(self.hidden_layer_size, self.nbr_classes), "x6 -> x_out")
                ]
            )
        else:
            self.model = Sequential(
                # define the input arguments of self.model
                "x, edge_index, batch_index",
                [
                    # GCNConv layer format: (GCNConv(in_channels, out_channels), "input1, input2 -> output")
                    (GCNConv(self.nbr_features, self.hidden_layer_size), "x, edge_index -> x1"),
                    # ReLU format: (ReLU(), "input -> output")
                    (ReLU(), "x1 -> x1a"),
                    (Dropout(p=0.5), "x1a -> x1d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x1d, edge_index -> x2"),
                    (ReLU(), "x2 -> x2a"),
                    (Dropout(p=0.5), "x2a -> x2d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x2d, edge_index -> x3"),
                    (ReLU(), "x3 -> x3a"),
                    (Dropout(p=0.5), "x3a -> x3d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x3d, edge_index -> x4"),
                    (ReLU(), "x4 -> x4a"),
                    (Dropout(p=0.5), "x4a -> x4d"),
                    (GCNConv(self.hidden_layer_size, self.hidden_layer_size), "x4d, edge_index -> x5"),
                    (ReLU(), "x5 -> x5a"),
                    (Dropout(p=0.5), "x5a -> x5d"),
                    # linear transformation layer format: (Linear(in_channels, out_channels), "input -> output")
                    (Linear(self.hidden_layer_size, self.nbr_classes), "x5d -> x_out")
                ]
            )

    def forward(self, x: torch.tensor, edge_index: torch.tensor, batch_index: torch.tensor):
        """
        Parse the input, run it through the model, and return output.

        :param x: node features
        :param edge_index: tensor with the node-node edges in the batch, in COO format
        :param batch_index: tensor with the node IDs in the batch
        """
        return self.model(x, edge_index, batch_index)

    def training_step(
            self,
            batch: Batch,
            batch_index: torch.tensor
    ):
        """
        Defines how the model can be trained.

        :param batch: a batch from a DataLoader instance
        :param batch_index: tensor with the node IDs in the batch

        :return: batch level loss
        """
        # get the node features for this batch, the edge index, and the batch index
        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch

        # forward pass through the network to get the logits, and calculate the loss
        logits = self.forward(x, edge_index, batch_index)
        loss = F.cross_entropy(logits, batch.y)

        # predicted class = argmax, determine accuracy
        pred = logits.argmax(-1)
        label = batch.y
        accuracy = (pred == label).sum() / pred.shape[0]

        # calling self.log automatically creates scalars for Tensorboard
        self.log("loss/train", loss)
        self.log("accuracy/train", accuracy)

        return loss

    def validation_step(
            self,
            batch: Batch,
            batch_index: torch.tensor
    ):
        """

        :param batch: a batch from a DataLoader instance
        :param batch_index: tensor with the node IDs in the batch

        :return: tuple of the logits, predicted class, ground truth class
        """
        # get the node features for this batch, the edge index, and the batch index
        x, edge_index = batch.x, batch.edge_index
        batch_index = batch.batch

        # forward pass through the network to get the logits, and calculate the loss
        logits = self.forward(x, edge_index, batch_index)
        loss = F.cross_entropy(logits, batch.y)

        # predicted class = argmax
        pred = logits.argmax(-1)

        return logits, pred, batch.y

    def validation_epoch_end(self, validation_step_outputs):
        """
        Tells Pytorch Lightning what to do with the outputs of each validation_step()

        :param validation_step_outputs: tuple of logits, predicted class, and ground truth class
        """
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for output, pred, labels in validation_step_outputs:
            # calculate the loss and accuracy
            val_loss += F.cross_entropy(output, labels, reduction="sum")

            correct_preds += (pred == labels).sum()
            total_preds += pred.shape[0]

            val_accuracy = correct_preds / total_preds
            val_loss = val_loss / total_preds

        # calling self.log automatically creates scalars for Tensorboard
        self.log("accuracy/val", val_accuracy)
        self.log("loss/val", val_loss)

    def configure_optimizers(self):
        """
        Configures optimization algorithm for the model.

        :return: optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # load a toy dataset
    # dataset = TUDataset(root="./tmp", name="PROTEINS")
    # dataset = GNNBenchmarkDataset(root="./tmp", name="MNIST")
    dataset = GitHub(root="./tmp")

    # shuffle dataset and get train/validation/test splits
    dataset = dataset.shuffle()
    num_samples = len(dataset)
    # Some datasets consist of 1 Data instance.  In these cases, make the batch size equal to the number of samples.
    batch_size = 32 if len(dataset) > 1 else dataset[0].x.shape[0]
    batch_pool = True if batch_size == 32 else False
    num_val = num_samples // 10
    val_dataset = dataset[:num_val]
    test_dataset = dataset[num_val:2 * num_val]
    train_dataset = dataset[2 * num_val:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # determine feature dimensions and number of classes
    batch = iter(train_loader)
    sample_batch = batch.next()
    nbr_features = sample_batch.x.shape[1]
    nbr_classes = dataset.num_classes

    ## GCN ##

    gcn_model = GCNClassifier(
        nbr_features=nbr_features,
        nbr_classes=nbr_classes,
        batch_pool=batch_pool,
    )

    # define the loss function and optimization algorithm
    lr = 1e-3
    epochs = 50
    loss_fxn = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr)

    train_gcn_classifier(
        model=gcn_model,
        train_loader=train_loader,
        loss_fxn=loss_fxn,
        optimizer=optimizer,
        epochs=epochs,
        print_progress_every_n_epochs=5,
        verbose=True,
        val_loader=val_loader
    )

    # save the model
    save_path = Path("models")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(gcn_model.state_dict(), save_path / "gcn.pt")

    # reload the model for inference
    gcn_model_reloaded = GCNClassifier(
        nbr_features=nbr_features,
        nbr_classes=nbr_classes,
        batch_pool=batch_pool
    )
    gcn_model_reloaded.load_state_dict(torch.load(save_path / "gcn.pt"))
    gcn_model_reloaded.eval()

    ## Lightning ##

    lightning_gcn_model = LightningGCNClassifier(
        nbr_features=nbr_features,
        nbr_classes=nbr_classes,
        batch_pool=batch_pool
    )

    epochs = 50
    val_check_interval = len(train_loader)

    # lightning trainer automatically does 1) tensorboard logging, 2) model checkpointing,
    # 3) training and validation looping, and 4) early-stopping
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=epochs,
        val_check_interval=val_check_interval,
    )
    trainer.fit(lightning_gcn_model, train_loader, val_loader)

    # save the model
    save_path = Path("models")
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(lightning_gcn_model.state_dict(), save_path / "lightning_gcn.pt")
    # trainer.save_checkpoint(save_path / "lightning_gcn.ckpt")

    # reload the model for inference
    lightning_gcn_model_reloaded = LightningGCNClassifier(
        nbr_features=nbr_features,
        nbr_classes=nbr_classes,
        batch_pool=batch_pool
    )
    lightning_gcn_model_reloaded.load_state_dict(torch.load(save_path / "lightning_gcn.pt"))
    lightning_gcn_model_reloaded.eval()


# to see accuracy and loss after training:
# tensorboard --logdir lightning_logs/version_0

# to see how to use masks to split the GitHub data into train/val/test, see:
# https://awadrahman.medium.com/hands-on-graph-neural-networks-for-social-network-using-pytorch-30231c130b38
