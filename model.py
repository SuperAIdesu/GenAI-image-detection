from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy, Recall
import pytorch_lightning as pl
import timm  
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import *
from utils_sampling import *
from data_split import *
import logging
import os

class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
        self.accuracy = Accuracy(task='binary', threshold=0.5)
        self.recall = Recall(task='binary', threshold=0.5)  
        self.validation_outputs = []
        logging.basicConfig(filename='training.log',filemode='w',level=logging.INFO)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels, _ = batch
        outputs = self.forward(images).squeeze()
        
        print(f"Shape of outputs (training): {outputs.shape}")
        print(f"Shape of labels (training): {labels.shape}")
        
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        logging.info(f"Training Step - Batch loss: {loss.item()}")
        return loss

    def validation_step(self, batch):
        images, labels, _ = batch
        outputs = self.forward(images).squeeze()
        
        print(f"Shape of outputs (validation): {outputs.shape}")
        print(f"Shape of labels (validation): {labels.shape}")

        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        preds = torch.sigmoid(outputs)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(preds, labels.int()), prog_bar=True)
        self.log('val_recall', self.recall(preds, labels.int()), prog_bar=True)  
        output = {"val_loss": loss, "preds": preds, "labels": labels}
        self.validation_outputs.append(output)
        logging.info(f"Validation Step - Batch loss: {loss.item()}")
        return output

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            logging.warning("No outputs in validation step to process")
            return
        preds = torch.cat([x['preds'] for x in self.validation_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        if labels.unique().size(0) == 1:
            logging.warning("Only one class in validation step")
            return
        auc_score = roc_auc_score(labels.cpu(), preds.cpu())
        self.log('val_auc', auc_score, prog_bar=True)
        logging.info(f"Validation Epoch End - AUC score: {auc_score}")
        self.validation_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./model_checkpoints/',
    filename='image-classifier-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
    every_n_epochs=1,
)

train_domains = [0, 1]  
val_domains = [0, 1]  

checkpoint_dir = './model_checkpoints/'
latest_checkpoint = None

if os.path.exists(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        logging.info(f"Resuming from checkpoint: {latest_checkpoint}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")

model = ImageClassifier()

train_dl = load_dataloader(train_domains, "train", batch_size=32, num_workers=8)
logging.info("Training dataloader loaded")
val_dl = load_dataloader(val_domains, "val", batch_size=32, num_workers=8)
logging.info("Validation dataloader loaded")

trainer = pl.Trainer(callbacks=[checkpoint_callback],max_epochs=10)
trainer.fit(
    model=model, 
    train_dataloaders=train_dl,
    val_dataloaders=val_dl,
    ckpt_path=latest_checkpoint
)
