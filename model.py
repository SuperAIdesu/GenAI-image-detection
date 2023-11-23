from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy, Recall
import pytorch_lightning as pl
import timm  
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import *
from utils_sampling import *
from data_split import *

class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
        self.accuracy = Accuracy()
        self.recall = Recall()  

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels, _ = batch
        outputs = self.forward(images).squeeze()
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        return loss

    def validation_step(self, batch):
        images, labels, _ = batch
        outputs = self.forward(images).squeeze()
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        preds = torch.sigmoid(outputs)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy(preds, labels.int()), prog_bar=True)
        self.log('val_recall', self.recall(preds, labels.int()), prog_bar=True)  
        return {"val_loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        auc_score = roc_auc_score(labels.cpu(), preds.cpu())
        self.log('val_auc', auc_score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer

    def train_dataloader(self):
        return load_dataloader(train_domains, "train", batch_size=32, num_workers=4)
    
    def val_dataloader(self):
        return load_dataloader(val_domains, "eval", batch_size=32, num_workers=4)
    
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

model = ImageClassifier()
trainer = pl.Trainer(gpus=1, callbacks=[checkpoint_callback],max_epochs=10)
trainer.fit(model)