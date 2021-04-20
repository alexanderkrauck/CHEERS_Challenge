import torch
from torch import nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from transformers import BertModel

class RelevantModuleV00(LightningModule):
    def __init__(self, bert: BertModel, input_size: int, output_size: int, start_lr=1e-4):
        super().__init__()        
        self.bert = bert
        self.linear_after_bert = nn.Linear(bert.config.hidden_size, 256)
        self.feed_forward = nn.Sequential(
            #nn.BatchNorm1d(bert.config.hidden_size + input_size),#just a feeling this might be nice
            nn.Linear(256 + input_size, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, output_size)
        )
        
        
        self.start_lr = start_lr

    def forward(self, x):
        x_bert = x[0]
        x_other = x[1]
        y_bert = self.bert(x[0])["last_hidden_state"][:,0] #all batches but only clf output
        y_bert = self.linear_after_bert(y_bert)
        x = torch.cat((y_bert, x_other), dim=1)#dim=1 is feature dimensions (0 is batch dim)
        
        return self.feed_forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        #return optimizer
        return {
           'optimizer': optimizer,
           'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor =.2, patience =1, cooldown =2, min_lr =1e-6),
           'monitor': 'val_loss'
       }
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torch.sum(torch.argmax(y_hat, dim=-1) == y) / len(y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

class RelevantModuleV01(LightningModule):

    def __init__(self, bert: BertModel, input_size: int, output_size: int, start_lr=1e-4):
        super().__init__()        
        self.bert = bert
        self.linear_after_bert = nn.Linear(bert.config.hidden_size, 256)
        self.feed_forward = nn.Sequential(
            nn.Linear(256 + input_size, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, output_size)
        )
        
        
        for m in self.feed_forward:
            if type(m) is nn.Linear:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        
        self.start_lr = start_lr

    def forward(self, x):
        x_bert = x[0]
        x_other = x[1]

        x_bert = self.bert(x_bert)["last_hidden_state"][:,0] #all batches but only clf output
        x_bert = self.linear_after_bert(x_bert)
        x_bert = torch.relu(x_bert)#is new (not sure if improves by much)

        x = torch.cat((x_bert, x_other), dim=1)#dim=1 is feature dimensions (0 is batch dim)
        
        return self.feed_forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        #return optimizer
        return {
           'optimizer': optimizer,
           'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor =.2, patience =1, cooldown =2, min_lr =1e-6),
           'monitor': 'val_loss'
       }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        y_hat = self.forward(x)
        #maybe use BCE (with logits) loss. Note that in this case the required output size is just 1 (i.e. (batch_size, 1))
        loss = F.cross_entropy(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torch.sum(torch.argmax(y_hat, dim=-1) == y) / len(y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

class RelevantModuleV02(LightningModule):
    """
    This class is similar to the V01 but it implements a binary cross entropy (BCE) loss (since it is only 1 target)
    """

    def __init__(self, bert: BertModel, input_size: int, output_size: int = 1, start_lr=1e-4):
        super().__init__()        
        self.bert = bert
        self.linear_after_bert = nn.Linear(bert.config.hidden_size, 256)
        self.feed_forward = nn.Sequential(
            nn.Linear(256 + input_size, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1)
        )
        
        
        for m in self.feed_forward:
            if type(m) is nn.Linear:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        
        self.start_lr = start_lr

    def forward(self, x):
        x_bert = x[0]
        x_other = x[1]

        x_bert = self.bert(x_bert)["last_hidden_state"][:,0] #all batches but only clf output
        x_bert = self.linear_after_bert(x_bert)
        x_bert = torch.relu(x_bert)#is new (not sure if improves by much)

        x = torch.cat((x_bert, x_other), dim=1)#dim=1 is feature dimensions (0 is batch dim)

        return self.feed_forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        #return optimizer
        return {
           'optimizer': optimizer,
           'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor =.2, patience =1, cooldown =2, min_lr =1e-6),
           'monitor': 'val_loss'
       }
       
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y.float())#BCE needs y to be float

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat.flatten(), y.float())#BCE needs y to be float
        acc = torch.sum(torch.round(torch.sigmoid(y_hat.flatten())) == y) / len(y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

class SectorModuleV00(LightningModule):
    """Simple implementation of a sector module

    Does not implment weighting for samples with multiple sector ids
    """

    def __init__(self, bert: BertModel, input_size: int, output_size: int, start_lr=1e-4):
        super().__init__()        
        self.bert = bert
        self.linear_after_bert = nn.Linear(bert.config.hidden_size, 256)
        self.feed_forward = nn.Sequential(
            nn.Linear(256 + input_size, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, output_size)
        )
        
        
        for m in self.feed_forward:
            if type(m) is nn.Linear:
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        
        self.start_lr = start_lr

    def forward(self, x):
        x_bert = x[0]
        x_other = x[1]

        x_bert = self.bert(x_bert)["last_hidden_state"][:,0] #all batches but only clf output
        x_bert = self.linear_after_bert(x_bert)
        x_bert = torch.relu(x_bert)#is new (not sure if improves by much)

        x = torch.cat((x_bert, x_other), dim=1)#dim=1 is feature dimensions (0 is batch dim)

        return self.feed_forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        #return optimizer
        return {
           'optimizer': optimizer,
           'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor =.2, patience =1, cooldown =2, min_lr =1e-6),
           'monitor': 'val_loss'
       }
       
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        y_hat = self.forward(x)

        loss = F.cross_entropy(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)