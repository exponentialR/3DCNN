import pytorch_lightning as pl
import torch.optim
from models import Example3DCNN
import torch.nn as nn


class experimental3DCNN(pl.LightningModule):
    def __int__(self, learning_rate):
        super(experimental3DCNN).__int__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = Example3DCNN()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        inputs = x['frames']
        output = self.model(inputs)

    def _evaluate(self, data):
        label = data['label']
        class_output = self(data)
        loss = self.cross_entropy(class_output, label)
        return {'loss': loss}

    def _step(self, batch, step_name):
        res = self._evaluate(batch)
        self.log_dict({f'{step_name}/{key}': val for key, val in res.items()}, prog_bar=True, on_epoch=True,
                      logger=True, sync_dist=True)

    def training_step(self, batch, _):
        return self._step(batch, 'train')

    def test_step(self, batch, _):
        return self._step(batch, 'test')

    def validation_step(self, batch, _):
        return self._step(batch, 'valid')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
