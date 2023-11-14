import pytorch_lightning as pl
import torch.optim
from models import Example3DCNN
import torch.nn as nn
import torchmetrics


class experimental3DCNN(pl.LightningModule):
    def __init__(self, learning_rate):
        super(experimental3DCNN, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = float(learning_rate)
        self.model = Example3DCNN()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=10)

    def forward(self, x):
        inputs = x['frames']
        inputs = inputs.permute(0, 2, 1, 3, 4)

        output = self.model(inputs)
        return output

    def _evaluate(self, data):
        label = data['label']
        class_output = self(data)
        loss = self.cross_entropy(class_output, label)
        acc = self.accuracy(class_output.argmax(dim=1), label)
        return {'loss': loss, 'accuracy':acc}

    def _step(self, batch, step_name):
        res = self._evaluate(batch)
        self.log_dict({f'{step_name}/{key}': val for key, val in res.items()}, prog_bar=True, on_epoch=True,
                      logger=True, sync_dist=True)
        return res

    def training_step(self, batch, _):
        return self._step(batch, 'train')

    def test_step(self, batch, _):
        return self._step(batch, 'test')

    def validation_step(self, batch, _):
        return self._step(batch, 'valid')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
