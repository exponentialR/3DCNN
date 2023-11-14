import pytorch_lightning as pl
from pl_model import Example3DCNN


class Tester:
    def __init__(self, args, configfile_head):
        self.args = args
        self.configfile_head = configfile_head

    def test(self, test_loader):
        test_model = self.configfile_head[f'output_model']
        learner = Example3DCNN.load_from_checkpoint(checkpoint_path=test_model)
        trainer = pl.Trainer(accelerator='gpu', devices=1, strategy='ddp', num_nodes=1)
        print(f'STARTING TEST.......')
        trainer.test(model=learner, dataloaders=test_loader)
        print('TESTING DONE')
