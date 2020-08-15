from MSD.trainer.trainer import NeuralNetworkLearner
from MSD.settings.config import MAX_EPOCH, NEPTUNE_TOKEN, BATCH_SIZE
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.logging.neptune import NeptuneLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

import neptune


neptune.init('wsz/MSD-net', api_token=NEPTUNE_TOKEN)

checkpoint_callback = ModelCheckpoint(filepath='./{epoch}-{val_loss:.2f}')

def main():
    model = NeuralNetworkLearner()
    trainer = Trainer(max_epochs=MAX_EPOCH,
                      checkpoint_callback=checkpoint_callback,
                      gpus=-1)
    trainer.fit(model)

if __name__ == '__main__':
    main()
