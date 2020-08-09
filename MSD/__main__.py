from MSD.trainer.trainer import NeuralNetworkLearner
from MSD.settings.config import MAX_EPOCH, NEPTUNE_TOKEN, BATCH_SIZE
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.logging.neptune import NeptuneLogger
import pytorch_lightning as pl

import neptune

neptune.init('wsz/MSD-net', api_token=NEPTUNE_TOKEN)
neptune.create_experiment('wsz/MSD')
neptune_logger = NeptuneLogger(
    api_key=NEPTUNE_TOKEN,
    project_name="wsz/MSD-net",
    params={"max_epochs": MAX_EPOCH,
            "batch_size": BATCH_SIZE},
)


def main():
    model = NeuralNetworkLearner()
    trainer = Trainer(max_epochs=MAX_EPOCH, logger=neptune_logger, progress_bar_refresh_rate=1)
    trainer.fit(model)


if __name__ == '__main__':
    main()

