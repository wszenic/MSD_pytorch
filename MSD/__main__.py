from MSD.trainer.trainer import NeuralNetworkLearner
from pytorch_lightning.trainer import Trainer


def main():
    model = NeuralNetworkLearner()
    trainer = Trainer(max_epochs=5)
    trainer.fit(model)


if __name__ == '__main__':
    main()

