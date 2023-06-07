from hydra.utils import instantiate


def configure_optimizers_and_schedulers(self):
    optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)

    if self.scheduler_config is None:
        return optimizer

    self.scheduler_config["optimizer"] = optimizer
    scheduler = instantiate(self.scheduler_config)

    return (
        [optimizer],
        [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "reduce_on_plateau": False,
            }
        ],
    )
