from torch.utils.data import DataLoader

class UAPDataLoader(DataLoader):
    def __init__(self, dataset, uap, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.uap = uap

    def __iter__(self):
        for batch in super().__iter__():
            images, labels = batch
            perturbed_images = images + self.uap
            yield perturbed_images, labels
