from torch.utils.data import DataLoader
from tqdm import tqdm

from clix.datasets import YandexDataset


def main():
    dataset = YandexDataset("data/yandex.csv")
    loader = DataLoader(dataset, batch_size=512, collate_fn=dataset.collate_fn)

    for batch in tqdm(loader):
        pass


if __name__ == "__main__":
    main()
