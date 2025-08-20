import glob
import os

import hydra
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="clax/config/", config_name="config")
def main(config: DictConfig):
    unique_ids = set()
    print(f"Current working directory: {os.getcwd()}")
    files = glob.glob("/ivi/ilps/personal/phager/clax-datasets/baidu-ultr/*")

    print(f"Processing {len(files)} files...")

    for file in files:
        df = pd.read_parquet(file, columns=["query_doc_ids"])
        unique_ids.update(df["query_doc_ids"].explode().to_list())

        print(
            f"Processed {file}: {len(df)} rows, running total: {len(unique_ids)} unique IDs"
        )

    print(len(unique_ids))


if __name__ == "__main__":
    main()
