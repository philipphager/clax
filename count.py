import glob

import hydra
import polars as pl
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="clax/config/", config_name="config")
def main(config: DictConfig):
    unique_ids = set()
    files = glob.glob("clax-datasets/baidu-ultr/*.parquet")

    print(f"Processing {len(files)} files...")

    for file in files:
        df = pl.read_parquet(file)
        file_ids = (
            df.select(pl.col("query_doc_ids").list.explode()).to_series().to_list()
        )

        unique_ids.update(file_ids)
        print(
            f"Processed {file}: {len(df)} rows, running total: {len(unique_ids)} unique IDs"
        )

    print(len(unique_ids))


if __name__ == "__main__":
    main()
