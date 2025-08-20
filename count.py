import hydra
import polars as pl
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="clax/config/", config_name="config")
def main(config: DictConfig):
    unique_count = (
        pl.scan_parquet("/ivi/ilps/personal/phager/clax-datasets/baidu-ultr/*.parquet")
        .select(pl.col("query_doc_ids").list.explode().unique().len())
        .collect(streaming=True)  # Enable streaming mode
        .item()
    )
    print(unique_count)


if __name__ == "__main__":
    main()
