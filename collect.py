from pathlib import Path

import pandas as pd


def main():
    experiment_dfs = []
    experiment = "3-baidu-ultr"
    stage = "test"

    for random_state in [1, 2, 3]:
        result_dir = Path(f"results/{experiment}/{random_state}")

        if not result_dir.exists():
            continue

        dfs = []

        for file in result_dir.glob(f"{stage}_*.csv"):
            df = pd.read_csv(file)
            dfs.append(df)

        if len(dfs) > 0:
            experiment_df = pd.concat(dfs, ignore_index=True)
            experiment_df["random_state"] = random_state
            experiment_dfs.append(experiment_df)

    experiment_df = pd.concat(experiment_dfs, ignore_index=True)
    experiment_df.to_csv(f"results/{experiment}/{stage}.csv", index=False)


if __name__ == "__main__":
    main()
