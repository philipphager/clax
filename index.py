import json
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm


def build_index(path: Path, per_partition: int, query_indicator: bytes = b"\tQ\t"):
    print(
        f"Creating index for {path}, storing access every {per_partition:_} sessions..."
    )
    partition_begins = []
    partition_ends = []
    total_sessions = 0
    bytes_since_last_index = 0

    with open(path, "rb") as f:
        file_size = path.stat().st_size
        progress_bar = tqdm(total=file_size, unit="B", unit_scale=True)

        while True:
            byte_position = f.tell()
            line = f.readline()

            if not line:
                break

            bytes_since_last_index += len(line)

            if line.find(query_indicator) != -1:
                if total_sessions % per_partition == 0:
                    if len(partition_begins) > len(partition_ends):
                        partition_ends.append(byte_position)

                    partition_begins.append(byte_position)

                    progress_bar.update(bytes_since_last_index)
                    bytes_since_last_index = 0

                total_sessions += 1

        partition_ends.append(byte_position)
        progress_bar.close()

    return {
        "sessions_per_partition": per_partition,
        "total_sessions": total_sessions,
        "partition_begins": partition_begins,
        "partition_ends": partition_ends,
        "total_partitions": len(partition_begins),
    }


if __name__ == "__main__":
    path = Path("data/wscd-2012/YandexClicks.txt")
    index = build_index(path, per_partition=100_000)
    json.dump(index, open("data/wscd-2012/index.json", "w"))
