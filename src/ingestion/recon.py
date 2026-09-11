from collections import Counter
from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    r"C:\Users\umang\.cache\kagglehub\datasets"
    r"\thoughtvector\customer-support-on-twitter"
    r"\versions\10\twcs\twcs.csv"
)

CHUNK_SIZE = 100_000


def main() -> None:
    total_rows = 0
    inbound_count = 0
    outbound_count = 0

    authors = Counter()
    inbound_authors = Counter()
    outbound_authors = Counter()

    print(f"Reading: {DATASET_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print()

    for chunk_number, df in enumerate(
        pd.read_csv(
            DATASET_PATH,
            chunksize=CHUNK_SIZE,
            usecols=["author_id", "inbound"],
        ),
        start=1,
    ):
        total_rows += len(df)

        inbound_count += int(df["inbound"].sum())
        outbound_count += int((~df["inbound"]).sum())

        authors.update(df["author_id"].dropna().astype(str))
        inbound_authors.update(
            df.loc[df["inbound"], "author_id"].dropna().astype(str)
        )
        outbound_authors.update(
            df.loc[~df["inbound"], "author_id"].dropna().astype(str)
        )

        print(
            f"Processed chunk {chunk_number:>2} | "
            f"{total_rows:,} rows"
        )

    print("\n" + "=" * 60)
    print("DATASET RECONNAISSANCE")
    print("=" * 60)

    print(f"\nTotal tweets:      {total_rows:,}")
    print(f"Inbound tweets:    {inbound_count:,}")
    print(f"Outbound tweets:   {outbound_count:,}")
    print(f"Unique authors:    {len(authors):,}")

    print("\nTop 25 authors by total tweets:")
    for author, count in authors.most_common(25):
        print(f"  {author:<30} {count:>10,}")

    print("\nTop 25 inbound authors:")
    for author, count in inbound_authors.most_common(25):
        print(f"  {author:<30} {count:>10,}")

    print("\nTop 25 outbound authors:")
    for author, count in outbound_authors.most_common(25):
        print(f"  {author:<30} {count:>10,}")


if __name__ == "__main__":
    main()