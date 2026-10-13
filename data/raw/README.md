# Raw Dataset Directory (CTU-13)

Place original CTU-13 NetFlow `.binetflow` or PCAP files into this directory.

Recommended CTU-13 captures:
- Scenario 1: `capture20110810.binetflow` (Neris botnet)
- Scenario 2: `capture20110811.binetflow` (Neris botnet)
- Scenario 9: `capture20110817.binetflow` (NerIS / botnet traffic)
- Scenario 10: `capture20110818.binetflow` (Rbot)

Once downloaded, run:
```bash
python data/reduce_dataset.py
```
This extracts relevant network features and outputs `data/processed/NetForecast_clean_dataset.csv`.

---

# Multi-File Batch Preprocessing (UNSW-NB15 / Large Datasets)

Place raw dataset files (e.g. `UNSW-NB15_1.csv` ... `UNSW-NB15_4.csv` or PCAPs) into a subdirectory such as `data/raw/UNSW-NB15/`.

To process all files into Snappy Parquet chunks under a strict `< 2.5 GB` RAM boundary:
```bash
python src/batch_preprocess.py --input-dir data/raw/UNSW-NB15 --output-dir data/processed/chunks --max-ram-gb 2.5
```
This outputs chunked Parquet files (`data/processed/chunks/chunk_0000.parquet`...) and updates the feature normalizer at `data/processed/scaler.joblib`.

