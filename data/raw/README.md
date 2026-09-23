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
