"""Build a capture containing the one host CTU-13 does not have.

Every infected host in CTU-13 is malicious in every window it appears, so the
risk head has never seen a machine become compromised and its output is
saturated: across all 13 scenarios and 1,530 hosts, every host reads either
~1.00 or <0.05 and nothing sits in between.

This splices one host's clean traffic in front of another's botnet traffic and
gives them a single source address, so the file contains a machine that is
benign for the first stretch and compromised afterwards. The ground-truth
labels are carried through untouched, which means the capture's own shading
marks where the infection really starts and the model's risk line can be
checked against it rather than taken on trust.

It is a constructed file and has to be introduced as one.
"""

import sys
from pathlib import Path

import pandas as pd

SRC = Path("data/raw/CTU-13-Dataset/1/capture20110810.binetflow")
OUT = Path("data/samples/host-becomes-infected.csv")
CLEAN_HOST = "147.32.84.171"     # a normal workstation, 26k flows
DIRTY_HOST = "147.32.84.165"     # the Neris-infected machine
NEW_IP = "10.0.0.7"
CLEAN_MINUTES = 45
DIRTY_MINUTES = 55


def main() -> None:
    print(f"reading {SRC} ...")
    df = pd.read_csv(SRC, usecols=[
        "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr",
        "Dport", "State", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes",
        "Label",
    ])
    ts = pd.to_datetime(df["StartTime"], format="%Y/%m/%d %H:%M:%S.%f",
                        errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"])

    halves = []
    for host, span in ((CLEAN_HOST, CLEAN_MINUTES),
                       (DIRTY_HOST, DIRTY_MINUTES)):
        part = df[df["SrcAddr"] == host].sort_values("_ts")
        if part.empty:
            raise SystemExit(f"no flows for {host}")
        start = part["_ts"].iloc[0]
        part = part[part["_ts"] < start + pd.Timedelta(minutes=span)]
        mal = part["Label"].str.contains("Botnet", case=False, na=False).sum()
        print(f"  {host}: {len(part):>6} flows over {span}m, "
              f"{mal} labelled botnet")
        halves.append(part)

    clean, dirty = halves
    # Butt the second half directly onto the end of the first, so the join is a
    # continuous timeline with no gap for the model to read as a break.
    gap = pd.Timedelta(seconds=60)
    shift = (clean["_ts"].iloc[-1] + gap) - dirty["_ts"].iloc[0]
    dirty = dirty.assign(_ts=dirty["_ts"] + shift)

    both = pd.concat([clean, dirty]).sort_values("_ts")
    both = both.assign(
        SrcAddr=NEW_IP,
        StartTime=both["_ts"].dt.strftime("%Y/%m/%d %H:%M:%S.%f"),
    ).drop(columns=["_ts"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    both.to_csv(OUT, index=False)

    span = (len(clean), len(dirty))
    print(f"\nwrote {OUT}  ({len(both)} flows, {OUT.stat().st_size // 1024} KB)")
    print(f"  first  {span[0]} flows: clean traffic from {CLEAN_HOST}")
    print(f"  then   {span[1]} flows: botnet traffic from {DIRTY_HOST}")
    print(f"  both rewritten to source {NEW_IP}")
    print(f"  changeover at minute ~{CLEAN_MINUTES}")


if __name__ == "__main__":
    sys.exit(main())
