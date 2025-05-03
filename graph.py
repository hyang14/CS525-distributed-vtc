import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_trace(path: str) -> pd.DataFrame:
    """
    Read one debug.jsonl file and return a flat DataFrame with
    one row per finished request.
    """
    rows = []
    with Path(path).open() as f:
        for line in f:
            obj = json.loads(line)
            for r in obj["result"]["responses"]:
                rows.append(r)

    df = pd.DataFrame(rows)

    # when did the request finish?
    df["t_finish"] = df["req_time"] + df["request_latency"]

    # how many tokens was the server’s work for this request?
    # – pick a definition that makes sense for you
    df["service_tokens"] = df["prompt_len"] + df["output_len"]
    return df.sort_values("t_finish")


def cum_service_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a trace with *two* clients, compute the |Δ service|
    time‑series.
    Returns a DataFrame with columns  ["t_finish", "abs_gap"].
    """
    # cumulative service for each client
    df["cum_service"] = (
        df.groupby("adapter_dir")["service_tokens"]
          .cumsum()
    )

    # pivot so every row has both clients’ totals
    pivot = (
        df[["t_finish", "adapter_dir", "cum_service"]]
        .pivot_table(index="t_finish",
                     columns="adapter_dir",
                     values="cum_service")
        .sort_index()
        .ffill()         # carry the last value forward
    )

    # If you have more than two clients, replace this with max‑min or
    # any fairness metric you prefer.
    clients = pivot.columns.tolist()
    assert len(clients) == 2, "the helper assumes exactly two clients"
    pivot["abs_gap"] = (pivot[clients[0]] - pivot[clients[1]]).abs()
    pivot = pivot.reset_index()
    return pivot[["t_finish", "abs_gap"]]


# ------------------------------------------------------------------
#  Load the traces for the two schedulers you want to compare
#     (replace with your real paths)

traces = {
    "VTC":  load_trace("result/debug.jsonl"),
    "FCFS": load_trace("result/fcfs_debug.jsonl"),
}

# 2⃣  Build the gap curves
curves = {
    name: cum_service_gap(df)
    for name, df in traces.items()
}

# ------------------------------------------------------------------
# 3⃣  Plot — one line per algorithm
plt.figure(figsize=(3.3, 3.3), dpi=160)

for name, curve in curves.items():
    if name == "VTC":
        plt.plot(
            curve["t_finish"], curve["abs_gap"],
            marker="v", markersize=4, linewidth=1.3,
            label=name)
    else:  # FCFS
        plt.plot(
            curve["t_finish"], curve["abs_gap"],
            marker="s", markersize=4, linewidth=1.3,
            label=name)

plt.xlabel("Time (s)")
plt.ylabel("Absolute Difference in Service")
plt.legend()
plt.tight_layout()
plt.savefig("abs_gap_over_time.svg")   # PDF/PGF/etc. work too
plt.show()

