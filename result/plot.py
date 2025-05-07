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
    # – pick a definition that makes sense for you
    df["service_tokens"] = df["prompt_len"] + df["output_len"]
    return df.sort_values("t_finish")

# ---------------------------------------------------------------
# Helper: down‑sample a (t_finish, abs_gap) curve to one point
# every STEP seconds.
#
# t_finish is assumed to be a numeric “seconds since start” column.
# If yours is a real timestamp, convert first:
#     curve["t_finish"] = pd.to_datetime(curve["t_finish"])
#     ... then use .dt.total_seconds() or resample("5S") instead.
# ---------------------------------------------------------------

def downsample_curve(curve: pd.DataFrame, *, step: int = 2,
                     agg: str = "mean") -> pd.DataFrame:
    """
    Collapse `curve` so we keep only one point every <step> seconds.

    Parameters
    ----------
    curve : DataFrame with columns ["t_finish", "abs_gap"]
    step  : size of the time bucket, in seconds
    agg   : how to combine points inside a bucket
            ("mean", "last", "max", "median", …)

    Returns
    -------
    DataFrame with the same two columns, but far fewer rows.
    """
    # assign each row to a bucket boundary 0, step, 2*step, …
    curve = curve.copy()
    curve["t_bin"] = (curve["t_finish"] // step) * step

    if agg == "mean":
        df_out = (curve.groupby("t_bin", as_index=False)["abs_gap"]
                         .mean())
    elif agg == "last":
        df_out = (curve.sort_values("t_finish")
                         .groupby("t_bin", as_index=False)["abs_gap"]
                         .last())
    else:                          # feel free to extend
        df_out = (curve.groupby("t_bin", as_index=False)["abs_gap"]
                         .agg(agg))

    df_out = df_out.rename(columns={"t_bin": "t_finish"})
    return df_out


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
# 1️⃣  Load the traces for the two schedulers you want to compare
#     (replace with your real paths)
traces = {
    "VTC":  load_trace("/home/lyang/cs525/CS525-distributed-vtc/result/adj_vtc.jsonl"),
    "FCFS": load_trace("/home/lyang/cs525/CS525-distributed-vtc/result/fcfs_result.jsonl"),
    "LATQ": load_trace("/home/lyang/cs525/CS525-distributed-vtc/result/adj_latq.jsonl"),
}

# 2️⃣  Build the gap curves
# curves = {
#     name: cum_service_gap(df)
#     for name, df in traces.items()
# }
# down sample the curves to one point every STEP seconds
STEP = 2          # seconds ‑‑ tweak until the plot looks clean
AGG  = "mean"     # or "last", "median", …

# 2️⃣  Build the gap curves, then immediately thin them
curves = {
    name: downsample_curve(cum_service_gap(df), step=STEP, agg=AGG)
    for name, df in traces.items()
}

# ------------------------------------------------------------------
# 3️⃣  Plot — one line per algorithm
plt.figure(figsize=(3.3, 3.3), dpi=160)

for name, curve in curves.items():
    if name == "VTC":
        plt.plot(
            curve["t_finish"], curve["abs_gap"],
            marker="v", markersize=4, linewidth=1.3,
            label=name)
    elif name == "LATQ":
        plt.plot(
            curve["t_finish"], curve["abs_gap"],
            marker="o", markersize=4, linewidth=1.3,
            label=name)
    # else:  # FCFS
    #     plt.plot(
    #         curve["t_finish"], curve["abs_gap"],
    #         marker="s", markersize=4, linewidth=1.3,
    #         label=name)

plt.xlabel("Time (s)")
plt.ylabel("Absolute Difference in Service")
plt.legend()
plt.xlim(0, 150)
plt.ylim(0, 1000)
plt.tight_layout()
plt.savefig("abs_gap_over_time_adj.svg")   # PDF/PGF/etc. work too
plt.show()
