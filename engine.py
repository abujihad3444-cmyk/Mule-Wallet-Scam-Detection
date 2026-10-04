from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RISK_WEIGHTS = {
    "anomaly": 0.25,
    "multi_hop": 0.25,
    "fan": 0.10,
    "velocity": 0.10,
    "pass_through": 0.10,
    "cycle": 0.10,
    "connectivity": 0.10,
}

RISK_THRESHOLDS = {"CRITICAL": 80, "HIGH": 60, "MEDIUM": 35}


def generate_data(wallets_count: int = 100, normal_transactions: int = 700, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible synthetic MFS transactions with injected patterns."""
    wallets_count = max(100, int(wallets_count))
    normal_transactions = max(1, int(normal_transactions))
    rng = np.random.default_rng(seed)
    wallets = [f"W{i:03d}" for i in range(1, wallets_count + 1)]
    transactions: List[dict] = []

    def add_tx(
        tx_id: str,
        sender: str,
        receiver: str,
        amount: float,
        timestamp: float,
        channel: str,
        device: str,
        location: str,
    ) -> None:
        transactions.append(
            {
                "transaction_id": tx_id,
                "sender": sender,
                "receiver": receiver,
                "amount": float(amount),
                "timestamp": float(timestamp),
                "transaction_type": "P2P",
                "channel": channel,
                "location": location,
                "device_id": device,
                "beneficiary_type": "PERSONAL",
            }
        )

    # Normal background traffic occupies its own time range.
    for i in range(1, normal_transactions + 1):
        sender, receiver = rng.choice(wallets, 2, replace=False)
        add_tx(
            f"T{i:05d}",
            sender,
            receiver,
            int(rng.integers(100, 5_000)),
            float(rng.integers(0, max(1000, normal_transactions * 2))),
            str(rng.choice(["APP", "USSD", "AGENT"])),
            f"D{int(rng.integers(1, 220)):03d}",
            str(rng.choice(["Dhaka", "Chattogram", "Savar", "Gazipur", "Khulna"])),
        )

    # A. Deliberately injected multi-hop mule chain.
    chain = ["W001", "W005", "W012", "W019", "W027", "W035"]
    for i in range(len(chain) - 1):
        add_tx(
            f"MH{i + 1:03d}",
            chain[i],
            chain[i + 1],
            20_000 - i * 500,
            10_000 + i * 2,
            "APP",
            "D901",
            "Dhaka",
        )

    # B. Fan-out.
    for i, target in enumerate(["W041", "W042", "W043", "W044", "W045"]):
        add_tx(f"FO{i + 1:03d}", "W050", target, 10_000 + i * 500, 10_100 + i * 2, "APP", "D902", "Dhaka")

    # C. Fan-in.
    for i, source in enumerate(["W071", "W072", "W073", "W074", "W075"]):
        add_tx(f"FI{i + 1:03d}", source, "W055", 8_000 + i * 400, 10_200 + i * 2, "APP", "D903", "Chattogram")

    # D. Cyclic flow.
    cycle = ["W060", "W061", "W062", "W063"]
    for i, sender in enumerate(cycle):
        add_tx(f"CY{i + 1:03d}", sender, cycle[(i + 1) % len(cycle)], 15_000, 10_300 + i * 2, "APP", "D904", "Savar")

    # E. Rapid pass-through.
    add_tx("RP001", "W090", "W080", 18_000, 10_400, "APP", "D905", "Dhaka")
    add_tx("RP002", "W080", "W081", 17_500, 10_402, "APP", "D905", "Dhaka")
    add_tx("RP003", "W080", "W082", 500, 10_404, "APP", "D905", "Dhaka")

    # F. Structuring / splitting.
    add_tx("ST001", "W095", "W085", 18_000, 10_500, "USSD", "D906", "Gazipur")
    for i, (target, amount) in enumerate(zip(["W086", "W087", "W088", "W089"], [4_500, 4_300, 4_200, 4_000]), start=2):
        add_tx(f"ST{i:03d}", "W085", target, amount, 10_500 + i, "USSD", "D906", "Gazipur")

    # G. Burst activity.
    for i in range(12):
        target = wallets[(i + 20) % len(wallets)]
        if target == "W099":
            target = "W010"
        add_tx(f"BU{i + 1:03d}", "W099", target, int(rng.integers(6_000, 12_000)), 10_600 + i * 0.5, "APP", "D907", "Khulna")

    df = pd.DataFrame(transactions)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
    return df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build an aggregated directed graph while keeping edge metadata."""
    graph = nx.DiGraph()
    for row in df.itertuples(index=False):
        u, v = row.sender, row.receiver
        amount, ts = float(row.amount), float(row.timestamp)
        if graph.has_edge(u, v):
            data = graph[u][v]
            data["total_amount"] += amount
            data["transactions"] += 1
            data["avg_amount"] = data["total_amount"] / data["transactions"]
            data["min_timestamp"] = min(data["min_timestamp"], ts)
            data["max_timestamp"] = max(data["max_timestamp"], ts)
        else:
            graph.add_edge(
                u,
                v,
                total_amount=amount,
                transactions=1,
                avg_amount=amount,
                min_timestamp=ts,
                max_timestamp=ts,
            )
    return graph


def graph_metrics(graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    pagerank = nx.pagerank(graph, weight="total_amount", max_iter=100)
    betweenness = nx.betweenness_centrality(graph, normalized=True, k=min(60, graph.number_of_nodes()))
    return {
        n: {
            "pagerank": float(pagerank.get(n, 0.0)),
            "betweenness": float(betweenness.get(n, 0.0)),
        }
        for n in graph.nodes()
    }


def transaction_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], List[Tuple[float, float, str]]]:
    lookup: Dict[Tuple[str, str], List[Tuple[float, float, str]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        lookup[(row.sender, row.receiver)].append((float(row.timestamp), float(row.amount), str(row.transaction_id)))
    for values in lookup.values():
        values.sort(key=lambda x: x[0])
    return lookup


def detect_multi_hop(
    df: pd.DataFrame,
    cutoff: int = 4,
    rapid_window: float = 10,
    candidate_window: float | None = None,
    max_paths: int = 150,
) -> pd.DataFrame:
    """Find temporal 2–4 hop paths without relying on static shortest paths.

    A valid path must have non-decreasing edge timestamps. This avoids the old
    failure mode where NetworkX selected a topological shortcut unrelated to
    the actual time-ordered money movement.
    """
    if df.empty:
        return pd.DataFrame()

    cutoff = max(2, min(int(cutoff), 4))
    rapid_window = max(float(rapid_window), 0.0)
    candidate_window = max(float(candidate_window or max(30.0, rapid_window * 3)), rapid_window)
    graph = build_graph(df)
    lookup = transaction_lookup(df)

    adjacency: Dict[str, List[Tuple[str, float, float, str]]] = defaultdict(list)
    for (sender, receiver), values in lookup.items():
        for ts, amount, tx_id in values:
            adjacency[sender].append((receiver, ts, amount, tx_id))
    for sender in adjacency:
        adjacency[sender].sort(key=lambda x: x[1])

    rows: List[dict] = []
    seen = set()
    expansion_budget = 100_000
    expansions = 0

    def dfs(source: str, current: str, path: List[str], chosen: List[Tuple[float, float, str]], start_ts: float) -> None:
        nonlocal expansions
        if expansions >= expansion_budget or len(rows) >= max_paths:
            return
        expansions += 1

        hops = len(path) - 1
        if 2 <= hops <= cutoff and chosen:
            timestamps = [x[0] for x in chosen]
            amounts = [x[1] for x in chosen]
            tx_ids = tuple(x[2] for x in chosen)
            time_span = timestamps[-1] - timestamps[0]
            if time_span < 0 or time_span > candidate_window:
                return

            retention = min(amounts) / max(amounts) if max(amounts) else 0.0
            # Velocity score remains positive at the configured boundary.
            # Example: with a 10-unit rapid window, a 10-unit path scores 50,
            # rather than incorrectly receiving a zero velocity contribution.
            rapid_score = max(0.0, 1.0 - time_span / max(2.0 * rapid_window, 1.0)) * 100 if time_span <= rapid_window else 0.0
            retention_score = retention * 100
            amount_score = min(min(amounts) / 20_000.0 * 100, 100)
            hop_score = min(hops / 4.0 * 100, 100)
            path_score = (
                0.40 * rapid_score
                + 0.25 * retention_score
                + 0.20 * amount_score
                + 0.15 * hop_score
            )
            pattern = "RAPID_MULTI_HOP" if time_span <= rapid_window else "TEMPORAL_MULTI_HOP"
            if hops >= 3 and time_span <= rapid_window and retention >= 0.85:
                pattern = "POTENTIAL_MULE_CHAIN"

            key = (source, tuple(path), tx_ids)
            if key not in seen:
                seen.add(key)
                rows.append(
                    {
                        "source": source,
                        "destination": current,
                        "hops": hops,
                        "path": " → ".join(path),
                        "transaction_ids": ", ".join(tx_ids),
                        "total_amount": round(float(sum(amounts)), 2),
                        "flow_amount": round(float(min(amounts)), 2),
                        "min_amount": round(float(min(amounts)), 2),
                        "amount_retention_ratio": round(float(retention), 3),
                        "time_span": round(float(time_span), 3),
                        "average_inter_hop_time": round(float(np.mean(np.diff(timestamps))) if len(timestamps) > 1 else 0.0, 3),
                        "rapid_transfer": "YES" if time_span <= rapid_window else "NO",
                        "pattern": pattern,
                        "path_score": round(float(path_score), 2),
                    }
                )

        if hops >= cutoff:
            return

        prev_ts = chosen[-1][0] if chosen else -np.inf
        for nxt, ts, amount, tx_id in adjacency.get(current, []):
            if nxt in path:
                continue
            if ts < prev_ts:
                continue
            if ts - start_ts > candidate_window:
                break
            dfs(source, nxt, path + [nxt], chosen + [(ts, amount, tx_id)], start_ts)
            if expansions >= expansion_budget or len(rows) >= max_paths:
                return

    for source in graph.nodes():
        for nxt, ts, amount, tx_id in adjacency.get(source, []):
            dfs(source, nxt, [source, nxt], [(ts, amount, tx_id)], ts)
            if expansions >= expansion_budget or len(rows) >= max_paths:
                break
        if expansions >= expansion_budget or len(rows) >= max_paths:
            break

    columns = [
        "source",
        "destination",
        "hops",
        "path",
        "transaction_ids",
        "total_amount",
        "flow_amount",
        "min_amount",
        "amount_retention_ratio",
        "time_span",
        "average_inter_hop_time",
        "rapid_transfer",
        "pattern",
        "path_score",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows)
    pattern_priority = {"POTENTIAL_MULE_CHAIN": 3, "RAPID_MULTI_HOP": 2, "TEMPORAL_MULTI_HOP": 1}
    result["_pattern_priority"] = result["pattern"].map(pattern_priority).fillna(0)
    return (
        result
        .sort_values(["_pattern_priority", "path_score", "flow_amount", "hops"], ascending=[False, False, False, False])
        .drop(columns=["_pattern_priority"])
        .head(max_paths)
        .reset_index(drop=True)
    )


def detect_cycles(
    graph: nx.DiGraph,
    df: pd.DataFrame | None = None,
    max_cycles: int = 25,
    max_len: int = 6,
    time_window: float = 30,
    min_edge_amount: float = 5_000,
) -> List[List[str]]:
    """Detect short cycles and filter them using transaction evidence.

    Static graph topology alone creates many ordinary cycles in a dense network.
    This version requires each cycle edge to have transaction activity in a
    compact time window and to carry a meaningful amount.
    """
    if graph.number_of_nodes() == 0:
        return []

    lookup = transaction_lookup(df) if df is not None and not df.empty else {}
    scored: List[Tuple[float, List[str]]] = []
    seen = set()

    try:
        candidates = nx.simple_cycles(graph, length_bound=max_len)
    except TypeError:
        candidates = nx.simple_cycles(graph)

    for cyc in candidates:
        if not 2 <= len(cyc) <= max_len:
            continue

        edge_values: List[Tuple[float, float]] = []
        valid = True
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            vals = lookup.get((u, v), [])
            if not vals:
                valid = False
                break
            # Use the strongest transaction on each edge for the cycle evidence.
            best = max(vals, key=lambda x: x[1])
            edge_values.append((best[0], best[1]))
        if not valid:
            continue

        timestamps = [x[0] for x in edge_values]
        amounts = [x[1] for x in edge_values]
        time_span = max(timestamps) - min(timestamps)
        if time_span > time_window or min(amounts) < min_edge_amount:
            continue

        canonical_rotations = [tuple(cyc[i:] + cyc[:i]) for i in range(len(cyc))]
        key = min(canonical_rotations)
        if key in seen:
            continue
        seen.add(key)

        amount_consistency = min(amounts) / max(amounts) if max(amounts) else 0.0
        score = amount_consistency * 60 + min(max(amounts) / 20_000, 1.0) * 40
        scored.append((float(score), cyc))

        if len(scored) >= max_cycles * 4:
            break

    scored.sort(key=lambda x: x[0], reverse=True)
    return [cyc for _, cyc in scored[:max_cycles]]


def create_features(
    df: pd.DataFrame,
    graph: nx.DiGraph,
    paths: pd.DataFrame,
    cycles: List[List[str]],
    gmetrics: Dict[str, Dict[str, float]],
    rapid_window: float = 10,
) -> pd.DataFrame:
    cycle_nodes = {n for cycle in cycles for n in cycle}
    wallets = sorted(set(df["sender"]) | set(df["receiver"]))
    sent_groups = {k: g for k, g in df.groupby("sender", sort=False)}
    recv_groups = {k: g for k, g in df.groupby("receiver", sort=False)}
    empty = df.iloc[0:0]

    path_rows = defaultdict(list)
    if not paths.empty:
        for r in paths.itertuples(index=False):
            path_score = float(getattr(r, "path_score", 0.0))
            for node in set(str(r.path).split(" → ")):
                path_rows[node].append(r)

    rows = []
    for wallet in wallets:
        sent = sent_groups.get(wallet, empty)
        received = recv_groups.get(wallet, empty)
        all_tx = pd.concat([sent, received], ignore_index=True)
        total_sent = float(sent["amount"].sum())
        total_received = float(received["amount"].sum())
        pass_through = min(total_sent / total_received, 1.0) if total_received > 0 else 0.0

        # Pair each outgoing transfer with the latest earlier incoming transfer.
        gaps: List[float] = []
        rapid_count = 0
        if not sent.empty and not received.empty:
            incoming = sorted(received["timestamp"].astype(float).tolist())
            for out_ts in sorted(sent["timestamp"].astype(float).tolist()):
                prior = [ts for ts in incoming if ts <= out_ts]
                if prior:
                    gap = float(out_ts - prior[-1])
                    gaps.append(gap)
                    if gap <= rapid_window:
                        rapid_count += 1

        largest_in = float(received["amount"].max()) if not received.empty else 0.0
        structuring = 0.0
        if largest_in > 0 and len(sent) >= 3:
            smaller = int((sent["amount"] < largest_in).sum())
            structuring = min(smaller / 5.0 * 100, 100)

        related_paths = path_rows.get(wallet, [])
        max_path_score = max([float(getattr(p, "path_score", 0.0)) for p in related_paths], default=0.0)
        rapid_path_count = sum(str(getattr(p, "rapid_transfer", "NO")) == "YES" for p in related_paths)
        multi_hop_count = len(related_paths)

        rows.append(
            {
                "wallet": wallet,
                "total_sent": total_sent,
                "total_received": total_received,
                "outgoing_count": int(len(sent)),
                "incoming_count": int(len(received)),
                "total_transactions": int(len(all_tx)),
                "unique_receivers": int(sent["receiver"].nunique()),
                "unique_senders": int(received["sender"].nunique()),
                "avg_transaction": float(all_tx["amount"].mean()) if not all_tx.empty else 0.0,
                "median_transaction": float(all_tx["amount"].median()) if not all_tx.empty else 0.0,
                "max_transaction": float(all_tx["amount"].max()) if not all_tx.empty else 0.0,
                "min_transaction": float(all_tx["amount"].min()) if not all_tx.empty else 0.0,
                "pass_through": pass_through,
                "retained_amount": max(total_received - total_sent, 0.0),
                "holding_ratio": max(total_received - total_sent, 0.0) / total_received if total_received else 0.0,
                "rapid_transfer_count": rapid_count,
                "rapid_transfer_ratio": rapid_count / len(sent) if len(sent) else 0.0,
                "min_time_gap": min(gaps) if gaps else 9999.0,
                "avg_time_gap": float(np.mean(gaps)) if gaps else 9999.0,
                "fan_out_score": min(sent["receiver"].nunique() / 5.0 * 100, 100),
                "fan_in_score": min(received["sender"].nunique() / 5.0 * 100, 100),
                "fan_score": min(max(sent["receiver"].nunique(), received["sender"].nunique()) / 5.0 * 100, 100),
                "structuring_score": structuring,
                "multi_hop_count": multi_hop_count,
                "rapid_multi_hop_count": rapid_path_count,
                "multi_hop_exposure": max_path_score,
                "rapid_path_score": min(rapid_path_count / 2.0 * 100, 100),
                "cycle_member": wallet in cycle_nodes,
                "in_degree": graph.in_degree(wallet),
                "out_degree": graph.out_degree(wallet),
                "weighted_in_degree": float(sum(graph[u][wallet]["total_amount"] for u in graph.predecessors(wallet))),
                "weighted_out_degree": float(sum(graph[wallet][v]["total_amount"] for v in graph.successors(wallet))),
                "pagerank": gmetrics.get(wallet, {}).get("pagerank", 0.0),
                "betweenness": gmetrics.get(wallet, {}).get("betweenness", 0.0),
            }
        )
    return pd.DataFrame(rows)


def anomaly_detection(features: pd.DataFrame, contamination: float = 0.10, estimators: int = 200) -> pd.DataFrame:
    result = features.copy()
    cols = [
        "total_sent",
        "total_received",
        "outgoing_count",
        "incoming_count",
        "total_transactions",
        "unique_receivers",
        "unique_senders",
        "avg_transaction",
        "median_transaction",
        "max_transaction",
        "pass_through",
        "holding_ratio",
        "rapid_transfer_count",
        "rapid_transfer_ratio",
        "fan_out_score",
        "fan_in_score",
        "structuring_score",
        "in_degree",
        "out_degree",
        "weighted_in_degree",
        "weighted_out_degree",
        "pagerank",
        "betweenness",
    ]
    x = result[cols].replace([np.inf, -np.inf], 0).fillna(0)
    x_scaled = StandardScaler().fit_transform(x)
    model = IsolationForest(
        contamination=float(np.clip(contamination, 0.01, 0.49)),
        n_estimators=max(50, int(estimators)),
        random_state=42,
    )
    result["anomaly"] = model.fit_predict(x_scaled) == -1
    raw = -model.score_samples(x_scaled)
    lo, hi = float(raw.min()), float(raw.max())
    result["anomaly_score"] = 0.0 if hi == lo else np.round((raw - lo) / (hi - lo) * 100, 2)
    return result


def calculate_risk(features: pd.DataFrame, paths: pd.DataFrame, cycles: List[List[str]]) -> pd.DataFrame:
    result = features.copy()
    cycle_nodes = {n for cycle in cycles for n in cycle}
    components = {
        "anomaly": result["anomaly_score"].clip(0, 100),
        "multi_hop": result["multi_hop_exposure"].clip(0, 100),
        "fan": result["fan_score"].clip(0, 100),
        "velocity": np.maximum(
            result["rapid_transfer_ratio"] * 100,
            result["rapid_path_score"],
        ).clip(0, 100),
        "pass_through": (result["pass_through"] * 100).clip(0, 100),
        "cycle": result["wallet"].isin(cycle_nodes).astype(float) * 100,
        "connectivity": ((result["in_degree"] + result["out_degree"]) / 10 * 100).clip(0, 100),
    }
    risk = sum(components[k] * RISK_WEIGHTS[k] for k in RISK_WEIGHTS)
    for name, values in components.items():
        result[f"risk_{name}"] = values.round(2)
    result["risk_score"] = risk.clip(0, 100).round(2)
    result["risk_level"] = pd.cut(
        result["risk_score"],
        bins=[-1, 34.9999, 59.9999, 79.9999, 100.0001],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    ).astype(str)
    return result.sort_values("risk_score", ascending=False).reset_index(drop=True)


def compute_pipeline(
    wallets: int = 100,
    normal_transactions: int = 700,
    seed: int = 42,
    contamination: float = 0.10,
    estimators: int = 200,
    rapid_window: int = 10,
    max_hops: int = 4,
):
    df = generate_data(wallets, normal_transactions, seed)
    graph = build_graph(df)
    cycles = detect_cycles(graph, df=df)
    paths = detect_multi_hop(df, max_hops, rapid_window, max_paths=150)
    gmetrics = graph_metrics(graph)
    features = create_features(df, graph, paths, cycles, gmetrics, rapid_window)
    scored = anomaly_detection(features, contamination, estimators)
    risk = calculate_risk(scored, paths, cycles)
    return df, graph, cycles, paths, risk
