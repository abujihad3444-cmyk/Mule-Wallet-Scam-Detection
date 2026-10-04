from engine import compute_pipeline


def main() -> None:
    df, graph, cycles, paths, risk = compute_pipeline()

    rapid = int((paths["rapid_transfer"] == "YES").sum()) if not paths.empty else 0
    high = int((risk["risk_level"] == "HIGH").sum())
    critical = int((risk["risk_level"] == "CRITICAL").sum())

    expected = "W001 → W005 → W012 → W019 → W027"
    chain_found = bool(paths["path"].astype(str).str.contains(expected, regex=False).any()) if not paths.empty else False
    expected_cycle = {"W060", "W061", "W062", "W063"}
    cycle_found = any(expected_cycle.issubset(set(c)) for c in cycles)

    print("PIPELINE CHECK: OK")
    print(f"transactions={len(df)}")
    print(f"nodes={graph.number_of_nodes()}")
    print(f"cycles={len(cycles)}")
    print(f"multi_hop_paths={len(paths)}")
    print(f"rapid_paths={rapid}")
    print(f"high={high}")
    print(f"critical={critical}")
    print(f"intended_mule_chain_detected={chain_found}")
    print(f"intended_cycle_detected={cycle_found}")

    print("\ntop_paths:")
    if paths.empty:
        print("none")
    else:
        print(paths[["path", "hops", "pattern", "time_span", "amount_retention_ratio", "path_score"]].head(10).to_string(index=False))

    print("\ntop_wallets:")
    print(risk[["wallet", "risk_score", "risk_level"]].head(10).to_string(index=False))

    if not chain_found or not cycle_found:
        raise SystemExit("Validation failed: injected demonstration pattern was not detected.")


if __name__ == "__main__":
    main()
