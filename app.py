import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from engine import (
    RISK_WEIGHTS,
    build_graph,
    calculate_risk,
    compute_pipeline,
    create_features,
    detect_cycles,
    detect_multi_hop,
    generate_data,
    graph_metrics,
    anomaly_detection,
)

st.set_page_config(page_title="Mule Wallet Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {padding: 0.75rem 0.8rem; border: 1px solid rgba(128,128,128,.18); border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Mule Wallet Detection")
st.subheader("Graph-Based Multi-Hop Mule Wallet & Scam Detection System")
st.caption("Synthetic MFS fraud-intelligence prototype: Graph + ML + Explainable Risk + Optional GenAI")
st.warning("Synthetic demonstration only. Risk scores are prioritization signals, not a fraud verdict or probability.")
st.markdown("### 🎯 Who is this for?")
u1, u2, u3 = st.columns(3)

with u1:
    st.markdown("**TARGET USER**")
    st.write("MFS Fraud & Risk Analysts")

with u2:
    st.markdown("**PROBLEM**")
    st.write(
        "Suspicious activity can move across multiple connected wallets, "
        "making isolated-wallet rules difficult to investigate."
    )

with u3:
    st.markdown("**SOLUTION**")
    st.write(
        "Graph-based intelligence that detects suspicious money flows, "
        "prioritizes risky wallets, and supports analyst investigation."
    )

st.info(
    "🔎 Workflow: Detect → Trace → Explain → Investigate → Human Review"
)

# --------------------------- sidebar -------------------------
st.sidebar.header("Controls")
with st.sidebar.expander("Dataset", True):
    wallet_count = st.slider("Wallets", 100, 300, 100, 10)
    normal_tx = st.slider("Normal transactions", 300, 3000, 700, 100)
    seed = st.number_input("Random seed", 1, 999999, 42)
with st.sidebar.expander("Detection", True):
    contamination = st.slider("IsolationForest contamination", 0.02, 0.25, 0.10, 0.01)
    estimators = st.slider("IsolationForest trees", 100, 500, 200, 50)
    rapid_window = st.slider("Rapid transfer window", 2, 30, 10, 1)
    max_hops = st.slider("Maximum hops", 2, 4, 4, 1)
with st.sidebar.expander("Graph", True):
    min_edge = st.number_input("Minimum edge amount (৳)", 0, 50000, 0, 500)
    suspicious_only = st.checkbox("Show high/critical network only", False)

@st.cache_data(show_spinner=False)
def get_results(wallets, txs, seed_value, contam, trees, rapid, hops):
    return compute_pipeline(wallets, txs, seed_value, contam, trees, rapid, hops)

with st.spinner("Running graph and anomaly engines..."):
    df, graph, cycles, multi_hop, risk = get_results(wallet_count, normal_tx, int(seed), contamination, estimators, rapid_window, max_hops)

high = int((risk["risk_level"] == "HIGH").sum())
critical = int((risk["risk_level"] == "CRITICAL").sum())
rapid_count = int((multi_hop["rapid_transfer"] == "YES").sum()) if not multi_hop.empty else 0

# --------------------------- KPI ------------------------------
k = st.columns(6)
k[0].metric("Wallets", graph.number_of_nodes())
k[1].metric("Transactions", len(df))
k[2].metric("High Risk", high)
k[3].metric("Critical", critical)
k[4].metric("Multi-Hop Paths", len(multi_hop))
k[5].metric("Rapid Paths", rapid_count)

st.markdown("### 🎯 Quick Demo")

if st.button("🎯 Demo Scam Scenario", type="primary"):
    demo_chain = "W001 → W005 → W012 → W019 → W027"
    st.session_state["demo_scenario"] = True

    st.success(f"Suspicious multi-hop chain detected: **{demo_chain}**")

    st.markdown("### 🔎 What Happened?")
    st.write(
        "Funds moved through a sequence of connected wallets across multiple hops, "
        "forming a suspicious multi-hop transaction chain."
    )

    st.markdown("### ⚠️ Why Is It Risky?")
    st.write(
        "The chain shows repeated onward transfers, high pass-through behavior, "
        "rapid movement of funds, and strong wallet-to-wallet linkage."
    )

    st.markdown("### ✅ What Should Happen Next?")
    st.write(
        "Flag the chain for analyst review, inspect linked wallets, and verify "
        "transaction chronology and supporting account context."
    )

    st.info(
        "Synthetic demonstration scenario. Risk score is a prioritization signal, "
        "not a fraud verdict."
    )
if st.button("🤖 Analyze Demo Chain with AI"):
   st.session_state["analyze_demo"] = True

# --------------------------- demo evidence --------------------

if not multi_hop.empty:

    if st.session_state.get("demo_scenario", False):
        demo_matches = multi_hop[
            multi_hop["path"] == "W001 → W005 → W012 → W019 → W027"
        ]

        if not demo_matches.empty:
            strongest = demo_matches.iloc[0]
        else:
            strongest = multi_hop.sort_values(
                "path_score", ascending=False
            ).iloc[0]

    else:
        strongest = multi_hop.sort_values(
            "path_score", ascending=False
        ).iloc[0]

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Strongest Path Score", f"{strongest['path_score']:.1f}/100")
    d2.metric("Hops", int(strongest["hops"]))
    d3.metric("Time Span", f"{strongest['time_span']:g}")
    d4.metric("Retention", f"{strongest['amount_retention_ratio']:.0%}")

    st.info(
        f"**Highest-evidence temporal path:** {strongest['path']} | "
        f"Pattern: {strongest['pattern']} | "
        f"Transactions: {strongest['transaction_ids']}"
    )

# --------------------------- helpers --------------------------
def graph_figure(selected_wallet=None):
    shown_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d["total_amount"] >= min_edge]
    if suspicious_only:
        risky = set(risk.loc[risk["risk_score"] >= 60, "wallet"])
        shown_edges = [(u, v, d) for u, v, d in shown_edges if u in risky or v in risky]
    shown_edges = sorted(shown_edges, key=lambda x: x[2]["total_amount"], reverse=True)[:200]
    dg = nx.DiGraph()
    for u, v, d in shown_edges:
        dg.add_edge(u, v, **d)
    if selected_wallet:
        neighbors = list(graph.predecessors(selected_wallet)) + list(graph.successors(selected_wallet))
        for n in neighbors:
            dg.add_node(n)
        dg.add_node(selected_wallet)
        for n in neighbors:
            if graph.has_edge(n, selected_wallet):
                dg.add_edge(n, selected_wallet, **graph[n][selected_wallet])
            if graph.has_edge(selected_wallet, n):
                dg.add_edge(selected_wallet, n, **graph[selected_wallet][n])
    if dg.number_of_nodes() == 0:
        return go.Figure()
    pos = nx.spring_layout(dg, seed=42, k=0.65, iterations=60)
    ex, ey = [], []
    for u, v in dg.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=0.8), hoverinfo="none", showlegend=False)
    nx_, ny_, ht, colors, sizes, labels = [], [], [], [], [], []
    for node in dg.nodes():
        nx_.append(pos[node][0]); ny_.append(pos[node][1]); labels.append(node if node == selected_wallet else "")
        rr = risk.loc[risk["wallet"] == node]
        if rr.empty:
            colors.append(5); sizes.append(10); ht.append(node)
        else:
            r = rr.iloc[0]
            colors.append(float(r["risk_score"]))
            sizes.append(min(34, 10 + (float(r["total_transactions"]) ** 0.5) * 2.2))
            ht.append(f"Wallet: {node}<br>Risk: {r['risk_score']:.1f}<br>Level: {r['risk_level']}<br>Received: ৳{r['total_received']:,.0f}<br>Sent: ৳{r['total_sent']:,.0f}<br>Anomaly: {r['anomaly_score']:.1f}")
    node_trace = go.Scatter(x=nx_, y=ny_, mode="markers+text", text=labels, textposition="top center", hovertext=ht, hoverinfo="text", marker=dict(size=sizes, color=colors, colorscale="Turbo", cmin=0, cmax=100, showscale=True, colorbar=dict(title="Risk")), showlegend=False)
    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(height=680, margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(showgrid=False,zeroline=False,showticklabels=False), yaxis=dict(showgrid=False,zeroline=False,showticklabels=False), hovermode="closest")
    return fig


def risk_breakdown(row):
    labels = ["ML Anomaly", "Multi-Hop", "Fan In/Out", "Velocity", "Pass-Through", "Cycle", "Connectivity"]
    values = [row["risk_anomaly"], row["risk_multi_hop"], row["risk_fan"], row["risk_velocity"], row["risk_pass_through"], row["risk_cycle"], row["risk_connectivity"]]
    fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", text=[f"{v:.0f}" for v in values], textposition="auto"))
    fig.update_layout(height=360, xaxis_title="Component score (0–100)", margin=dict(l=10,r=10,t=15,b=15))
    return fig


def investigation(wallet):
    r = risk.loc[risk["wallet"] == wallet].iloc[0]
    lines = [
        f"### Investigation Summary — {wallet}",
        f"**Risk score:** {r['risk_score']:.2f}/100  |  **Level:** {r['risk_level']}",
        f"Received **৳{r['total_received']:,.0f}** and sent **৳{r['total_sent']:,.0f}** across {int(r['total_transactions'])} transactions.",
    ]
    indicators = []
    if r["multi_hop_exposure"] > 0: indicators.append("involvement in candidate multi-hop paths")
    if r["cycle_member"]: indicators.append("participation in a detected short cycle")
    if r["fan_score"] >= 60: indicators.append("elevated fan-in/fan-out connectivity")
    if r["rapid_transfer_ratio"] >= 0.5: indicators.append("high share of outgoing transactions shortly following an incoming transaction")
    if r["structuring_score"] >= 60: indicators.append("splitting/structuring indicator")
    if r["anomaly_score"] >= 70: indicators.append("high isolation-based anomaly score")
    lines.append("\n**Indicators:** " + ("; ".join(indicators) + "." if indicators else "No strong indicator crossed the current thresholds."))
    linked = multi_hop[(multi_hop["source"] == wallet) | (multi_hop["destination"] == wallet)] if not multi_hop.empty else multi_hop
    if not linked.empty:
        lines.append("\n**Multi-hop evidence:**")
        for x in linked.head(5).itertuples(index=False):
            lines.append(f"- `{x.path}` — {x.hops} hops, {x.time_span:g} time units, rapid={x.rapid_transfer}, tx={x.transaction_ids}")
    lines.append("\n> Review chronology, linked wallets, device/channel context and account information before any operational action. This prototype does not establish fraud.")
    return "\n".join(lines)

# --------------------------- tabs -----------------------------
t1, t2, t3, t4, t5 = st.tabs(["📊 Risk Dashboard", "🕸️ Transaction Graph", "🚨 Wallet Investigation", "🔗 Multi-Hop Analysis", "🤖 AI Assistant"])

with t1:
    a, b = st.columns([1.3,1])
    with a:
        st.subheader("Top Risky Wallets")
        cols = ["wallet","risk_score","risk_level","anomaly_score","total_received","total_sent","pass_through","unique_receivers","cycle_member"]
        st.dataframe(risk.sort_values("risk_score", ascending=False).head(15)[cols], width="stretch", hide_index=True)
    with b:
        st.subheader("Risk Distribution")
        dist = risk["risk_level"].value_counts().reindex(["LOW","MEDIUM","HIGH","CRITICAL"], fill_value=0)
        fig = go.Figure(go.Bar(x=dist.index,y=dist.values,text=dist.values,textposition="auto"))
        fig.update_layout(height=320, margin=dict(l=5,r=5,t=15,b=10), yaxis_title="Wallet count")
        st.plotly_chart(fig, width="stretch")
    st.subheader("Risk Score Distribution")
    hist = go.Figure(go.Histogram(x=risk["risk_score"], nbinsx=20))
    hist.update_layout(height=260, margin=dict(l=5,r=5,t=15,b=10), xaxis_title="Risk score", yaxis_title="Wallets")
    st.plotly_chart(hist, width="stretch")
    q = st.columns(4)
    q[0].metric("Cycles", len(cycles))
    q[1].metric("Fan-in/out", int((risk["fan_score"] >= 60).sum()))
    q[2].metric("Rapid wallets", int((risk["rapid_transfer_ratio"] >= 0.5).sum()))
    q[3].metric("Structuring candidates", int((risk["structuring_score"] >= 60).sum()))
    st.subheader("Synthetic Transactions")
    st.dataframe(df.head(100), width="stretch", hide_index=True)
    st.download_button("⬇️ Download Risk Report", risk.to_csv(index=False).encode(), "wallet_risk_report.csv", "text/csv")

with t2:
    st.subheader("Dynamic Transaction Graph")
    st.plotly_chart(graph_figure(), width="stretch")
    st.caption("Node color = risk score. Edges are aggregated for visualization; original transactions remain available in the investigation view.")

with t3:
    wallets = risk.sort_values("risk_score", ascending=False)["wallet"].tolist()
    selected = st.selectbox("Select Wallet", wallets)
    row = risk.loc[risk["wallet"] == selected].iloc[0]
    m = st.columns(5)
    m[0].metric("Risk", f"{row['risk_score']:.1f}/100")
    m[1].metric("Level", row["risk_level"])
    m[2].metric("Anomaly", f"{row['anomaly_score']:.1f}")
    m[3].metric("Pass-through", f"{row['pass_through']:.2f}")
    m[4].metric("Tx count", int(row["total_transactions"]))
    st.markdown(investigation(selected))
    st.subheader("Risk Component Breakdown")
    st.plotly_chart(risk_breakdown(row), width="stretch")
    st.subheader("Selected Wallet Neighborhood")
    st.plotly_chart(graph_figure(selected), width="stretch")
    st.subheader("Transaction History")
    history = df[(df["sender"] == selected) | (df["receiver"] == selected)].sort_values("timestamp")
    st.dataframe(history, width="stretch", hide_index=True)
    st.download_button("⬇️ Download Wallet Transactions", history.to_csv(index=False).encode(), f"{selected}_transactions.csv", "text/csv")

with t4:
    st.subheader("Multi-Hop Transaction Detection")
    if multi_hop.empty:
        st.info("No qualifying multi-hop paths detected.")
    else:
        st.dataframe(multi_hop.head(200), width="stretch", hide_index=True)
        st.download_button("⬇️ Download Multi-Hop Report", multi_hop.to_csv(index=False).encode(), "multi_hop_report.csv", "text/csv")
    st.subheader("🔄 Detected Cyclic Money Flows")
    if cycles:
        st.caption("Cycles shown here passed temporal/amount filters; static topology-only cycles are not displayed.")
        for i, cyc in enumerate(cycles[:20], 1):
            st.write(f"**Cycle {i}:** " + " → ".join(cyc + [cyc[0]]))
    else:
        st.success("No evidence-qualified short cycles detected.")

with t5:
    st.subheader("🤖 AI Investigation Assistant")

    st.write(
        "The assistant uses structured evidence only. "
        "Without an API key, it returns a deterministic evidence-based explanation."
    )

    ai_wallet = st.selectbox(
        "Wallet",
        risk.sort_values("risk_score", ascending=False)["wallet"].tolist(),
        key="ai_wallet"
    )

    language = st.selectbox("Language", ["English", "বাংলা"])

    analyze_demo = st.session_state.pop("analyze_demo", False)

    if st.button("Generate Investigation", type="primary") or analyze_demo:

        if analyze_demo:
            ai_wallet = "W012"

        row = risk.loc[risk["wallet"] == ai_wallet].iloc[0]
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

        if api_key:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)

                if analyze_demo:
                    paths = multi_hop[
                        multi_hop["path"]
                        == "W001 → W005 → W012 → W019 → W027"
                    ]
                else:
                    paths = (
                        multi_hop[
                            (multi_hop["source"] == ai_wallet)
                            | (multi_hop["destination"] == ai_wallet)
                        ].head(10)
                        if not multi_hop.empty
                        else multi_hop
                    )

                evidence = {
                    "wallet": ai_wallet,
                    "risk_score": float(row["risk_score"]),
                    "risk_level": str(row["risk_level"]),
                    "anomaly_score": float(row["anomaly_score"]),
                    "total_received": float(row["total_received"]),
                    "total_sent": float(row["total_sent"]),
                    "pass_through": float(row["pass_through"]),
                    "unique_receivers": int(row["unique_receivers"]),
                    "unique_senders": int(row["unique_senders"]),
                    "cycle_member": bool(row["cycle_member"]),
                    "rapid_transfer_ratio": float(row["rapid_transfer_ratio"]),
                    "paths": (
                        paths[
                            [
                                "path",
                                "hops",
                                "transaction_ids",
                                "total_amount",
                                "time_span",
                                "rapid_transfer"
                            ]
                        ].to_dict("records")
                        if not paths.empty
                        else []
                    ),
                }

                instruction = (
                    "বাংলায় লিখুন।"
                    if language == "বাংলা"
                    else "Write in English."
                )

                prompt = (
                    "You are a financial-fraud investigation assistant. "
                    "Use ONLY the evidence below. "
                    "Do not invent facts, transactions, amounts, times, "
                    "probabilities, legal conclusions, or motives. "
                    "Use cautious language such as potential indicator "
                    "and requires review. "
                    + instruction
                    + "\n\n"
                    + str(evidence)
                    + "\n\n"
                    "Return: Investigation Summary; Key Indicators; "
                    "Transaction Flow; Recommended Review Steps."
                )

                response = client.models.generate_content(
                    model=os.getenv(
                        "GEMINI_MODEL",
                        "gemini-3.5-flash-lite"
                    ),
                    contents=prompt
                )

                output = (response.text or "").strip()

                if output:
                    st.markdown(output)
                else:
                    st.markdown(investigation(ai_wallet))

            except Exception as exc:
                st.warning(
                    f"LLM unavailable; deterministic fallback used. {exc}"
                )
                st.markdown(investigation(ai_wallet))

        else:
            text = investigation(ai_wallet)

            if language == "বাংলা":
                text = text.replace(
                    "Investigation Summary",
                    "তদন্ত সারাংশ"
                ).replace(
                    "Indicators",
                    "প্রধান নির্দেশক"
                )

            st.markdown(text)

    st.caption(
        "Optional LLM: set GEMINI_API_KEY. "
        "Otherwise the application remains fully functional."
    )
st.divider()
st.caption("Mule Wallet Detection | Hackathon Prototype | Synthetic MFS Transaction Data")
