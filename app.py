"""PaySim fraud detection dashboard (Step 4).

Reads only results/ and data/processed/ — never the raw CSV.
Run:  .venv\\Scripts\\streamlit run dashboard\\app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"


@st.cache_data
def load_csv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_parquet(name):
    p = RES / name
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data
def load_json(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else None


@st.cache_data
def load_checks():
    return load_json("checks.json") if False else json.loads((RES / "checks.json").read_text(encoding="utf8"))


@st.cache_data
def load_meta():
    return json.loads((PROC / "meta.json").read_text(encoding="utf8"))


st.set_page_config(page_title="PaySim Fraud Detection", layout="wide")
st.title("PaySim Fraud Detection — Foundations of Data Science")

page = st.sidebar.radio(
    "Section",
    ["Overview", "Model comparison", "Alert budget", "Why is it fraud",
     "Overfitting", "Bias audit", "Ablation", "Extras"],
)

# ---------------------------------------------------------------- overview ----
if page == "Overview":
    checks = load_checks()
    meta = load_meta()
    m = load_csv("metrics.csv")
    sel = load_json("selection.json")
    st.subheader("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full rows", f"{checks['full']['rows']:,}")
    c2.metric("Frauds", f"{checks['full']['frauds']:,} ({checks['full']['fraud_share']:.3%})")
    c3.metric("Modelling subset (TRANSFER/CASH_OUT)", f"{meta['splits']['train']['rows']+meta['splits']['val']['rows']+meta['splits']['test']['rows']:,}")
    c4.metric("Subset fraud rate", f"{checks['subset']['fraud_share']:.3%}")
    st.write("Fraud occurs only in TRANSFER and CASH_OUT. Other types are legitimate by rule.")
    st.write(f"Steps: {checks['full']['step_min']}–{checks['full']['step_max']} ({checks['full']['distinct_steps']} distinct). "
             f"Split (by row count): train steps 1–{meta['b_train']}, val {meta['b_train']+1}–{meta['b_val']}, "
             f"test {meta['b_val']+1}–{checks['full']['step_max']}.")

    st.subheader("Split sizes")
    sp = pd.DataFrame(meta["splits"]).T
    st.dataframe(sp, use_container_width=True)

    st.subheader("Selected winners")
    if sel:
        for setting, info in sel["settings"].items():
            st.write(f"- **{setting}**: `{info['winner']}` — {info.get('winner_reason')}")

    if m is not None:
        st.subheader("Test AUC-PR by model (setting = all)")
        sub = m[(m.setting == "all") & (m.split == "test")].sort_values("ap", ascending=False)
        fig = px.bar(sub, x="model", y="ap", error_y=sub["ap_hi"] - sub["ap"],
                     title="Test AUC-PR (95% bootstrap CI)")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- comparison --
elif page == "Model comparison":
    m = load_csv("metrics.csv")
    pr = load_parquet("pr_curves.parquet")
    st.subheader("Metrics table")
    split = st.selectbox("Split", ["val", "test"])
    setting = st.selectbox("Setting", ["all", "no_balance", "no_balance_no_hour"])
    if m is not None:
        sub = m[(m.split == split) & (m.setting == setting)].sort_values("ap", ascending=False)
        cols = ["model", "family", "ap", "ap_lo", "ap_hi", "roc", "prec", "rec", "f1", "r90", "mcc", "alert", "amt_rec"]
        st.dataframe(sub[cols].round(4), use_container_width=True)

    if pr is not None:
        st.subheader("Precision–recall curves")
        prs = pr[(pr.split == split) & (pr.setting == setting)]
        models = st.multiselect("Models", sorted(prs.model.unique()), default=sorted(prs.model.unique())[:6])
        fig = go.Figure()
        for md in models:
            d = prs[prs.model == md]
            fig.add_trace(go.Scatter(x=d.recall, y=d.precision, mode="lines", name=md))
        fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", title=f"PR curves ({setting}, {split})")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- budget -----
elif page == "Alert budget":
    b = load_csv("budget.csv")
    bd = load_csv("budget_days.csv")
    bc = load_parquet("budget_curves.parquet")
    st.subheader("Per-day volume (test)")
    if bd is not None:
        td = bd[bd.split == "test"]
        st.dataframe(td, use_container_width=True)
        st.write("No window is skipped at any K (all 15 have > 1,000 rows). "
                 "The 6-step tail (70 rows, all fraud) is dropped as a partial window.")

    setting = st.selectbox("Setting", ["all", "no_balance", "no_balance_no_hour", "-"])
    split = st.selectbox("Split", ["test", "val"], key="b_split")
    K = st.selectbox("K", [100, 250, 500, 1000], index=1)
    if b is not None:
        sub = b[(b.split == split) & (b.K == K)]
        if setting != "-":
            sub = sub[(sub.setting == setting) | (sub.model.str.startswith("baseline"))]
        sub = sub.sort_values("money_caught", ascending=False)
        st.dataframe(sub[["setting", "model", "ranker", "n_days", "precision", "recall",
                          "money_caught", "money_total", "net_saving_c0.001", "net_saving_c0.05"]].round(4),
                     use_container_width=True)

    if bc is not None:
        st.subheader("Budget curve (precision vs K)")
        bcs = bc[(bc.split == split)]
        if setting != "-":
            bcs = bcs[(bcs.setting == setting) | (bcs.model.str.startswith("baseline"))]
        models = st.multiselect("Models", sorted(bcs.model.unique()), default=["rf", "xgb", "lr"], key="bc_models")
        ranker = st.selectbox("Ranker", ["score", "expected_loss"], key="bc_ranker")
        fig = go.Figure()
        for md in models:
            d = bcs[(bcs.model == md) & (bcs.ranker == ranker)]
            fig.add_trace(go.Scatter(x=d.K, y=d.precision, mode="lines", name=md))
        # baselines
        for md in ["baseline_flag", "baseline_rule"]:
            d = bcs[(bcs.model == md) & (bcs.ranker == md)]
            if len(d):
                fig.add_trace(go.Scatter(x=d.K, y=d.precision, mode="lines", name=md, line=dict(dash="dash")))
        fig.update_layout(xaxis_title="K (alerts per day)", yaxis_title="Precision@K", title=f"Budget curve ({setting}, {split})")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- why --------
elif page == "Why is it fraud":
    ex = load_json("explain.json")
    sh = load_parquet("shap_sample.parquet")
    r = load_csv("reasons.csv")
    st.subheader("SHAP feature ranking (top 10k test rows)")
    if ex and ex.get("shap", {}).get("mean_abs"):
        ma = ex["shap"]["mean_abs"]
        sdf = pd.DataFrame({"feature": list(ma.keys()), "mean_abs_shap": list(ma.values())}).sort_values("mean_abs_shap", ascending=True)
        fig = px.bar(sdf, x="mean_abs_shap", y="feature", orientation="h", title="Mean |SHAP|")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Stability (Spearman between halves):", ex["shap"]["stability"])
    st.subheader("Reason codes on the alerts")
    if r is not None:
        st.write(f"Alerts: {len(r)}, frauds: {int(r.isFraud.sum())}, precision: {r.isFraud.mean():.3%}")
        rc = r["reasons"].fillna("").str.split(";").explode()
        rc = rc[rc != ""]
        fig = px.bar(rc.value_counts().reset_index(), x="reasons", y="count", title="Reason-code coverage")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(r.head(20), use_container_width=True)
    st.subheader("Surrogate tree (depth 3)")
    if ex:
        st.text(ex["surrogate"]["tree_text"])
        st.write("Fidelity:", ex["surrogate"]["fidelity"])

# ---------------------------------------------------------------- overfit ----
elif page == "Overfitting":
    ov = load_csv("overfit.csv")
    rl = load_csv("rolling.csv")
    lc = load_csv("learning_curves.csv")
    st.subheader("Overfit ratios (prevalence-matched)")
    if ov is not None:
        g = ov[ov.pair == "train->val"]
        st.dataframe(g[["setting", "model", "value_a", "value_b", "overfit_ratio", "or_lo", "or_hi", "passes_guard"]].round(4), use_container_width=True)
    st.subheader("Rolling-window validation")
    if rl is not None:
        st.dataframe(rl.round(4), use_container_width=True)
    else:
        st.info("Rolling windows still running or not yet generated.")
    st.subheader("Learning curves")
    if lc is not None:
        setting = st.selectbox("Setting", sorted(lc.setting.unique()), key="lc_setting")
        fig = px.line(lc[lc.setting == setting], x="fraction", y="val_auc_pr", color="model", markers=True,
                      title=f"Learning curves ({setting})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Learning curves still running or not yet generated.")

# ---------------------------------------------------------------- bias -------
elif page == "Bias audit":
    bias = load_csv("bias.csv")
    if bias is None:
        st.info("bias.csv not yet generated.")
    else:
        setting = st.selectbox("Setting", sorted(bias.setting.unique()))
        model = st.selectbox("Model", sorted(bias[bias.setting == setting].model.unique()))
        split = st.selectbox("Split", ["val", "test"], key="bias_split")
        sub = bias[(bias.setting == setting) & (bias.model == model) & (bias.split == split)]
        group_type = st.selectbox("Group", ["type", "amount_band", "hour_block"])
        d = sub[sub.group_type == group_type]
        st.dataframe(d[["group", "n_frauds", "n_legit", "recall", "far", "recall_ratio", "far_ratio", "pass_recall", "pass_far"]].round(3), use_container_width=True)
        fig = px.bar(d, x="group", y="recall_ratio", title=f"Recall ratio by {group_type} (pass >= 0.8)")
        fig.add_hline(y=0.8, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- ablation ---
elif page == "Ablation":
    m = load_csv("metrics.csv")
    if m is not None:
        sub = m[(m.split == "test") & (m.model.isin(["lr", "rf", "xgb", "iforest"]))].pivot(index="model", columns="setting", values="ap")
        st.dataframe(sub.round(4), use_container_width=True)
        fig = px.bar(m[(m.split == "test") & (m.model.isin(["lr", "rf", "xgb"]))], x="model", y="ap", color="setting", barmode="group",
                     title="Test AUC-PR across ablation settings")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- extras -----
elif page == "Extras":
    m = load_csv("metrics.csv")
    if m is not None:
        st.subheader("Autoencoder / SMOTE / amount-weighted variants (test AUC-PR)")
        sub = m[(m.split == "test") & (m.setting == "all") & (m.model.isin(["ae", "lr_smote", "rf_smote", "xgb_smote", "lr_amt", "rf_amt", "xgb_amt", "lr", "rf", "xgb"]))]
        st.dataframe(sub[["model", "ap", "ap_lo", "ap_hi", "roc", "rec", "prec"]].round(4), use_container_width=True)
    st.subheader("Cost sensitivity (K = 250, test)")
    b = load_csv("budget.csv")
    if b is not None:
        sub = b[(b.split == "test") & (b.K == 250) & (b.setting == "all") & (b.model.isin(["rf", "xgb"]))]
        st.dataframe(sub[["model", "ranker", "net_saving_c0.001", "net_saving_c0.01", "net_saving_c0.05", "money_caught"]].round(0), use_container_width=True)
    st.subheader("Expected-loss ranking (K = 100, test, rf)")
    if b is not None:
        sub = b[(b.split == "test") & (b.K == 100) & (b.setting == "all") & (b.model == "rf")]
        st.dataframe(sub[["ranker", "precision", "recall", "money_caught"]].round(4), use_container_width=True)
