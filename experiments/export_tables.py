"""Generate LaTeX and Markdown/CSV tables from metrics YAML files."""

import yaml
import pandas as pd
from pathlib import Path

paper_tables_dir = Path("research_paper/tables")
results_tables_dir = Path("results/tables")
paper_tables_dir.mkdir(parents=True, exist_ok=True)
results_tables_dir.mkdir(parents=True, exist_ok=True)

# 1. Dataset class distribution table
classes_data = [
    ("Normal", 56000, 37000, "1.0x"),
    ("Generic", 40000, 18871, "1.4x"),
    ("Exploits", 33393, 11132, "1.7x"),
    ("Fuzzers", 18184, 6062, "3.1x"),
    ("DoS", 12264, 4089, "4.6x"),
    ("Reconnaissance", 10491, 3496, "5.3x"),
    ("Analysis", 2000, 677, "28.0x"),
    ("Backdoor", 1746, 583, "32.1x"),
    ("Shellcode", 1133, 378, "49.4x"),
    ("Worms", 130, 44, "430.8x"),
]

t1_latex = (
    "\\begin{table}[htbp]\n"
    "\\caption{UNSW-NB15 Class Distribution Across Partitions}\n"
    "\\label{tab:class_distribution}\n"
    "\\centering\n"
    "\\begin{tabular}{lrrr}\n"
    "\\hline\n"
    "\\textbf{Class} & \\textbf{Train Count} & \\textbf{Test Count} & \\textbf{Imbalance Ratio} \\\\\n"
    "\\hline\n"
)
for c, tr, te, r in classes_data:
    t1_latex += f"{c} & {tr:,} & {te:,} & {r} \\\\\n"
t1_latex += (
    "\\hline\n"
    "\\textbf{Total} & \\textbf{175,341} & \\textbf{82,332} & - \\\\\n"
    "\\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

with open(paper_tables_dir / "table1_class_distribution.tex", "w") as f:
    f.write(t1_latex)
with open(results_tables_dir / "table1_class_distribution.tex", "w") as f:
    f.write(t1_latex)

# 2. Classifier Comparison Table
with open("results/metrics/classifier_results.yaml", "r") as f:
    clf_res = yaml.safe_load(f)

rf = clf_res["random_forest"]
xgb = clf_res["xgboost"]

t2_latex = (
    "\\begin{table}[htbp]\n"
    "\\caption{Classifier Comparison on 10-Class UNSW-NB15 Test Set}\n"
    "\\label{tab:classifier_comparison}\n"
    "\\centering\n"
    "\\begin{tabular}{lcccc}\n"
    "\\hline\n"
    "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Macro-F1} & \\textbf{Weighted-F1} & \\textbf{Train Time (s)} \\\\\n"
    "\\hline\n"
    f"Random Forest & {rf['accuracy']:.4f} & {rf['macro_f1']:.4f} & {rf['weighted_f1']:.4f} & {rf['training_time_s']:.2f} \\\\\n"
    f"XGBoost & \\textbf{{{xgb['accuracy']:.4f}}} & \\textbf{{{xgb['macro_f1']:.4f}}} & \\textbf{{{xgb['weighted_f1']:.4f}}} & {xgb['training_time_s']:.2f} \\\\\n"
    "\\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

with open(paper_tables_dir / "table2_classifier_comparison.tex", "w") as f:
    f.write(t2_latex)
with open(results_tables_dir / "table2_classifier_comparison.tex", "w") as f:
    f.write(t2_latex)

# 3. Per-Class Results for XGBoost Table
t3_latex = (
    "\\begin{table}[htbp]\n"
    "\\caption{Per-Class Performance for XGBoost Multi-Class Classifier}\n"
    "\\label{tab:per_class_results}\n"
    "\\centering\n"
    "\\begin{tabular}{lrrrr}\n"
    "\\hline\n"
    "\\textbf{Attack Category} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{Test Support} \\\\ \n"
    "\\hline\n"
)
for cat, m in sorted(xgb["per_class_report"].items()):
    if cat in ["accuracy", "macro avg", "weighted avg"]:
        continue
    t3_latex += f"{cat} & {m['precision']:.4f} & {m['recall']:.4f} & {m['f1-score']:.4f} & {int(m['support']):,} \\\\\n"
t3_latex += (
    "\\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

with open(paper_tables_dir / "table3_per_class_results.tex", "w") as f:
    f.write(t3_latex)
with open(results_tables_dir / "table3_per_class_results.tex", "w") as f:
    f.write(t3_latex)

# 4. Fusion Results Table
with open("results/metrics/fusion_results.yaml", "r") as f:
    fus_res = yaml.safe_load(f)

sim = fus_res["simple_fusion"]
tun = fus_res["tuned_fusion"]
op = fus_res["openset_worms"]

t4_latex = (
    "\\begin{table}[htbp]\n"
    "\\caption{Decision Fusion and Open-Set Novel Threat Detection Results}\n"
    "\\label{tab:fusion_results}\n"
    "\\centering\n"
    "\\begin{tabular}{lcccc}\n"
    "\\hline\n"
    "\\textbf{Configuration} & \\textbf{Known Acc.} & \\textbf{Unknown Recall} & \\textbf{FPR} & \\textbf{Precision (Unknown)} \\\\\n"
    "\\hline\n"
    f"Simple Fusion & {sim['known_accuracy']:.4f} & {sim['unknown_recall']:.4f} & {sim['false_positive_rate']:.4f} & {sim['precision_unknown']:.4f} \\\\\n"
    f"Tuned Fusion & \\textbf{{{tun['known_accuracy']:.4f}}} & {tun['unknown_recall']:.4f} & {tun['false_positive_rate']:.4f} & {tun['precision_unknown']:.4f} \\\\\n"
    f"Open-Set Worms Eval & {tun['known_accuracy']:.4f} & \\textbf{{{op['unknown_recall']:.4f}}} & {op['false_positive_rate']:.4f} & {op['precision_unknown']:.4f} \\\\\n"
    "\\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

with open(paper_tables_dir / "table4_fusion_results.tex", "w") as f:
    f.write(t4_latex)
with open(results_tables_dir / "table4_fusion_results.tex", "w") as f:
    f.write(t4_latex)

# 5. Top SHAP features table
with open("results/metrics/shap_feature_ranking.txt", "r") as f:
    lines = [l.strip() for l in f.readlines() if "|" in l and not l.startswith("Rank") and not l.startswith("-")]

t5_latex = (
    "\\begin{table}[htbp]\n"
    "\\caption{Top-10 Discriminative Features Identified by TreeSHAP}\n"
    "\\label{tab:shap_features}\n"
    "\\centering\n"
    "\\begin{tabular}{rlrl}\n"
    "\\hline\n"
    "\\textbf{Rank} & \\textbf{Feature Name} & \\textbf{Mean |SHAP|} & \\textbf{Domain Description} \\\\\n"
    "\\hline\n"
)
descriptions = {
    "sbytes": "Source-to-destination transaction bytes",
    "smean": "Mean packet size transmitted by source",
    "sttl": "Source-to-destination time-to-live",
    "service": "Application layer service (http, dns, etc.)",
    "ct_srv_dst": "Connections to same service and destination",
    "proto": "Transaction protocol (tcp, udp, etc.)",
    "ct_dst_src_ltm": "Connections between dst and src in 100 flows",
    "ct_srv_src": "Connections to same service and source",
    "ct_dst_sport_ltm": "Connections from dst to same source port",
    "ct_src_dport_ltm": "Connections from src to same destination port",
}

for line in lines[:10]:
    parts = [p.strip() for p in line.split("|")]
    rank = parts[0]
    feat = parts[1]
    shap_val = parts[2]
    desc = descriptions.get(feat, "Network flow feature")
    t5_latex += f"{rank} & \\texttt{{{feat}}} & {shap_val} & {desc} \\\\\n"

t5_latex += (
    "\\hline\n"
    "\\end{tabular}\n"
    "\\end{table}\n"
)

with open(paper_tables_dir / "table5_shap_features.tex", "w") as f:
    f.write(t5_latex)
with open(results_tables_dir / "table5_shap_features.tex", "w") as f:
    f.write(t5_latex)

print("All LaTeX tables generated successfully!")
