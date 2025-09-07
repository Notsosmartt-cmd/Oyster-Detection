import pandas as pd
import numpy as np
import os

csv_file_path = r'evaluation_results\model_comparison_20250717_001353.csv'

# Load data from CSV
try:
    df = pd.read_csv(csv_file_path)
    print("CSV loaded successfully. Columns found:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: File '{csv_file_path}' not found.")
    exit()
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Required accuracy metrics
required_metrics = ['mAP@0.5:0.95','mAP@0.5', 'Precision', 'Recall']

# Optional speed metrics
speed_metrics = ['Inference (ms/img)', 'Total (ms/img)', 'FPS']

# Check which speed metrics actually exist
available_columns = df.columns.tolist()
present_speed = [m for m in speed_metrics if m in available_columns]

# Convert everything to numeric
for col in required_metrics + present_speed:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter models common to all datasets
datasets = df['Dataset'].unique()
print("Datasets found:", datasets)
models_per_dataset = [set(df[df['Dataset'] == ds]['Model']) for ds in datasets]
valid_models = sorted(set.intersection(*models_per_dataset))
if not valid_models:
    print("Error: No models found that appear in all datasets")
    exit()
print("Models present in all datasets:", valid_models)

# Choose your primary metric
primary_metric = 'mAP@0.5:0.95'
print(f"\nUsing primary metric: {primary_metric}")

# Tie‑breaker order: accuracy metrics minus primary
tie_breakers = [m for m in required_metrics if m != primary_metric]

# Build stats for each model
model_stats = {}
for model in valid_models:
    subset = df[df['Model'] == model]
    stats = {'model': model}

    # Avg primary metric
    stats['avg_primary'] = subset.groupby('Dataset')[primary_metric].first().mean()

    # Avg tie‑breakers
    stats['avg_tie'] = [subset.groupby('Dataset')[m].first().mean() for m in tie_breakers]

    # Avg speed metrics
    stats['avg_inference'] = subset.groupby('Dataset')['Inference (ms/img)'].first().mean() if 'Inference (ms/img)' in present_speed else None
    stats['avg_total']     = subset.groupby('Dataset')['Total (ms/img)'].first().mean()     if 'Total (ms/img)'     in present_speed else None
    stats['avg_fps']       = subset.groupby('Dataset')['FPS'].first().mean()                  if 'FPS'               in present_speed else None

    model_stats[model] = stats

# Pairwise wins on primary metric
wins = {m:0 for m in valid_models}
for i, mi in enumerate(valid_models):
    for mj in valid_models[i+1:]:
        if model_stats[mi]['avg_primary'] > model_stats[mj]['avg_primary']:
            wins[mi] += 1
        else:
            wins[mj] += 1

# Prepare for sorting
results = []
for m in valid_models:
    s = model_stats[m]
    # key: wins, avg_primary, tie‑breakers...
    key = (wins[m], s['avg_primary']) + tuple(s['avg_tie'])
    results.append((m, wins[m], s['avg_primary'], s['avg_inference'], s['avg_total'], s['avg_fps'], key))

# Sort descending
results.sort(key=lambda x: x[6], reverse=True)

# Print ranking with speed columns
print("\nModel Ranking:")
header = f"{'Rank':<5} {'Model':<20} {'Wins':<5} {'Avg_'+primary_metric:<10}"
if 'Inference (ms/img)' in present_speed:
    header += "  Inference(ms) "
if 'Total (ms/img)' in present_speed:
    header += "  Total(ms) "
if 'FPS' in present_speed:
    header += "  FPS "
print(header)

for idx, (m, w, ap, inf, tot, fps, _) in enumerate(results, 1):
    line = f"{idx:<5} {m:<25} {w:<5} {ap:>10.3f}"
    if inf is not None:
        line += f"    {inf:>8.2f}"
    if tot is not None:
        line += f"    {tot:>8.2f}"
    if fps is not None:
        line += f"    {fps:>6.1f}"
    print(line)

# Best model
print(f"\nBest model is: {results[0][0]}")
