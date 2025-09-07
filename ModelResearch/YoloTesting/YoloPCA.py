import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

csv_file_path = r'evaluation_results\model_comparison_20250803_222742.csv'

# 1) Load CSV
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: File '{csv_file_path}' not found."); exit()
except Exception as e:
    print(f"Error loading CSV file: {e}"); exit()

# 2) Metrics of interest
required_metrics = ['mAP@0.5:0.95','mAP@0.5','Precision','Recall']
for col in required_metrics:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3) Restrict to models present in all datasets
datasets = df['Dataset'].unique()
models_per_dataset = [set(df[df['Dataset']==d]['Model']) for d in datasets]
valid_models = sorted(set.intersection(*models_per_dataset))

# 4) Compute per‑model averages
rows = []
for m in valid_models:
    sub = df[df['Model']==m]
    avg_metrics = {
        metric: sub.groupby('Dataset')[metric].first().mean()
        for metric in required_metrics
    }
    avg_metrics['Model'] = m
    rows.append(avg_metrics)
agg_df = pd.DataFrame(rows).set_index('Model')

# 5) Run PCA → PC1
pca = PCA(n_components=1)
agg_df['PC1_score'] = pca.fit_transform(agg_df[required_metrics])

# 6) Rank by PC1 (descending)
agg_df = agg_df.sort_values('PC1_score', ascending=False)

# 7) Show results
print("Model ranking by first principal component (PC1):\n")
print(agg_df[['PC1_score'] + required_metrics].to_string(float_format="%.4f"))

# 8) Save to CSV
output_path = os.path.splitext(csv_file_path)[0] + '_PC1_ranking.csv'
agg_df[['PC1_score'] + required_metrics].to_csv(output_path, float_format="%.4f")
print(f"\n✅ Ranking saved to: {output_path}")