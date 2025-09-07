import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

csv_file_path = r'evaluation_results\model_comparison_20250803_222742.csv'

# 1) Load CSV
df = pd.read_csv(csv_file_path)

# 2) Metrics of interest
metrics = ['mAP@0.5:0.95','mAP@0.5','Precision','Recall']
for m in metrics:
    df[m] = pd.to_numeric(df[m], errors='coerce')

# 3) Pivot into one row per (Model), one column per (Dataset,Metric)
pivot = df.pivot(index='Model', columns='Dataset', values=metrics)
pivot.columns = [f"{metric}_{ds}" for metric, ds in pivot.columns]
pivot = pivot.dropna()

# 4) Build the feature matrix
X = pivot.values
models = pivot.index.tolist()

# 5) Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6) Run PCA (2 components)
pca = PCA(n_components=2)
embedding = pca.fit_transform(X_scaled)

# 7) Define and project the “ideal” vector
ideal_raw    = np.ones_like(X[0])
ideal_scaled = scaler.transform(ideal_raw.reshape(1, -1))
ideal_emb    = pca.transform(ideal_scaled)[0]

# 8) Compute distances to ideal
dists = np.linalg.norm(embedding - ideal_emb, axis=1)

# 9) Assemble & rank
out = pd.DataFrame({
    'Model': models,
    'PC1': embedding[:,0],
    'PC2': embedding[:,1],
    'Distance_to_Ideal': dists
}).sort_values('Distance_to_Ideal')

# 10) Show ranking
print("\nModels ranked by closeness to the ideal (perfect) metrics:\n")
print(out.to_string(index=False, float_format="%.4f"))

# 11) Save to CSV
output_path = os.path.splitext(csv_file_path)[0] + '_PCA_distances.csv'
out.to_csv(output_path, index=False, float_format="%.4f")
print(f"\n✅ Ranking also saved to: {output_path}")