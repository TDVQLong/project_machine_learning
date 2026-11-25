import os
import sys
import json

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập thẩm mỹ cho đồ thị chuẩn khoa học
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.titlesize'] = 14

os.makedirs('images', exist_ok=True)

# 1. ĐỌC DỮ LIỆU
print("1. Đang đọc dữ liệu...")
raw_df = pd.read_csv('data/drug_10_raw_features.csv')
ddi_matrix = pd.read_csv('data/drug_drug_matrix..csv', header=None).values

n_drugs = len(raw_df)
print(f"Số lượng thuốc: {n_drugs}")
print(f"Kích thước ma trận DDI: {ddi_matrix.shape}")

# Danh sách 10 ma trận similarity trong data/output_matrices
matrix_files = {
    'ATC': 'data/output_matrices/atc_code_jaccard_sim.csv',
    'Chem': 'data/output_matrices/chem_tanimoto_sim.csv',
    'Enzyme': 'data/output_matrices/enzyme_jaccard_sim.csv',
    'Indication': 'data/output_matrices/indication_sim.csv',
    'MeSH': 'data/output_matrices/mesh_code_jaccard_sim.csv',
    'Offside': 'data/output_matrices/offside_effect_jaccard_sim.csv',
    'Pathway': 'data/output_matrices/pathway_jaccard_sim.csv',
    'Side_Effect': 'data/output_matrices/side_effect_jaccard_sim.csv',
    'Target': 'data/output_matrices/target_jaccard_sim.csv',
    'Transporter': 'data/output_matrices/transporter_jaccard_sim.csv',
}

sim_dict = {}
for name, path in matrix_files.items():
    if os.path.exists(path):
        sim_dict[name] = pd.read_csv(path, header=None).values
        print(f"Đã nạp {name}: shape {sim_dict[name].shape}")

# 2. THỐNG KÊ THUỘC TÍNH VÀ GIÁ TRỊ THIẾU
print("\n2. Thống kê thuộc tính và dữ liệu thiếu...")
missing_stats = {}
for col in raw_df.columns:
    null_cnt = int(raw_df[col].isnull().sum())
    missing_stats[col] = {
        'missing_count': null_cnt,
        'missing_pct': round(null_cnt / n_drugs * 100, 2)
    }

# 3. THỐNG KÊ MẠNG TƯƠNG TÁC THUỐC (DDI)
print("\n3. Thống kê ma trận tương tác DDI...")
triu_indices = np.triu_indices(n_drugs, k=1)
total_possible_pairs = len(triu_indices[0])  # 149878
y_all = ddi_matrix[triu_indices]
pos_count = int(np.sum(y_all == 1))  # 48584
unobs_count = int(np.sum(y_all == 0)) # 101294
pos_density = round(pos_count / total_possible_pairs * 100, 2)

# Bậc tương tác của từng thuốc
degrees = np.sum(ddi_matrix, axis=1)
degree_stats = {
    'mean': round(float(np.mean(degrees)), 2),
    'std': round(float(np.std(degrees)), 2),
    'median': float(np.median(degrees)),
    'min': int(np.min(degrees)),
    'max': int(np.max(degrees)),
    'q25': float(np.percentile(degrees, 25)),
    'q75': float(np.percentile(degrees, 75))
}

# 4. PHÂN TÍCH TƯƠNG QUAN GIỮA 10 NGUỒN ĐẶC TRƯNG TƯƠNG ĐỒNG
print("\n4. Tính ma trận tương quan giữa 10 nguồn tương đồng...")
sim_pair_values = {}
for name, mat in sim_dict.items():
    sim_pair_values[name] = mat[triu_indices]

sim_df = pd.DataFrame(sim_pair_values)
corr_matrix = sim_df.corr(method='pearson')

# Tính độ tương đồng trung bình đối với cặp tương tác (y=1) vs không tương tác (y=0)
diff_dict = {}
for name in sim_dict.keys():
    vals = sim_pair_values[name]
    mean_pos = float(np.mean(vals[y_all == 1]))
    mean_neg = float(np.mean(vals[y_all == 0]))
    diff_dict[name] = {
        'mean_pos': round(mean_pos, 4),
        'mean_neg': round(mean_neg, 4),
        'diff': round(mean_pos - mean_neg, 4),
        'ratio': round(mean_pos / (mean_neg + 1e-8), 2)
    }

# 5. LƯU BÁO CÁO THỐNG KÊ JSON
report = {
    'dataset': {
        'num_drugs': n_drugs,
        'total_possible_pairs': total_possible_pairs,
        'positive_ddi_pairs': pos_count,
        'unobserved_ddi_pairs': unobs_count,
        'interaction_density_pct': pos_density
    },
    'degree_statistics': degree_stats,
    'missing_values': missing_stats,
    'feature_vs_label_diff': diff_dict,
    'correlation_matrix': corr_matrix.round(3).to_dict()
}

with open('eda_statistics_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print("Đã lưu eda_statistics_report.json")

# =====================================================================
# VẼ CÁC ĐỒ THỊ CHUẨN MỰC XUẤT BẢN KHOA HỌC
# =====================================================================

# HÌNH 1: Phân bố nhãn trước và sau khi lấy mẫu âm (Negative Sampling)
print("\nVẽ Hình 1: Phân bố nhãn...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

# Trước lấy mẫu
colors_pre = ['#2b5c8f', '#d95f02']
bars1 = ax1.bar(['Tương tác DDI (1)', 'Chưa ghi nhận (0)'], [pos_count, unobs_count], 
                color=colors_pre, width=0.55, edgecolor='black', linewidth=1.1)
ax1.set_title('(a) Toàn bộ không gian cặp thuốc (Trước xử lý)', pad=10, fontweight='bold')
ax1.set_ylabel('Số lượng cặp thuốc', fontweight='bold')
ax1.set_ylim(0, 120000)
for bar in bars1:
    yval = bar.get_height()
    pct = yval / total_possible_pairs * 100
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 2500, f'{yval:,}\n({pct:.1f}%)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Sau lấy mẫu cân bằng (1:1)
bars2 = ax2.bar(['Tương tác DDI (1)', 'Lấy mẫu âm (0)'], [pos_count, pos_count], 
                color=['#2b5c8f', '#2ca02c'], width=0.55, edgecolor='black', linewidth=1.1)
ax2.set_title('(b) Bộ dữ liệu cân bằng (Sau Negative Sampling 1:1)', pad=10, fontweight='bold')
ax2.set_ylabel('Số lượng cặp thuốc', fontweight='bold')
ax2.set_ylim(0, 70000)
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1500, f'{yval:,}\n(50.0%)', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('images/fig_1_label_distribution.png', bbox_inches='tight')
plt.close()

# HÌNH 2: Phân bố bậc tương tác của 548 thuốc (Degree Distribution)
print("Vẽ Hình 2: Phân bố bậc tương tác của thuốc...")
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
sns.histplot(degrees, bins=35, kde=True, color='#1f77b4', edgecolor='black', alpha=0.65, ax=ax)
ax.axvline(degree_stats['mean'], color='red', linestyle='--', linewidth=1.8, label=f"Trung bình: {degree_stats['mean']:.1f}")
ax.axvline(degree_stats['median'], color='green', linestyle=':', linewidth=2.0, label=f"Trung vị: {degree_stats['median']:.1f}")
ax.set_title('Phân bố số lượng tương tác (Bậc tương tác) của 548 thuốc', pad=12, fontweight='bold')
ax.set_xlabel('Số lượng tương tác trên một thuốc (Degree)', fontweight='bold')
ax.set_ylabel('Số lượng thuốc (Tần số)', fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig('images/fig_2_drug_degree_dist.png', bbox_inches='tight')
plt.close()

# HÌNH 3: Biểu đồ Boxplot phân bố độ tương đồng của 10 nguồn đặc trưng
print("Vẽ Hình 3: Phân bố độ tương đồng 10 kênh...")
fig, ax = plt.subplots(figsize=(12, 5.2), dpi=300)
# Lấy mẫu ngẫu nhiên 10,000 cặp để vẽ boxplot nhanh và chuẩn xác
sample_idx = np.random.choice(total_possible_pairs, size=15000, replace=False)
sample_df = sim_df.iloc[sample_idx]

sns.boxplot(data=sample_df, ax=ax, palette='Set3', showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"black", "markersize":"5"})
ax.set_title('Phân bố giá trị độ tương đồng giữa các cặp thuốc trên 10 kênh đặc trưng sinh y học', pad=12, fontweight='bold')
ax.set_ylabel('Giá trị độ tương đồng [0, 1]', fontweight='bold')
ax.set_xlabel('Kênh đặc trưng sinh y tế', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontweight='bold')
plt.tight_layout()
plt.savefig('images/fig_3_features_similarity_box.png', bbox_inches='tight')
plt.close()

# HÌNH 4: Ma trận tương quan giữa 10 nguồn đặc trưng
print("Vẽ Hình 4: Ma trận tương quan 10 nguồn đặc trưng...")
fig, ax = plt.subplots(figsize=(8.8, 7.2), dpi=300)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1.0, 
            cbar_kws={'label': 'Hệ số tương quan Pearson'}, square=True, linewidths=0.7, ax=ax)
ax.set_title('Ma trận tương quan Pearson giữa 10 nguồn đặc trưng tương đồng của thuốc', pad=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/fig_4_correlation_heatmap.png', bbox_inches='tight')
plt.close()

# HÌNH 5: So sánh độ tương đồng trung bình của cặp DDI=1 vs DDI=0
print("Vẽ Hình 5: Độ tương đồng DDI=1 vs DDI=0...")
channels = list(diff_dict.keys())
mean_pos_list = [diff_dict[ch]['mean_pos'] for ch in channels]
mean_neg_list = [diff_dict[ch]['mean_neg'] for ch in channels]

x = np.arange(len(channels))
width = 0.38

fig, ax = plt.subplots(figsize=(11, 5.0), dpi=300)
rects1 = ax.bar(x - width/2, mean_pos_list, width, label='Có tương tác (DDI = 1)', color='#2b5c8f', edgecolor='black')
rects2 = ax.bar(x + width/2, mean_neg_list, width, label='Không tương tác (DDI = 0)', color='#d95f02', edgecolor='black')

ax.set_title('Độ tương đồng trung bình ở từng kênh: Cặp có tương tác (1) so với Cặp không tương tác (0)', pad=12, fontweight='bold')
ax.set_xlabel('Kênh đặc trưng sinh y tế', fontweight='bold')
ax.set_ylabel('Độ tương đồng trung bình', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=30, ha='right', fontweight='bold')
ax.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig('images/fig_5_feature_vs_label.png', bbox_inches='tight')
plt.close()

# HÌNH 6: Tỷ lệ thiếu dữ liệu ở các thuộc tính ban đầu
print("Vẽ Hình 6: Tỷ lệ thiếu dữ liệu...")
raw_cols = [c for c in raw_df.columns if c not in ['Internal_ID', 'DrugBank_ID']]
missing_pcts = [missing_stats[c]['missing_pct'] for c in raw_cols]

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
colors_missing = ['#d95f02' if p > 20 else '#2b5c8f' for p in missing_pcts]
bars = ax.bar(raw_cols, missing_pcts, color=colors_missing, width=0.55, edgecolor='black', linewidth=1.1)
ax.set_title('Tỷ lệ khuyết thiếu dữ liệu (%) ở các trường thuộc tính thô của 548 thuốc', pad=12, fontweight='bold')
ax.set_ylabel('Tỷ lệ thiếu (%)', fontweight='bold')
ax.set_xlabel('Trường thuộc tính y sinh', fontweight='bold')
ax.set_ylim(0, 80)
ax.set_xticklabels(raw_cols, rotation=30, ha='right', fontweight='bold')

for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f'{yval:.1f}%', 
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2.0, 1.0, '0%', 
                ha='center', va='bottom', fontsize=9.5, color='gray')

plt.tight_layout()
plt.savefig('images/fig_6_missing_values_bar.png', bbox_inches='tight')
plt.close()

# HÌNH 7: Sơ đồ kiểm soát rò rỉ dữ liệu (Warm-start vs Cold-start)
print("Vẽ Hình 7: Minh họa chống rò rỉ dữ liệu...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

# Vẽ ma trận chia cạnh (Edge split - Warm start)
mat_edge = np.zeros((10, 10))
np.random.seed(42)
mask_train = np.random.rand(10, 10) < 0.8
mat_edge[mask_train] = 1 # Train
mat_edge[~mask_train] = 2 # Test
ax1.imshow(mat_edge, cmap='coolwarm', alpha=0.8)
ax1.set_title('(a) Random Edge Split (Nguy cơ rò rỉ dữ liệu)\nThuốc xuất hiện ở cả Train và Test', fontweight='bold', fontsize=11)
ax1.set_xlabel('Chỉ số thuốc A')
ax1.set_ylabel('Chỉ số thuốc B')
ax1.grid(color='white', linestyle='-', linewidth=1.5)

# Vẽ ma trận chia node rời rạc (Drug-disjoint node split - Cold start)
mat_node = np.zeros((10, 10))
mat_node[:8, :8] = 1 # Train x Train
mat_node[8:, 8:] = 2 # Test x Test (Case 2: Cả 2 thuốc mới)
mat_node[:8, 8:] = 3 # Train x Test (Case 3: 1 cũ, 1 mới)
mat_node[8:, :8] = 3
ax2.imshow(mat_node, cmap='viridis', alpha=0.85)
ax2.set_title('(b) Drug-Disjoint Split (Kiểm soát hoàn toàn)\nKhông rò rỉ thực thể thuốc giữa Train & Test', fontweight='bold', fontsize=11)
ax2.set_xlabel('Chỉ số thuốc A')
ax2.set_ylabel('Chỉ số thuốc B')
ax2.grid(color='white', linestyle='-', linewidth=1.5)

plt.tight_layout()
plt.savefig('images/fig_7_data_leakage_scenarios.png', bbox_inches='tight')
plt.close()

print("\nHoàn tất tạo tất cả đồ thị và báo cáo thống kê EDA!")
