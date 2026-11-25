# =====================================================================
#  PIPELINE END-TO-END DỰ ĐOÁN TƯƠNG TÁC THUỐC - THUỐC (DDI)
#  Quy trình hoàn chỉnh: 
#    1. Đọc dữ liệu thô & 10 ma trận tương đồng
#    2. Lọc nhiễu đặc trưng dựa trên Entropy (Entropy-based Refinement)
#    3. Hợp nhất mạng tương đồng đa nguồn (Similarity Network Fusion - SNF)
#    4. Lấy mẫu âm cân bằng (Balanced Negative Sampling 1:1)
#    5. Phân chia 4 kịch bản kiểm thử chống rò rỉ dữ liệu (4-Case Splitting)
#    6. Trích xuất đặc trưng cặp thuốc (Pair Feature Representation)
#    7. Huấn luyện các mô hình học máy & Soft Voting Ensemble
#    8. Đánh giá đa chỉ số (ROC-AUC, PR-AUC, F1-score) & Xuất kết quả
# =====================================================================

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Bỏ qua các cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# Thiết lập bảng mã UTF-8 cho console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Cố định SEED ngẫu nhiên để tái lập kết quả 100%
SEED = 42
np.random.seed(SEED)

print("=" * 80)
print(" PIPELINE END-TO-END: LỌC NHIỄU ENTROPY -> HỢP NHẤT SNF -> HUẤN LUYỆN MÔ HÌNH")
print("=" * 80)

# =====================================================================
# GIAI ĐOẠN 1: NẠP DỮ LIỆU ĐẶC TRƯNG & NHÃN TƯƠNG TÁC DDI
# =====================================================================
print("\n>>> [GIAI ĐOẠN 1] ĐANG NẠP DỮ LIỆU NGUỒN...")

# Tự động xác định thư mục gốc dự án (dù chạy từ thư mục gốc hay từ src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR

def find_data_path(*subpaths):
    candidates = [
        os.path.join(PROJECT_ROOT, *subpaths),
        os.path.join(os.getcwd(), *subpaths),
        os.path.join(*subpaths)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

# Xác định đường dẫn file nhãn DDI
DDI_MATRIX_PATH = find_data_path('data', 'raw', 'drug_drug_matrix.csv')
if not os.path.exists(DDI_MATRIX_PATH):
    DDI_MATRIX_PATH = find_data_path('data', 'drug_drug_matrix..csv')
if not os.path.exists(DDI_MATRIX_PATH):
    DDI_MATRIX_PATH = find_data_path('data', 'drug_drug_matrix.csv')

# Xác định đường dẫn thư mục ma trận tương đồng
MATRICES_DIR = find_data_path('data', 'processed', 'output_matrices')
if not os.path.exists(MATRICES_DIR):
    MATRICES_DIR = find_data_path('data', 'output_matrices')

# Kiểm tra sự tồn tại của file nhãn
if not os.path.exists(DDI_MATRIX_PATH):
    raise FileNotFoundError(f"Không tìm thấy file nhãn DDI tại: {DDI_MATRIX_PATH}")

# Đọc ma trận kề tương tác DDI (nhãn)
ddi_matrix = pd.read_csv(DDI_MATRIX_PATH, header=None).values
n_drugs = ddi_matrix.shape[0]
print(f" [+] Số lượng thuốc: {n_drugs}")
print(f" [+] Kích thước ma trận nhãn DDI: {ddi_matrix.shape}")

# Danh sách 10 kênh ma trận tương đồng
matrix_files = {
    'ATC': 'atc_code_jaccard_sim.csv',
    'Chem': 'chem_tanimoto_sim.csv',
    'Enzyme': 'enzyme_jaccard_sim.csv',
    'Indication': 'indication_sim.csv',
    'MeSH': 'mesh_code_jaccard_sim.csv',
    'Offside': 'offside_effect_jaccard_sim.csv',
    'Pathway': 'pathway_jaccard_sim.csv',
    'Side_Effect': 'side_effect_jaccard_sim.csv',
    'Target': 'target_jaccard_sim.csv',
    'Transporter': 'transporter_jaccard_sim.csv',
}

raw_matrices = {}
for name, fname in matrix_files.items():
    fpath = os.path.join(MATRICES_DIR, fname)
    if os.path.exists(fpath):
        mat = pd.read_csv(fpath, header=None).values
        # Đảm bảo đường chéo bằng 1 (tự tương đồng của một thuốc) và giá trị trong [0, 1]
        np.fill_diagonal(mat, 1.0)
        mat = np.clip(mat, 0.0, 1.0)
        # Đối xứng hóa ma trận
        mat = (mat + mat.T) / 2.0
        raw_matrices[name] = mat
        print(f" [+] Đã nạp ma trận {name:12s}: shape {mat.shape}, min={mat.min():.2f}, max={mat.max():.2f}")
    else:
        print(f" [!] CẢNH BÁO: Không tìm thấy file {fpath}")

print(f" -> Tổng số kênh đặc trưng tương đồng đã nạp: {len(raw_matrices)}/10")


# =====================================================================
# GIAI ĐOẠN 2: LỌC NHIỄU DỰA TRÊN SHANNON ENTROPY (ENTROPY REFINEMENT)
# =====================================================================
print("\n>>> [GIAI ĐOẠN 2] LỌC NHIỄU ĐẶC TRƯNG BẰNG ENTROPY (GAMMA >= 0.85)...")

def compute_entropy_ratio(matrix, eps=1e-12):
    """
    Tính tỷ lệ Entropy gamma = E / E_max cho một ma trận tương đồng.
    Ma trận có thông tin phong phú sẽ có entropy cao; ma trận quá thưa/nhiễu sẽ có entropy thấp.
    """
    n = matrix.shape[0]
    # Lấy các phần tử nửa trên tam giác (loại đường chéo)
    triu_vals = matrix[np.triu_indices(n, k=1)]
    # Chuẩn hóa về phân bố xác suất p
    total_val = np.sum(triu_vals)
    if total_val <= eps:
        return 0.0
    p = triu_vals / total_val
    # Lọc bỏ giá trị bằng 0 để tránh log(0)
    p = p[p > eps]
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(len(triu_vals))
    gamma = entropy / max_entropy if max_entropy > 0 else 0.0
    return gamma

GAMMA_THRESHOLD = 0.85
filtered_matrices = {}
entropy_stats = []

for name, mat in raw_matrices.items():
    gamma = compute_entropy_ratio(mat)
    status = "GIỮ LẠI" if gamma >= GAMMA_THRESHOLD else "LOẠI BỎ (Nhiễu/Thưa)"
    entropy_stats.append({
        'Kênh đặc trưng': name,
        'Hệ số Entropy (gamma)': round(gamma, 4),
        'Ngưỡng (0.85)': GAMMA_THRESHOLD,
        'Trạng thái': status
    })
    # Trong thực tế, các kênh đạt chuẩn sẽ được đưa vào SNF
    if gamma >= GAMMA_THRESHOLD:
        filtered_matrices[name] = mat

df_entropy = pd.DataFrame(entropy_stats)
print(df_entropy.to_string(index=False))

# Nếu số lượng ma trận lọc được < 2, giữ lại toàn bộ để đảm bảo thuật toán SNF hoạt động
if len(filtered_matrices) < 2:
    print(" [!] Lưu ý: Số kênh vượt ngưỡng 0.85 ít hơn 2, tiếp tục sử dụng tất cả các kênh hợp lệ.")
    filtered_matrices = raw_matrices

print(f" -> Số lượng kênh ma trận đưa vào bước hợp nhất SNF: {len(filtered_matrices)}")


# =====================================================================
# GIAI ĐOẠN 3: HỢP NHẤT MẠNG TƯƠNG ĐỒNG ĐA NGUỒN (SNF ALGORITHM)
# =====================================================================
print("\n>>> [GIAI ĐOẠN 3] HỢP NHẤT MẠNG TƯƠNG ĐỒNG BẰNG THUẬT TOÁN SNF (K=20, t=20)...")

def compute_local_affinity_matrix(W, K=20):
    """
    Tính ma trận láng giềng cục bộ S cho mỗi node dựa trên K láng giềng gần nhất.
    S_ij = W_ij / sum_{k in N_i} W_ik nếu j in N_i, ngược lại bằng 0.
    """
    n = W.shape[0]
    S = np.zeros_like(W)
    for i in range(n):
        # Lấy K chỉ số láng giềng có độ tương đồng lớn nhất (loại bỏ chính nó)
        row = W[i].copy()
        row[i] = -1.0 # Không lấy chính nó
        top_k_indices = np.argsort(row)[::-1][:K]
        sum_top_k = np.sum(W[i, top_k_indices])
        if sum_top_k > 0:
            S[i, top_k_indices] = W[i, top_k_indices] / sum_top_k
    return S

def compute_full_transition_matrix(W):
    """
    Tính ma trận xác suất chuyển trạng thái toàn cục P.
    P_ij = W_ij / (2 * sum_{k != i} W_ik) nếu i != j; P_ii = 1/2.
    """
    n = W.shape[0]
    P = np.zeros_like(W)
    for i in range(n):
        row_sum = np.sum(W[i]) - W[i, i]
        if row_sum > 0:
            P[i, :] = W[i, :] / (2.0 * row_sum)
            P[i, i] = 0.5
        else:
            P[i, i] = 1.0
    return P

def similarity_network_fusion(similarity_dict, K=20, t_iter=20):
    """
    Thuật toán Similarity Network Fusion (Wang et al., Nature Methods 2014)
    Hợp nhất m ma trận tương đồng qua t vòng lặp trao đổi thông tin cục bộ và toàn cục.
    """
    names = list(similarity_dict.keys())
    m = len(names)
    matrices = [similarity_dict[k].copy() for k in names]
    
    # Bước 1: Khởi tạo P và S cho từng nguồn
    P_list = [compute_full_transition_matrix(W) for W in matrices]
    S_list = [compute_local_affinity_matrix(W, K=K) for W in matrices]
    
    # Bước 2: Vòng lặp cập nhật trạng thái chéo (Iterative Fusion)
    print(f" [+] Bắt đầu {t_iter} vòng lặp khuếch tán thông tin mạng...")
    for it in range(t_iter):
        P_next_list = []
        for v in range(m):
            # Tính trung bình trạng thái của các nguồn còn lại (ngoại trừ nguồn v)
            other_sum = np.zeros_like(P_list[0])
            for j in range(m):
                if j != v:
                    other_sum += P_list[j]
            P_other_avg = other_sum / (m - 1)
            
            # Cập nhật: P_v^(t+1) = S_v * P_other_avg * (S_v)^T
            P_v_next = S_list[v] @ P_other_avg @ (S_list[v].T)
            
            # Chuẩn hóa đối xứng lại ma trận sau mỗi bước
            P_v_next = (P_v_next + P_v_next.T) / 2.0
            np.fill_diagonal(P_v_next, 0.5)
            P_next_list.append(P_v_next)
            
        P_list = P_next_list
        if (it + 1) % 5 == 0 or it == t_iter - 1:
            print(f"     -> Hoàn thành vòng lặp {it+1:2d}/{t_iter}")
            
    # Bước 3: Tổng hợp trạng thái cuối cùng thành ma trận hợp nhất duy nhất
    P_fused = np.mean(P_list, axis=0)
    # Chuẩn hóa về [0, 1] và gán đường chéo bằng 1.0
    P_fused = (P_fused - P_fused.min()) / (P_fused.max() - P_fused.min() + 1e-12)
    np.fill_diagonal(P_fused, 1.0)
    P_fused = (P_fused + P_fused.T) / 2.0
    
    return P_fused

t_start = time.time()
P_fused = similarity_network_fusion(filtered_matrices, K=20, t_iter=20)
print(f" [+] SNF hoàn tất sau {time.time() - t_start:.2f} giây.")
print(f" [+] Kích thước ma trận hợp nhất SNF: {P_fused.shape}")
print(f" [+] Giá trị trung bình: {P_fused.mean():.4f}, min: {P_fused.min():.4f}, max: {P_fused.max():.4f}")


# =====================================================================
# GIAI ĐOẠN 4: THIẾT LẬP 4 KỊCH BẢN CHỐNG RÒ RỈ DỮ LIỆU & LẤY MẪU ÂM 1:1
# =====================================================================
print("\n>>> [GIAI ĐOẠN 4] THIẾT LẬP 4 KỊCH BẢN PHÂN CHIA CHỐNG RÒ RỈ (4 SCENARIOS)...")

class ExperimentDataSplitter:
    """
    Bộ phân chia dữ liệu kiểm soát rò rỉ cấu trúc:
      - Case 1 (Warm-start): Chia ngẫu nhiên theo cặp (Edge-level)
      - Case 2 (Strict Cold-start): Cả 2 thuốc mới hoàn toàn (New x New, Node-disjoint)
      - Case 3 (Semi Cold-start): 1 thuốc cũ x 1 thuốc mới (Old x New)
      - Case 4 (Mixed Cold-start): Gộp cả Case 2 và Case 3
    """
    def __init__(self, interaction_matrix, n_drugs, test_size=0.2, seed=42):
        self.inter = interaction_matrix
        self.n_drugs = n_drugs
        self.test_size = test_size
        self.seed = seed
        
    def get_split(self, case_id):
        np.random.seed(self.seed)
        
        if case_id == 1:
            # Case 1: Warm-start (Random pair split)
            idx1, idx2 = np.triu_indices(self.n_drugs, k=1)
            y_all = self.inter[idx1, idx2]
            pos_idx = np.where(y_all == 1)[0]
            neg_idx = np.where(y_all == 0)[0]
            
            # Lấy mẫu âm cân bằng 1:1
            neg_sample = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
            all_idx = np.concatenate([pos_idx, neg_sample])
            np.random.shuffle(all_idx)
            
            pairs = np.column_stack([idx1[all_idx], idx2[all_idx]])
            labels = y_all[all_idx]
            
            n_total = len(labels)
            n_test = int(n_total * self.test_size)
            perm = np.random.permutation(n_total)
            
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]
            return pairs[train_idx], pairs[test_idx], labels[train_idx], labels[test_idx]

        # Case 2, 3, 4: Chia theo nút thuốc rời rạc (Drug-disjoint node split)
        all_drugs = np.arange(self.n_drugs)
        np.random.shuffle(all_drugs)
        n_test_drugs = int(self.n_drugs * self.test_size)
        test_drugs = all_drugs[:n_test_drugs]
        train_drugs = all_drugs[n_test_drugs:]
        
        # Tập huấn luyện (Seen x Seen: cặp thuốc nằm hoàn toàn trong tập Train)
        tr1, tr2 = np.triu_indices(len(train_drugs), k=1)
        r_tr1, r_tr2 = train_drugs[tr1], train_drugs[tr2]
        y_tr = self.inter[r_tr1, r_tr2]
        pos_tr = np.where(y_tr == 1)[0]
        neg_tr = np.where(y_tr == 0)[0]
        neg_tr_s = np.random.choice(neg_tr, size=len(pos_tr), replace=False)
        bal_tr = np.concatenate([pos_tr, neg_tr_s])
        np.random.shuffle(bal_tr)
        pairs_train = np.column_stack([r_tr1[bal_tr], r_tr2[bal_tr]])
        y_train = y_tr[bal_tr]
        
        if case_id == 2:
            # Case 2 (Strict Cold): New x New (chỉ cặp giữa các thuốc Test)
            te1, te2 = np.triu_indices(len(test_drugs), k=1)
            r_te1, r_te2 = test_drugs[te1], test_drugs[te2]
            y_te = self.inter[r_te1, r_te2]
            pos_te = np.where(y_te == 1)[0]
            neg_te = np.where(y_te == 0)[0]
            neg_te_s = np.random.choice(neg_te, size=len(pos_te), replace=False)
            bal_te = np.concatenate([pos_te, neg_te_s])
            np.random.shuffle(bal_te)
            pairs_test = np.column_stack([r_te1[bal_te], r_te2[bal_te]])
            y_test = y_te[bal_te]
            
        elif case_id == 3:
            # Case 3 (Semi Cold): Old x New (kết hợp chéo 1 thuốc cũ x 1 thuốc mới)
            mesh_seen, mesh_unseen = np.meshgrid(train_drugs, test_drugs, indexing='ij')
            s_flat, u_flat = mesh_seen.flatten(), mesh_unseen.flatten()
            y_cross = self.inter[s_flat, u_flat]
            pos_c = np.where(y_cross == 1)[0]
            neg_c = np.where(y_cross == 0)[0]
            size_c = min(len(pos_c), len(neg_c))
            pos_c_s = np.random.choice(pos_c, size=size_c, replace=False)
            neg_c_s = np.random.choice(neg_c, size=size_c, replace=False)
            bal_c = np.concatenate([pos_c_s, neg_c_s])
            np.random.shuffle(bal_c)
            pairs_test = np.column_stack([s_flat[bal_c], u_flat[bal_c]])
            y_test = y_cross[bal_c]
            
        elif case_id == 4:
            # Case 4 (Mixed): Gộp cả Case 2 và Case 3
            # New x New
            te1, te2 = np.triu_indices(len(test_drugs), k=1)
            r_te1, r_te2 = test_drugs[te1], test_drugs[te2]
            y_te = self.inter[r_te1, r_te2]
            pos_te = np.where(y_te == 1)[0]
            neg_te = np.where(y_te == 0)[0]
            neg_te_s = np.random.choice(neg_te, size=len(pos_te), replace=False)
            bal_te = np.concatenate([pos_te, neg_te_s])
            p_te = np.column_stack([r_te1[bal_te], r_te2[bal_te]])
            y_te_bal = y_te[bal_te]
            
            # Old x New
            m_s, m_u = np.meshgrid(train_drugs, test_drugs, indexing='ij')
            s_f, u_f = m_s.flatten(), m_u.flatten()
            y_cr = self.inter[s_f, u_f]
            pos_cr = np.where(y_cr == 1)[0]
            neg_cr = np.where(y_cr == 0)[0]
            size_cr = min(len(pos_cr), len(neg_cr))
            pos_cr_s = np.random.choice(pos_cr, size=size_cr, replace=False)
            neg_cr_s = np.random.choice(neg_cr, size=size_cr, replace=False)
            bal_cr = np.concatenate([pos_cr_s, neg_cr_s])
            p_cr = np.column_stack([s_f[bal_cr], u_f[bal_cr]])
            y_cr_bal = y_cr[bal_cr]
            
            pairs_test = np.concatenate([p_te, p_cr], axis=0)
            y_test = np.concatenate([y_te_bal, y_cr_bal], axis=0)
            perm = np.random.permutation(len(y_test))
            pairs_test, y_test = pairs_test[perm], y_test[perm]
            
        return pairs_train, pairs_test, y_train, y_test

splitter = ExperimentDataSplitter(ddi_matrix, n_drugs, test_size=0.2, seed=SEED)


# =====================================================================
# GIAI ĐOẠN 5: TRÍCH XUẤT VECTOR ĐẶC TRƯNG CẶP THUỐC (PAIR REPRESENTATION)
# =====================================================================
print("\n>>> [GIAI ĐOẠN 5] TRÍCH XUẤT VECTOR ĐẶC TRƯNG CHO MỖI CẶP THUỐC (u, v)...")

def extract_pair_features(pairs, fused_matrix, individual_matrices_dict):
    """
    Xây dựng vector đặc trưng cho cặp thuốc (u, v):
      1. Độ tương đồng hợp nhất từ SNF: P_fused[u, v]
      2. Tích Hadamard vector profile SNF: P_fused[u, :] * P_fused[v, :] (thống kê mean, std, max, min)
      3. Độ tương đồng riêng biệt từ từng ma trận thành phần sau lọc nhiễu
      4. Khoảng cách Cosine giữa biểu diễn mạng của hai thuốc
    """
    u_idx = pairs[:, 0]
    v_idx = pairs[:, 1]
    n_samples = len(pairs)
    
    feature_list = []
    
    # 1. Giá trị tương đồng trực tiếp từ mạng SNF
    snf_direct = fused_matrix[u_idx, v_idx].reshape(-1, 1)
    feature_list.append(snf_direct)
    
    # 2. Giá trị tương đồng trực tiếp từ từng kênh đặc trưng
    for name, mat in individual_matrices_dict.items():
        sim_val = mat[u_idx, v_idx].reshape(-1, 1)
        feature_list.append(sim_val)
        
    # 3. Phép đo tương quan giữa 2 vector biểu diễn mạng: Cosine Similarity
    u_vecs = fused_matrix[u_idx, :]
    v_vecs = fused_matrix[v_idx, :]
    
    dot_prod = np.sum(u_vecs * v_vecs, axis=1)
    norm_u = np.linalg.norm(u_vecs, axis=1) + 1e-12
    norm_v = np.linalg.norm(v_vecs, axis=1) + 1e-12
    cosine_sim = (dot_prod / (norm_u * norm_v)).reshape(-1, 1)
    feature_list.append(cosine_sim)
    
    # 4. Thống kê tích Hadamard giữa 2 hàng SNF
    hadamard = u_vecs * v_vecs
    had_mean = np.mean(hadamard, axis=1, keepdims=True)
    had_std = np.std(hadamard, axis=1, keepdims=True)
    had_max = np.max(hadamard, axis=1, keepdims=True)
    feature_list.extend([had_mean, had_std, had_max])
    
    X = np.hstack(feature_list)
    return X


# =====================================================================
# GIAI ĐOẠN 6 & 7: HUẤN LUYỆN CÁC MÔ HÌNH HỌC MÁY & ĐÁNH GIÁ ĐA CHỈ SỐ
# =====================================================================
print("\n>>> [GIAI ĐOẠN 6 & 7] KHỞI TẠO MÔ HÌNH, HUẤN LUYỆN & ĐÁNH GIÁ TRÊN 4 KỊCH BẢN...")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score

# Kiểm tra thư viện XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print(" [!] Thư viện xgboost chưa cài đặt, hệ thống sẽ sử dụng GradientBoostingClassifier của Scikit-learn.")
    from sklearn.ensemble import GradientBoostingClassifier

def evaluate_metrics(y_true, y_prob, threshold=0.5):
    """Tính toán 3 chỉ số then chốt: ROC-AUC, PR-AUC (AUPR) và F1-score"""
    # 1. ROC-AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # 2. PR-AUC (AUPR)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    # 3. F1-score tại ngưỡng 0.5
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return roc_auc, pr_auc, f1

# Danh sách 4 kịch bản đánh giá
scenarios = [
    (1, "Case 1: Warm-start (Random Split)"),
    (2, "Case 2: Strict Cold-start (New x New)"),
    (3, "Case 3: Semi Cold-start (Old x New)"),
    (4, "Case 4: Mixed Cold-start (NewxNew + OldxNew)")
]

all_results = []

for case_id, case_name in scenarios:
    print("-" * 80)
    print(f"[*] ĐANG THỰC HIỆN: {case_name}")
    print("-" * 80)
    
    # 1. Chia tập Train và Test
    P_tr, P_te, y_tr, y_te = splitter.get_split(case_id)
    print(f" [+] Số lượng mẫu Train: {len(y_tr):,} (Dương: {np.sum(y_tr==1):,}, Âm: {np.sum(y_tr==0):,})")
    print(f" [+] Số lượng mẫu Test:  {len(y_te):,} (Dương: {np.sum(y_te==1):,}, Âm: {np.sum(y_te==0):,})")
    
    # 2. Trích xuất vector đặc trưng
    X_tr = extract_pair_features(P_tr, P_fused, filtered_matrices)
    X_te = extract_pair_features(P_te, P_fused, filtered_matrices)
    print(f" [+] Số chiều vector đặc trưng: {X_tr.shape[1]}")
    
    # 3. Khởi tạo 4 mô hình cơ sở với tham số chuẩn của đề tài
    rf_model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    lr_model = LogisticRegression(max_iter=1000, random_state=SEED)
    knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    
    if HAS_XGB:
        xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED, eval_metric='logloss', n_jobs=-1)
    else:
        xgb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED)
        
    models = {
        'Random Forest': rf_model,
        'Logistic Regression': lr_model,
        'KNN': knn_model,
        'XGBoost': xgb_model
    }
    
    # Huấn luyện và dự đoán xác suất từng mô hình
    prob_dict = {}
    for m_name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]
        prob_dict[m_name] = probs
        roc_val, pr_val, f1_val = evaluate_metrics(y_te, probs)
        train_time = time.time() - t0
        print(f"     -> {m_name:20s}: AUC = {roc_val:.4f} | AUPR = {pr_val:.4f} | F1 = {f1_val:.4f} ({train_time:.1f}s)")
        all_results.append({
            'Kịch bản': f"Case {case_id}",
            'Mô hình': m_name,
            'ROC-AUC': round(roc_val, 4),
            'PR-AUC (AUPR)': round(pr_val, 4),
            'F1-score': round(f1_val, 4)
        })
        
    # 4. Mô hình tổ hợp đề xuất: SOFT VOTING ENSEMBLE
    # Trung bình cộng xác suất dự đoán của 4 mô hình cơ sở
    ensemble_probs = np.mean([prob_dict['Random Forest'], 
                              prob_dict['Logistic Regression'], 
                              prob_dict['KNN'], 
                              prob_dict['XGBoost']], axis=0)
    
    ens_roc, ens_pr, ens_f1 = evaluate_metrics(y_te, ensemble_probs)
    print(f"     -> {'Soft Voting Ensemble':20s}: AUC = {ens_roc:.4f} | AUPR = {ens_pr:.4f} | F1 = {ens_f1:.4f} [MÔ HÌNH ĐỀ XUẤT]")
    
    all_results.append({
        'Kịch bản': f"Case {case_id}",
        'Mô hình': 'Soft Voting Ensemble (Đề xuất)',
        'ROC-AUC': round(ens_roc, 4),
        'PR-AUC (AUPR)': round(ens_pr, 4),
        'F1-score': round(ens_f1, 4)
    })
    print()

# =====================================================================
# BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM
# =====================================================================
print("=" * 80)
print(" BẢNG TỔNG HỢP HIỆU NĂNG CÁC MÔ HÌNH TRÊN 4 KỊCH BẢN THỰC NGHIỆM")
print("=" * 80)

df_final_results = pd.DataFrame(all_results)
print(df_final_results.to_string(index=False))

# Lưu kết quả ra file CSV
results_dir = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(results_dir, exist_ok=True)
output_csv_path = os.path.join(results_dir, 'model_evaluation_results.csv')
df_final_results.to_csv(output_csv_path, index=False, encoding='utf-8')
print(f"\n[+] Đã lưu toàn bộ kết quả thực nghiệm ra file: {output_csv_path}")

print("\n" + "=" * 80)
print(" CHÚC MỪNG: PIPELINE END-TO-END ĐÃ CHẠY HOÀN TẤT THÀNH CÔNG 100%!")
print("=" * 80)

