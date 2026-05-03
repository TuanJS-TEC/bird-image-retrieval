CUB_ROOT = "./CUB_200_2011/CUB_200_2011"
OUTPUT_DIR = "./dataset_process"
# True (khuyen nghi sau khi da co anh loc): dung metadata.csv + images/ trong OUTPUT_DIR,
# KHONG chay lai find_perching / compute_perching_score / process_and_save_dataset.
# De chay lai buoc loc tu dau: dat False hoac export BIRD_REUSE_FILTERED_DATASET=0
REUSE_EXISTING_FILTERED_DATASET = True

# Chi anh huong khi REUSE_EXISTING_FILTERED_DATASET=True nhung metadata.csv + images/ khong hop le:
#   False (mac dinh): dung pipeline, in huong dan — khong tu dong ghi de bo loc (tranh chay lai loc vo tinh).
#   True: fallback nhu cu — chay day du buoc loc.
# Env: BIRD_ALLOW_RERUN_FILTERING=1
ALLOW_RERUN_FILTERING = False
TARGET_SIZE = (224, 224)
MIN_IMAGES = 500
ALLOW_RELAX_FALLBACK = False

EXTRACT_RECOGNITION_FEATURES = True
CUB_ATTR_CERTAINTY_THRESHOLD = 3
REQUIRE_TORCH_FOR_CNN = True

# ── Checkpoint / cache cho features (Nhom 1) ─────────────────────────────────
# True (mac dinh): neu file output cua buoc do da ton tai thi load va bo qua,
# khong tinh lai tu dau. Dat False de ep tinh lai toan bo (huu ich khi thay doi
# tham so anh huong dac trung, vi du TARGET_SIZE, certainty_threshold, v.v.)
REUSE_EXISTING_FEATURES = True

# Kiem soat tung buoc rieng le (chi co hieu luc khi REUSE_EXISTING_FEATURES=True):
#   FORCE_REBUILD_TIER1 = True  -> luon tinh lai Tier 1 du file da co.
#   FORCE_REBUILD_TIER2 = True  -> luon tinh lai Tier 2 embedding (giu nguyen fit .pkl).
#   FORCE_REBUILD_TIER3 = True  -> luon tinh lai Tier 3 handcrafted.
#   FORCE_REBUILD_FUSION = True -> luon tinh lai fusion / recognition_features_all.npz.
FORCE_REBUILD_TIER1 = False
FORCE_REBUILD_TIER2 = False
FORCE_REBUILD_TIER3 = False
FORCE_REBUILD_FUSION = False

BUILD_METADATA_DB = True
# True (mac dinh): chi build lai SQLite + FAISS khi chua co, hoac khi features
# moi duoc tinh lai trong lan chay nay. False: luon build lai.
REUSE_EXISTING_DB = True
SQLITE_DB_PATH = "birds.db"
FAISS_INDEX_PATH = "birds.faiss"
FAISS_USE_COSINE = True

# Fusion weights for 3-tier rerank in bird_search_gui.py
# (tier1_custom, tier2_algorithmic, tier3_handcrafted)
RERANK_FUSION_WEIGHTS = (0.15, 0.70, 0.15)

# Smart retrieval parameters
AQE_ALPHA = 0.60
DIFFUSION_ALPHA = 0.25
# (head, breast, back_wing, tail, leg)
PART_WEIGHTS = (0.35, 0.30, 0.15, 0.10, 0.10)

# Adaptive score-level fusion in GUI rerank
ADAPTIVE_FUSION_ENABLED = True
ADAPTIVE_CONFIDENCE_THRESHOLD = 0.80
ADAPTIVE_PROTOTYPE_MIN_SCORE = 0.70
ADAPTIVE_PROTOTYPE_MARGIN = 0.02

# Tier-wise feature fusion for recognition_features_all.npz
# Apply per-tier z-score normalization before weighted concatenation.
ENABLE_TIER_NORMALIZATION = True
TIER1_WEIGHT = 2.0
TIER2_WEIGHT = 1.0
TIER3_WEIGHT = 1.0
# Balanced Hu-moment emphasis on top of Tier-1 weighting.
ENABLE_HU_BOOST = True
HU_MOMENT_WEIGHT = 1.0

# Sparse green ratio handling in Tier-1 custom attributes.
# If enabled, cattr_color_green_ratio is replaced by binary signal
# (1 if ratio >= GREEN_RATIO_BINARY_THRESHOLD else 0).
ENABLE_GREEN_BINARY = True
GREEN_RATIO_BINARY_THRESHOLD = 0.01

# Vector compaction for retrieval space
# Exclude noisy Tier-1 attributes from distance space (still kept in raw reports).
RETRIEVAL_EXCLUDED_TIER1_ATTRS = (
    "cattr_fg_aspect_ratio",
    "cattr_centroid_x",
    "cattr_centroid_y",
)
# Apply PCA to Tier-2 before fusion (None or <=0 means disabled).
TIER2_PCA_DIM = 448

# ── Feature Fitting Pool ─────────────────────────────────────────────────────
# When True, Tier-2 model fitting (PCA, GMM, KMeans, anatomy priors) uses the
# full set of original CUB images for all 126 filtered species (~7 486 images)
# instead of only the 511 filtered/perching images.
# The 511 filtered images are still the only ones encoded into FAISS + SQLite.
USE_FULL_SPECIES_FOR_FIT = True
# Gioi han so anh goc trong extended pool CHI cho thong ke z-score (Tang1/Tang3 + encode Tier2 fit).
# 0 = dung toan bo (~7k anh, rat cham). >0: lay mau stratified theo class_id (giu can bang loai).
EXTENDED_FIT_STATS_MAX_IMAGES = 2800
