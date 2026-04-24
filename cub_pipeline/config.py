CUB_ROOT = "./CUB_200_2011/CUB_200_2011"
OUTPUT_DIR = "./dataset_processed"
TARGET_SIZE = (224, 224)
MIN_IMAGES = 500
ALLOW_RELAX_FALLBACK = False

EXTRACT_RECOGNITION_FEATURES = True
CUB_ATTR_CERTAINTY_THRESHOLD = 3
REQUIRE_TORCH_FOR_CNN = True

BUILD_METADATA_DB = True
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
