1. Chuẩn bị hệ thống
Python 3.10+ (khuyến nghị; kiểm tra: python3 --version).
Git (nếu clone từ remote).
Mở Terminal; mọi lệnh pipeline/GUI nên chạy từ thư mục gốc dự án bird-image-retrieval/ (nơi có cub_pipeline/, bird_search_gui.py), vì đường dẫn birds.db, birds.faiss, CUB_ROOT, OUTPUT_DIR trong config.py là tương đối theo thư mục làm việc.
Apple Silicon: dùng bản Python arm64 (tránh mix Rosetta nếu không cần).

2. Môi trường ảo và cài phụ thuộc
cd /đường/dẫn/tới/bird-image-retrieval
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.base.txt
pip install -r requirements.macos.txt
requirements.macos.txt chỉ kéo thêm requirements.base.txt (FAISS CPU, không gói CUDA). Torch trên Mac thường chạy CPU (đủ cho pipeline hiện tại vì Tier 2 “algorithmic” không bắt buộc GPU theo README).

3. Dataset CUB-200-2011
Tải bộ CUB (ví dụ Kaggle: wenewone/cub2002011 — link trong README.md).
Giải nén sao cho cấu trúc khớp config.py:
CUB_ROOT = "./CUB_200_2011/CUB_200_2011"
Tức trong bird-image-retrieval/ cần có:

bird-image-retrieval/
  CUB_200_2011/
    CUB_200_2011/
      images/
      attributes.txt
      ...
Nếu bạn để dataset chỗ khác, sửa CUB_ROOT trong cub_pipeline/config.py cho đúng.

4. Chạy pipeline offline (lọc ảnh → features 3 tier → SQLite + FAISS)
Trong thư mục bird-image-retrieval, venv đã activate:

python3 -m cub_pipeline.pipeline
Pipeline sẽ (tùy config.py):

Lọc / crop / lưu ảnh vào OUTPUT_DIR (mặc định ./dataset_process/: images/, metadata.csv).
Bước 6: features/, reports/, v.v.
Bước 7: tạo birds.db và birds.faiss trong thư mục hiện tại (working directory).
Cache (đã tích hợp): với REUSE_EXISTING_FEATURES = True và REUSE_EXISTING_DB = True, lần chạy sau sẽ bỏ qua tier/DB nếu file checkpoint đã có và không ép rebuild. Muốn tính lại hết: đặt REUSE_EXISTING_FEATURES = False (và nếu cần) REUSE_EXISTING_DB = False, hoặc dùng các FORCE_REBUILD_*.

Biến môi trường hữu ích (tùy chọn):

# Bỏ qua bước lọc nếu đã có metadata.csv + images/ hợp lệ trong OUTPUT_DIR
export BIRD_REUSE_FILTERED_DATASET=1
# Số worker tier 1/2/3 (mặc định auto trong code)
export CUB_TIER_EXTRACTION_WORKERS=8
# Worker encode tier2 cho extended fit pool (trong features.py)
export CUB_FIT_TIER2_WORKERS=8
# Nhanh hơn / chậm hơn cho algorithmic tier2
export ALGO_TIER2_SPEED_LEVEL=balanced   # hoặc fast | quality
GPU trên Mac: README nhắc BIRD_PIPELINE_USE_GPU; trên Mac thường không có CUDA như Linux — coi như CPU / optional Metal; không bắt buộc để chạy xong pipeline.

5. Chạy GUI tìm kiếm (bird_search_gui.py)
GUI mặc định trỏ ./dataset_process**ed**/images — không khớp OUTPUT_DIR = ./dataset_process. Bạn nên chỉ rõ đường dẫn trùng với pipeline:

cd /đường/dẫn/tới/bird-image-retrieval
source .venv/bin/activate
python3 bird_search_gui.py \
  --sqlite-path birds.db \
  --faiss-path birds.faiss \
  --images-dir ./dataset_process/images \
  --top-k 5 \
  --metric cosine
--sqlite-path / --faiss-path: cùng thư mục bạn đã chạy pipeline (thường là bird-image-retrieval/).
--images-dir: phải là OUTPUT_DIR/images (ví dụ ./dataset_process/images).
Trên Mac, cửa sổ Tkinter cần môi trường desktop (chạy local, không SSH headless không có display).

6. Script phụ (nếu cần)
run_metadata_db.py: build lại SQLite + FAISS từ feature đã có; mặc định --output-dir là ./dataset_processed — nếu bạn dùng ./dataset_process, sửa tham số:
python3 run_metadata_db.py --output-dir ./dataset_process --sqlite-path birds.db --faiss-path birds.faiss
7. Kiểm tra nhanh sau khi chạy
Trong bird-image-retrieval/:

dataset_process/metadata.csv, dataset_process/images/*.jpg
dataset_process/features/ (CSV, npy, pkl, npz…)
birds.db, birds.faiss
8. Lỗi thường gặp trên Mac
Triệu chứng	Gợi ý
Không tìm thấy CUB
Kiểm tra CUB_ROOT và cấp trúc thư mục lồng nhau CUB_200_2011/CUB_200_2011.
GUI không mở / lỗi Tk
Chạy trên máy có GUI; cài Python từ python.org hoặc Homebrew đủ framework.
Sai thư mục ảnh
Đồng bộ OUTPUT_DIR với --images-dir của GUI.
faiss import lỗi
pip install faiss-cpu trong đúng venv.
Pipeline lâu
Bật cache (REUSE_EXISTING_*), giảm EXTENDED_FIT_STATS_MAX_IMAGES, hoặc ALGO_TIER2_SPEED_LEVEL=fast.
Tóm lại: trên Mac bạn **tạo venv → cài requirements.base + requirements.macos → đặt CUB đúng CUB_ROOT → python3 -m cub_pipeline.pipeline từ thư mục dự án → python3 bird_search_gui.py với --images-dir ./dataset_process/images và cùng đường dẫn birds.db / birds.faiss. Nếu bạn muốn thống nhất tên thư mục với README (dataset_processed), có thể đổi OUTPUT_DIR trong config.py và dùng một bộ đường dẫn duy nhất để tránh nhầm.