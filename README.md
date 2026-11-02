# UNETI18 — YÊU CẦU 5: Ứng dụng AI mini

## 1. Đặt vào đúng thư mục (quan trọng)

Giải nén **cả thư mục `UNETI18_APP`** ngay bên trong thư mục `olympic vòng 2` đã dùng để chạy notebook.

```text
olympic vòng 2/
├── UNETI18_Vong2_ChotNop_DA_SUA_LOI_LUU_MODEL.ipynb
├── UNETI05_TEST_3000.csv
├── UNETI2026_TRAIN_CHUNG_1200000.csv
├── KET_QUA_UNETI18_CHOT_NOP/
│   ├── 06_thong_tin_lan_chay.json
│   ├── UNETI18_predictions_VONG2_CHOT.csv
│   └── mo_hinh/
│       ├── lgb_base_....txt
│       ├── lgb_base_....pkl
│       ├── cat_compact_....cbm
│       ├── cat_compact_....pkl
│       └── ...
└── UNETI18_APP/
    ├── app.py
    ├── model_runtime.py
    ├── preprocessing.py
    ├── prepare_app.py
    ├── requirements.txt
    ├── CAI_DAT_LAN_DAU.bat
    └── CHAY_UNG_DUNG.bat
```

## 2. Khởi chạy trên Windows

**Lần đầu:** Nháy đúp `UNETI18_APP/CAI_DAT_LAN_DAU.bat`.

Script cài Streamlit (nếu chưa có), **đóng gói mô hình đã huấn luyện**, kiểm tra dự đoán 3.000 chuyến Test có khớp CSV cuối, rồi mở ứng dụng trên trình duyệt. Không chạy lại LightGBM/CatBoost.

**Những lần sau:** Nháy đúp `UNETI18_APP/CHAY_UNG_DUNG.bat`.

Nếu nháy đúp không chạy, mở terminal VS Code tại thư mục `olympic vòng 2` và nhập tuần tự:

```powershell
py -m pip install -r UNETI18_APP/requirements.txt
py UNETI18_APP/prepare_app.py
py -m streamlit run UNETI18_APP/app.py
```

Streamlit sẽ cho biết URL nội bộ, thường là `http://localhost:8501`.

## 3. Đã làm đúng Yêu cầu 5 như thế nào?

- Giao diện Streamlit nhập/sửa tất cả 12 trường đầu vào gốc; không dùng `trip_id` hay dữ liệu sau chuyến đi.
- Chọn 2 kịch bản mẫu khác nhau, hoặc tự nhập; bấm nút **Dự đoán** để chạy suy luận.
- Sau mỗi lượt dự đoán hiện **biểu đồ so sánh LightGBM, CatBoost và kết quả kết hợp**; từ lượt thứ hai hiện thêm biểu đồ so sánh các chuyến đã thử trong phiên, có nút xóa lịch sử. Đây là dự đoán thật được tính khi bấm nút, không phải các con số tĩnh.
- Nạp **chính mô hình được chọn trong `06_thong_tin_lan_chay.json`**, dùng đúng đặc trưng, median và từ điển category đã lưu trong `.pkl`.
- Tổng hợp dự đoán trên **log1p**, chuyển về giây bằng `expm1`, hiển thị giây và phút.
- Nếu mô hình thiếu/sai, **báo lỗi**, không sinh dự đoán giả.
- `prepare_app.py` kiểm tra kết quả tính lại trên 3.000 chuyến Test có khớp CSV nộp; nếu không khớp thì không được dùng app demo.
- Khi đóng gói đúng, thư mục `UNETI18_APP` có `bundle.json` và `models/*.txt|*.cbm|*.pkl` — có thể copy nguyên thư mục sang máy khác đã cài dependencies, **không cần CSV Train/Test**.

## 4. Kịch bản quay video 2–3 phút

1. (0:00–0:20) Mở terminal, chạy `py -m streamlit run UNETI18_APP/app.py`, chuyển sang trình duyệt để thấy giao diện.
2. (0:20–0:55) Bấm **Ví dụ 1: Chuyến ngắn**, bấm **Dự đoán**, đọc kết quả bằng giây và phút.
3. (0:55–1:35) Bấm **Ví dụ 2: Chuyến dài**, bấm **Dự đoán**, đọc kết quả khác với ví dụ 1.
4. (1:35–2:10) So sánh **hai biểu đồ** (dự đoán từng mô hình và các chuyến đã thử), sau đó thay giờ hoặc khoảng cách để dự đoán thêm.
5. (2:10–2:40) Mở phần **Xem chi tiết** và sidebar để chứng minh đang tải ba mô hình đã huấn luyện và trọng số kết hợp.
6. (2:40–3:00) Hiển thị file nguồn `app.py`, `model_runtime.py`, `preprocessing.py` nếu cần.

*Theo đề: video màn hình 2–3 phút; không bắt buộc thuyết minh, quay mặt hay deploy công khai.*

## 5. Những lưu ý quan trọng

- Không xóa `KET_QUA_UNETI18_CHOT_NOP/mo_hinh` trước khi chạy lần đầu.
- Mô hình dùng trong app chính là mô hình từ Notebook chốt nộp (mô hình kết hợp được chọn), **không dùng** mô hình chuyên biệt ≤2 km có kết quả Tune kém hơn.
- Không thể đóng gói nếu notebook chỉ chạy cache V2 mà không có model tương ứng. `prepare_app.py` sẽ báo lỗi rõ ràng.
- Các tệp `.pkl` chỉ đọc từ thư mục do chính nhóm tạo ra; **không lấy `.pkl` không đáng tin cậy** từ Internet.
- Dữ liệu có tọa độ đã được làm tròn khoảng 0.01 độ, còn `straight_line_km_approx` làm tròn theo mốc 0.5 km; hai số có thể khác nhau. Khi sửa tọa độ trong giao diện, người nhập cần cập nhật cả khoảng cách.
- Thư viện Python được cài trong môi trường hiện tại (mở cùng Python 3.13.7 như VS Code). Nếu `py` dùng nhầm phiên bản, thay `py` bằng đường dẫn tới Python đang dùng.

## 6. Những tệp cần nộp cho phần ứng dụng

Cần giữ nguyên `app.py`, `model_runtime.py`, `preprocessing.py`, `requirements.txt`, `bundle.json`, toàn bộ thư mục `models/`, cùng video demo 2–3 phút. Không chỉ nộp `app.py` vì thiếu mô hình đã huấn luyện sẽ không dự đoán được.

## 7. Tạo ZIP ứng dụng có đầy đủ mô hình để gửi BTC

Sau khi đã chạy `prepare_app.py` thành công và thử hai tình huống:

- Nháy đúp `UNETI18_APP/DONG_GOI_DE_NOP.bat`, hoặc chạy `py UNETI18_APP/create_submission_zip.py`.
- Kết quả tạo cạnh thư mục `UNETI18_APP`: **`UNETI18_YeuCau5_UngDung_AI_Mini_NOP.zip`**.
- Tệp ZIP có **mã nguồn, requirements, bundle.json và các mô hình đã huấn luyện**; không chứa Train/Test.
- Nộp kèm video quay màn hình 2–3 phút. CSV dự đoán của Yêu cầu 4 và báo cáo PDF là các thành phần nộp riêng theo thông báo của BTC.
