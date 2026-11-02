"""UNETI18 — Giao diện ngang gọn, sửa lỗi lớp trắng và chồng chữ.

Chỉ thay giao diện app.py; KHÔNG chỉnh mô hình, pipeline tiền xử lý,
trọng số phối hợp hoặc phép tính dự đoán trong model_runtime.py.
Giữ nguyên bundle.json, models/, model_runtime.py, preprocessing.py,
assets/nyc_taxi_hero.webp và .streamlit/config.toml.
"""
from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from model_runtime import TaxiEnsemble, haversine_km

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "bundle.json"
HERO_IMAGE = ROOT / "assets" / "nyc_taxi_hero.webp"

st.set_page_config(
    page_title="UNETI18 | AI Taxi Forecast",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Đang tải các mô hình AI đã huấn luyện...")
def load_ensemble():
    return TaxiEnsemble(BUNDLE)


# Giữ nguyên 12 trường và hai ví dụ của app gốc.
PRESETS = {
    "short": {
        "vendor_id": 1,
        "pickup_month": 3,
        "pickup_dayofweek": 5,
        "pickup_hour": 18,
        "pickup_minute_bucket": 45,
        "passenger_count": 1,
        "pickup_lon_approx": -73.99,
        "pickup_lat_approx": 40.72,
        "dropoff_lon_approx": -74.00,
        "dropoff_lat_approx": 40.73,
        "straight_line_km_approx": 1.5,
        "store_and_fwd_flag": "N",
    },
    "long": {
        "vendor_id": 2,
        "pickup_month": 5,
        "pickup_dayofweek": 1,
        "pickup_hour": 8,
        "pickup_minute_bucket": 15,
        "passenger_count": 2,
        "pickup_lon_approx": -73.78,
        "pickup_lat_approx": 40.64,
        "dropoff_lon_approx": -73.98,
        "dropoff_lat_approx": 40.76,
        "straight_line_km_approx": 21.5,
        "store_and_fwd_flag": "N",
    },
}
INPUT_KEYS = {key: f"ride_{key}" for key in PRESETS["short"]}


def choose_preset(name: str) -> None:
    for field, value in PRESETS[name].items():
        st.session_state[INPUT_KEYS[field]] = value
    st.session_state["selected_preset"] = name
    st.session_state.pop("prediction", None)


if "selected_preset" not in st.session_state:
    choose_preset("short")

# Ảnh chỉ trang trí. Không có ảnh thì giao diện vẫn chạy bằng gradient.
image_style = ""
if HERO_IMAGE.is_file():
    encoded = base64.b64encode(HERO_IMAGE.read_bytes()).decode("ascii")
    image_style = f"background-image:url('data:image/webp;base64,{encoded}');"

# CSS giới hạn theo class riêng của ứng dụng, KHÔNG tạo lớp fixed/absolute
# trên thanh điều hướng và KHÔNG sửa chi tiết HTML/summary của st.expander.
CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root{--ink:#102d49;--navy:#102d49;--teal:#06a9ad;--line:#dce9f2;--muted:#688195;}
html,body,[data-testid="stAppViewContainer"]{font-family:'Be Vietnam Pro',system-ui,sans-serif}
[data-testid="stAppViewContainer"]{background:#f0f7fb}
[data-testid="stHeader"]{background:rgba(240,247,251,.94)}
/* Mở rộng nội dung tối đa thay vì khung 1.380px gây khoảng trắng hai bên. */
/* Dành đủ chỗ cho header TRẮNG MẶC ĐỊNH của Streamlit ở trên cùng.
   Header cao khoảng 3.5rem; đẩy navbar xuống hẳn phía dưới, không đụng vào
   lớp trắng và không dùng position fixed / sticky / margin âm. */
[data-testid="stMainBlockContainer"], .main .block-container{
    max-width:100% !important;width:100% !important;
    padding:4.75rem clamp(1rem,2vw,2.2rem) 1.4rem !important;
}
/* Click các mục Trang chủ / Dự đoán không bị header che điểm neo. */
#trang-chu, #du-doan, #gioi-thieu, #tai-lieu{
    scroll-margin-top:5rem;
}
footer{visibility:hidden}
/* Sidebar cố định tự nhiên theo Streamlit; chỉ đổi màu, không tạo overlay. */
[data-testid="stSidebar"]{background:linear-gradient(170deg,#0d2941,#143654) !important}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{padding-top:1rem}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{color:#e8f5fa}
[data-testid="stSidebar"] p{color:#e8f5fa !important}
[data-testid="stSidebar"] hr{border-color:#ffffff29}
/* Thanh menu nằm TRONG luồng tài liệu: không fixed, không sticky, không đè. */
.uneti-nav{display:flex;align-items:center;justify-content:space-between;flex-wrap:nowrap;
    min-height:53px;gap:10px;background:white;border:1px solid #e1eaf1;border-radius:17px;
    padding:8px 14px;margin:0 0 11px;box-shadow:0 5px 20px #16344c0b}
.uneti-brand{display:flex;align-items:center;gap:9px;flex-shrink:0;white-space:nowrap}
.uneti-brand-logo{font-size:25px;color:#00a5aa;line-height:1}
.uneti-brand-main{font-size:15px;font-weight:800;color:var(--ink)}
.uneti-brand-sub{font-size:10px;font-weight:600;color:var(--muted)}
.uneti-links{display:flex;gap:4px;align-items:center;min-width:0;flex-wrap:nowrap}
.uneti-links a{padding:9px 11px;white-space:nowrap;text-decoration:none !important;
    color:#435d74 !important;font-size:11px;font-weight:700;border-radius:11px}
.uneti-links a.active,.uneti-links a:hover{background:#e0f7fa;color:#08788b !important}
.uneti-tag{padding:9px 13px;border-radius:25px;white-space:nowrap;
    color:white;background:linear-gradient(110deg,#06c3bc,#134169);font-size:11px;font-weight:800;flex-shrink:0}
/* Banner thấp, ngang, ảnh nằm phía phải bên trong, tuyệt đối KHÔNG che biểu mẫu. */
.uneti-hero{position:relative;isolation:isolate;overflow:hidden;height:178px;max-height:178px;
    background:linear-gradient(110deg,#102c49 0%,#08607a 60%,#0b8797 100%);
    border-radius:21px;box-shadow:0 10px 24px #16374a1b;margin-bottom:12px}
.uneti-hero-image{position:absolute;inset:0 0 0 auto;width:40%;height:100%;
    background-size:cover;background-position:center 60%;opacity:.95;
    -webkit-mask-image:linear-gradient(90deg,transparent,black 35%);
    mask-image:linear-gradient(90deg,transparent,black 35%);z-index:0}
.uneti-hero-content{position:relative;z-index:1;width:73%;padding:16px 26px;color:#fff}
.uneti-eyebrow{font-size:9px;letter-spacing:.20em;font-weight:800;color:#a9eeee}
.uneti-hero h1{font-size:clamp(20px,1.8vw,27px);line-height:1.19;font-weight:800;
    color:white;margin:7px 0 5px}
.uneti-hero p{font-size:11px;line-height:1.50;font-weight:500;color:#d4f1f7;margin:0;max-width:780px}
.uneti-hero-chips{display:flex;gap:8px;margin-top:9px;flex-wrap:wrap}
.uneti-hero-chips span{font-size:9px;padding:4px 10px;border:1px solid #ffffff49;
    border-radius:30px;background:#ffffff18;color:#efffff;font-weight:700}
/* Sections và form: 2 nửa song song để giảm cuộn trang. */
.uneti-heading{display:flex;align-items:center;gap:10px;margin:5px 0 8px;color:var(--ink)}
.uneti-heading-icon{width:30px;height:30px;border-radius:10px;background:#dcf7fa;
    display:grid;place-items:center;color:#067f94;font-size:16px;font-weight:800}
.uneti-heading b{font-size:clamp(15px,1.1vw,20px);line-height:1.3;font-weight:800}
.uneti-muted{font-size:10px;color:var(--muted)}
.uneti-group-label{font-size:12px;font-weight:800;color:#24465d;margin:1px 0 2px}
.uneti-group-help{font-size:10px;line-height:1.5;color:#7990a0}
[data-testid="stButton"] button{border:1px solid #d4e5ef;background:#fff;color:#193c55;
    border-radius:12px;font-size:12px;font-weight:700;min-height:38px;padding:.3rem .7rem}
[data-testid="stButton"] button:hover{background:#e8fbfd;color:#076d7c;border-color:#10b8bb}
/* Chỉ áp dụng bên trong form; giảm khoảng cách nhưng không dùng margin âm. */
[data-testid="stForm"]{background:#fff;border:1px solid #deebf3;border-radius:18px;
    padding:13px 16px 11px;box-shadow:0 7px 22px #193e5c0e}
[data-testid="stForm"] [data-testid="stVerticalBlock"]{gap:.34rem !important}
[data-testid="stForm"] [data-testid="stWidgetLabel"] p{font-size:10px;font-weight:700;color:#4a6479}
[data-testid="stForm"] [data-baseweb="select"]>div{min-height:36px;border-radius:10px;background:#f8fbfd}
[data-testid="stForm"] [data-testid="stNumberInput"] input{min-height:36px;border-radius:9px;background:#f8fbfd}
[data-testid="stFormSubmitButton"] button{min-height:44px !important;border:0 !important;
    background:linear-gradient(103deg,#0ac8c3,#048999 55%,#123e62) !important;
    color:#fff !important;font-size:14px !important;font-weight:800 !important;
    border-radius:12px !important;box-shadow:0 8px 15px #148e962a}
[data-testid="stFormSubmitButton"] button:hover{filter:brightness(1.06)}
/* Kết quả hiển thị ngang 3 cột, dữ liệu minh bạch lấy từ model_runtime. */
.uneti-results{display:grid;grid-template-columns:1.08fr .95fr 1.1fr;gap:12px;margin:6px 0 12px}
.uneti-result-card{background:white;border:1px solid #dce9f2;border-radius:16px;padding:15px 17px;
    min-width:0;min-height:150px;box-shadow:0 6px 20px #102d4910}
.uneti-primary-card{background:linear-gradient(110deg,#e3faef,#e9f9fc);border-color:#bfeadf}
.uneti-card-label{font-size:12px;color:#214c65;font-weight:800;margin-bottom:8px}
.uneti-number{font-size:clamp(25px,2vw,36px);color:#07596b;font-weight:800;letter-spacing:-.03em}
.uneti-number-sub{font-size:17px;font-weight:800;color:#176077;margin:1px 0 7px}
.uneti-small{font-size:10px;color:#58798d;line-height:1.55}
.uneti-blend-row{display:grid;grid-template-columns:minmax(76px,110px) 1fr 43px;
    gap:7px;align-items:center;margin:9px 0;font-size:10px;color:#496479}
.uneti-track{height:8px;border-radius:15px;background:#edf2f5;overflow:hidden}
.uneti-fill{height:100%;border-radius:15px;background:linear-gradient(90deg,#008fa9,#17c0ba)}
.uneti-mini-row{display:grid;grid-template-columns:minmax(72px,113px) 1fr 62px;
    gap:7px;align-items:center;margin:8px 0;font-size:10px;color:#4c687c}
.uneti-mini-row strong{text-align:right;color:#164e62}
.uneti-mini-fill{height:100%;border-radius:10px;background:linear-gradient(90deg,#62d0e8,#07aaa7)}
.uneti-empty{font-size:12px;color:#7290a1;line-height:1.7;margin-top:17px}
/* Expander để Streamlit vẽ gốc: không chèn HTML/CSS vào summary! */
[data-testid="stExpander"]{border-radius:12px;border:1px solid #dbe9f1;background:#fff;
    overflow:visible;margin-top:4px}
.uneti-footer{text-align:center;font-size:10px;color:#8499a8;margin-top:12px;padding-bottom:12px}
@media(max-width:1330px){
 .uneti-links a{padding:7px;font-size:10px}
 .uneti-tag{display:none}
 .uneti-blend-row{grid-template-columns:90px 1fr 38px}
}
@media(max-width:960px){
 .uneti-links a:nth-child(n+3){display:none}
 .uneti-hero-content{width:88%}
 .uneti-results{grid-template-columns:1fr 1fr}
 .uneti-results>.uneti-result-card:last-child{grid-column:1/-1}
}
@media(max-width:700px){
 .uneti-nav{padding:9px}
 .uneti-links{display:none}
 .uneti-hero{height:auto;max-height:none;min-height:175px}
 .uneti-hero-image{width:80%;opacity:.18}
 .uneti-hero-content{width:100%;padding:20px}
 .uneti-hero h1{font-size:21px}
 .uneti-results{grid-template-columns:1fr}
 .uneti-results>.uneti-result-card:last-child{grid-column:auto}
 [data-testid="stMainBlockContainer"],.main .block-container{
    padding-inline:.75rem !important;
    padding-top:4.75rem !important;
 }
 [data-testid="stForm"]{padding:12px}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Một HTML block hoàn chỉnh, không lồng widget Streamlit vào <div> chưa đóng.
st.markdown(
    '<div class="uneti-nav">'
    '<div class="uneti-brand"><span class="uneti-brand-logo">✿</span>'
    '<span><span class="uneti-brand-main">UNETI18</span><br>'
    '<span class="uneti-brand-sub">Búp Sen Xanh · AI Lab</span></span></div>'
    '<nav class="uneti-links">'
    '<a class="active" href="#trang-chu">⌂ Trang chủ</a>'
    '<a href="#du-doan">↗ Dự đoán</a>'
    '<a href="#gioi-thieu">⬡ Giới thiệu</a>'
    '<a href="#tai-lieu">▤ Hướng dẫn</a>'
    '</nav><span class="uneti-tag">✦ AI TAXI FORECAST</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<div id="trang-chu" class="uneti-hero">'
    f'<div class="uneti-hero-image" style="{image_style}"></div>'
    '<div class="uneti-hero-content">'
    '<div class="uneti-eyebrow">UNETI18 · BÚP SEN XANH · OLYMPIC AI 2026</div>'
    '<h1>🚕 Dự đoán thời gian di chuyển taxi tại New York</h1>'
    '<p>Ứng dụng mô hình kết hợp LightGBM và CatBoost trên dữ liệu taxi New York. '
    'Nhập thông tin chuyến đi để dự đoán thời gian bằng giây và phút.</p>'
    '<div class="uneti-hero-chips"><span>✓ Mô hình đã huấn luyện</span>'
    '<span>✦ Dự đoán bằng mô hình AI thật</span></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

if not BUNDLE.is_file():
    st.error("Chưa tìm thấy bundle.json; cần giữ bộ mô hình gốc ở cùng thư mục với app.py.")
    st.stop()

try:
    ensemble = load_ensemble()
except Exception as error:
    st.error("Không thể nạp mô hình đã huấn luyện. Kiểm tra bundle.json và thư mục models/.")
    st.exception(error)
    st.stop()

rmsle = ensemble.manifest.get("validation_rmsle")
with st.sidebar:
    st.markdown('<div class="uneti-card-label" style="color:#f4fdff;font-size:15px">⬡ &nbsp; Trạng thái mô hình</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="background:linear-gradient(110deg,#086c5f,#125258);border:1px solid #17aa93;'
        'border-radius:14px;padding:14px;margin:10px 0 15px;color:white">'
        '✅ <b>Sẵn sàng dự đoán</b><div style="font-size:10px;margin-top:6px;color:#e0fdf8">'
        'Đã nạp và xác thực bộ mô hình.</div></div>', unsafe_allow_html=True,
    )
    st.markdown('<div style="font-size:11px;color:#c2d9e6">RMSLE Validation (kiểm định nội bộ)</div>', unsafe_allow_html=True)
    value = f"{float(rmsle):.6f}" if rmsle is not None else "Không có"
    st.markdown(f'<div style="font-size:28px;font-weight:800;color:white;margin:5px 0">{value}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;color:#aec8d9">Càng thấp càng tốt · Không phải RMSLE Test.</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:12px;color:#d7edf7;font-weight:800">Thành phần mô hình</div>', unsafe_allow_html=True)
    for i, item in enumerate(ensemble.items):
        name = escape(str(item["name"]))
        color = ("#45b4ff", "#a98bfa", "#0bd3ce")[i % 3]
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;gap:7px;border-bottom:1px solid #ffffff20;'
            f'padding:9px 0;font-size:11px;color:#e7f7fc"><span>'
            f'<span style="color:{color}">●</span> {name}</span>'
            f'<b style="color:white">{float(item["weight"]):.1%}</b></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div style="border:1px solid #ffffff24;border-radius:12px;background:#ffffff0a;'
                'padding:12px;margin-top:17px;font-size:10px;line-height:1.7;color:#c9e2ed">'
                'LightGBM + CatBoost · 12 thông tin trước khi khởi hành · Không dùng thời gian chuyến thực tế.</div>',
                unsafe_allow_html=True)

st.markdown(
    '<div id="du-doan" class="uneti-heading"><span class="uneti-heading-icon">◎</span>'
    '<b>01. Nhập thông tin chuyến xe</b>'
    '<span class="uneti-muted">&nbsp; · &nbsp; Điền nhanh từ ví dụ hoặc nhập thủ công</span></div>',
    unsafe_allow_html=True,
)

preset1, preset2, preset_hint = st.columns([1, 1, 1.25], gap="small")
if preset1.button("🚕 Ví dụ 1: Chuyến ngắn", key="preset_short", width="stretch"):
    choose_preset("short")
if preset2.button("🛫 Ví dụ 2: Chuyến dài", key="preset_long", width="stretch"):
    choose_preset("long")
with preset_hint:
    st.markdown('<div class="uneti-group-help" style="padding-top:10px">Chọn tình huống và chỉnh các thông số bên dưới.</div>', unsafe_allow_html=True)

with st.form("uneti18_ride_input", clear_on_submit=False):
    left, right = st.columns([1.02, 1.16], gap="large")
    with left:
        st.markdown('<div class="uneti-group-label">◷ &nbsp; Thời gian đón khách</div>', unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns([.8, 1.4, .8, .85], gap="small")
        weekdays = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
        t1.selectbox("Tháng", range(1, 13), key=INPUT_KEYS["pickup_month"])
        t2.selectbox("Thứ", range(7), key=INPUT_KEYS["pickup_dayofweek"],
                     format_func=lambda x: f"{x} · {weekdays[x]}")
        t3.selectbox("Giờ", range(24), key=INPUT_KEYS["pickup_hour"])
        t4.selectbox("Phút", [0, 15, 30, 45], key=INPUT_KEYS["pickup_minute_bucket"])
        st.markdown('<div class="uneti-group-label">▣ &nbsp; Thông tin chuyến đi</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3, gap="small")
        r1.selectbox("Nhà vận tải", [1, 2], key=INPUT_KEYS["vendor_id"])
        r2.selectbox("Hành khách", [1, 2, 3, 4, 5, 6], key=INPUT_KEYS["passenger_count"])
        r3.selectbox("Lưu/đồng bộ", ["N", "Y"], key=INPUT_KEYS["store_and_fwd_flag"])
    with right:
        st.markdown('<div class="uneti-group-label">⌖ &nbsp; Tọa độ đón và đến</div>', unsafe_allow_html=True)
        p1, p2, d1, d2 = st.columns(4, gap="small")
        p1.number_input("Kinh độ đón", min_value=-180.0, max_value=180.0,
                        step=0.01, format="%.2f", key=INPUT_KEYS["pickup_lon_approx"])
        p2.number_input("Vĩ độ đón", min_value=-90.0, max_value=90.0,
                        step=0.01, format="%.2f", key=INPUT_KEYS["pickup_lat_approx"])
        d1.number_input("Kinh độ đến", min_value=-180.0, max_value=180.0,
                        step=0.01, format="%.2f", key=INPUT_KEYS["dropoff_lon_approx"])
        d2.number_input("Vĩ độ đến", min_value=-90.0, max_value=90.0,
                        step=0.01, format="%.2f", key=INPUT_KEYS["dropoff_lat_approx"])
        st.markdown('<div class="uneti-group-label">↗ &nbsp; Khoảng cách đường chim bay</div>', unsafe_allow_html=True)
        km_col, note_col = st.columns([1.08, 2.0], gap="small")
        km_col.number_input("Khoảng cách (km)", min_value=0.0, max_value=1000.0,
                            step=0.5, format="%.1f", key=INPUT_KEYS["straight_line_km_approx"])
        with note_col:
            st.markdown('<div class="uneti-group-help" style="padding-top:9px">'
                        'Nếu thay tọa độ, hãy cập nhật khoảng cách tương ứng. '
                        'Dữ liệu đã làm tròn có thể khác quãng đường thực.</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("✦ Dự đoán thời gian di chuyển  →", type="primary", width="stretch")

if submitted:
    data = {key: st.session_state[INPUT_KEYS[key]] for key in INPUT_KEYS}
    try:
        seconds, component_arrays = ensemble.predict_details(data)
        duration = float(seconds[0])
        if duration <= 0:
            raise ValueError("Thời gian dự đoán không hợp lệ")
        coordinate_km = haversine_km(
            data["pickup_lon_approx"], data["pickup_lat_approx"],
            data["dropoff_lon_approx"], data["dropoff_lat_approx"],
        )
        st.session_state["prediction"] = {
            "seconds": duration,
            "components": {str(name): float(values[0]) for name, values in component_arrays.items()},
            "inputs": data,
            "haversine": coordinate_km,
        }
        history = st.session_state.get("prediction_history", [])
        history.append({"Lần": f"Lần {len(history)+1}", "Giây": round(duration, 2)})
        st.session_state["prediction_history"] = history[-30:]
    except Exception as error:
        st.session_state.pop("prediction", None)
        st.error("Không thể dự đoán với dữ liệu hiện tại; hãy kiểm tra thông số và mô hình.")
        st.exception(error)

st.markdown('<div class="uneti-heading"><span class="uneti-heading-icon">▥</span>'
            '<b>02. Kết quả dự đoán</b>'
            '<span class="uneti-muted">&nbsp; · &nbsp; Từ mô hình AI đã huấn luyện</span></div>',
            unsafe_allow_html=True)

result = st.session_state.get("prediction")
if result:
    total_sec = float(result["seconds"])
    secs_rounded = int(round(total_sec))
    primary = (
        '<div class="uneti-result-card uneti-primary-card">'
        '<div class="uneti-card-label">◷ &nbsp; Thời gian di chuyển dự đoán</div>'
        f'<div class="uneti-number">{total_sec:,.2f} giây</div>'
        f'<div class="uneti-number-sub">≈ {total_sec/60:,.2f} phút</div>'
        f'<div class="uneti-small">{secs_rounded//60} phút {secs_rounded%60} giây · Giá trị ước lượng, không phải thời gian thực tế.</div>'
        '</div>'
    )
else:
    primary = (
        '<div class="uneti-result-card uneti-primary-card">'
        '<div class="uneti-card-label">◷ &nbsp; Thời gian di chuyển dự đoán</div>'
        '<div class="uneti-empty">Chưa có dự đoán. Chọn ví dụ và bấm nút Dự đoán để xem kết quả thật.</div>'
        '</div>'
    )

weight_rows = []
for item in ensemble.items:
    weight = float(item["weight"]) * 100
    weight_rows.append(
        '<div class="uneti-blend-row">'
        f'<span>{escape(str(item["name"]))}</span>'
        f'<div class="uneti-track"><div class="uneti-fill" style="width:{weight:.2f}%"></div></div>'
        f'<b style="text-align:right">{weight:.1f}%</b></div>'
    )
weights = ('<div class="uneti-result-card"><div class="uneti-card-label">◈ &nbsp; Mô hình kết hợp (Blend)</div>'
           + ''.join(weight_rows) + '</div>')

if result:
    model_predictions = list(result["components"].items()) + [("Kết hợp", total_sec)]
    largest = max([float(value) for _, value in model_predictions] + [1.0])
    lines = []
    for name, value in model_predictions:
        value = float(value)
        # Chỉ dùng dự đoán THẬT; đây là so sánh mô hình, không phải biểu đồ phân phối.
        pct = max(1.0, min(100.0, 100.0 * value / largest))
        lines.append('<div class="uneti-mini-row">'
                     f'<span>{escape(str(name))}</span>'
                     f'<div class="uneti-track"><div class="uneti-mini-fill" style="width:{pct:.2f}%"></div></div>'
                     f'<strong>{value:.0f} giây</strong></div>')
    insight = ('<div class="uneti-result-card"><div class="uneti-card-label">▥ &nbsp; So sánh dự đoán các mô hình</div>'
               + ''.join(lines) + '</div>')
else:
    insight = ('<div class="uneti-result-card"><div class="uneti-card-label">▥ &nbsp; So sánh dự đoán</div>'
               '<div class="uneti-empty">Biểu đồ so sánh sẽ xuất hiện sau khi dự đoán. Không hiển thị dữ liệu giả.</div></div>')

st.markdown('<div class="uneti-results">' + primary + weights + insight + '</div>', unsafe_allow_html=True)

if result and result["inputs"]["straight_line_km_approx"] <= 2:
    st.caption("Lưu ý: Chuyến ≤ 2 km có sai số nội bộ cao hơn; kết quả chỉ là ước lượng.")

# CHỈ MỘT expander gốc của Streamlit, không can thiệp icon/summary bằng CSS hoặc HTML.
st.markdown('<div id="gioi-thieu"></div><div id="tai-lieu"></div>', unsafe_allow_html=True)
with st.expander("ⓘ  Giới thiệu mô hình, hướng dẫn và chi tiết dự đoán", expanded=False):
    st.write("Mô hình kết hợp hai LightGBM và một CatBoost, tính trung bình có trọng số trên thang log1p.")
    st.write("RMSLE Validation là chỉ số nội bộ, không phải điểm Test do ban tổ chức chấm.")
    st.write("Hướng dẫn: chọn ví dụ hoặc nhập đủ 12 trường, bấm Dự đoán, kiểm tra giây/phút và so sánh mô hình.")
    st.write("Hạn chế: không quan sát trực tiếp giao thông, thời tiết, thời gian chờ đèn đỏ; tọa độ được làm tròn.")
    if result:
        st.dataframe(pd.DataFrame([{"Mô hình": name, "Dự đoán riêng (giây)": round(float(value), 2)}
                                   for name, value in result["components"].items()]),
                     hide_index=True, width="stretch")
        # Đầu vào có cả số và ký tự (ví dụ store_and_fwd_flag = "N").
        # Ép giá trị về chuỗi TRƯỚC KHI hiển thị để PyArrow không suy diễn
        # toàn cột "Giá trị" là kiểu float và gây ArrowInvalid.
        input_rows = [
            {"Trường": str(field), "Giá trị": str(value)}
            for field, value in result["inputs"].items()
        ]
        input_table = pd.DataFrame(input_rows)
        input_table = input_table.astype({"Trường": "string", "Giá trị": "string"})
        st.dataframe(input_table, hide_index=True, width="stretch")
        st.caption(f'Khoảng cách từ tọa độ làm tròn: {result["haversine"]:.2f} km.')
    history = st.session_state.get("prediction_history", [])
    if len(history) >= 2:
        st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")
        if st.button("Xóa lịch sử", key="clear_history"):
            st.session_state["prediction_history"] = []
            st.rerun()

st.markdown('<div class="uneti-footer">UNETI18 · BÚP SEN XANH · OLYMPIC AI UNETI 2026 · '
            'Nguyễn Tiến Anh · Nguyễn Minh Hùng · Tạ Hồng Hòa</div>', unsafe_allow_html=True)
