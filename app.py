"""UNETI18 – Olympic AI 2026 YC5. Predicts using actual trained models."""

from pathlib import Path
import pandas as pd
import streamlit as st

from model_runtime import TaxiEnsemble, haversine_km

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'bundle.json'

st.set_page_config(page_title='UNETI18 | Dự đoán thời gian taxi', page_icon='🚕', layout='wide')

st.markdown('''
<style>
.block-container {max-width: 1080px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(105deg,#142d42,#235a69);
       color:white; margin: 0 0 1.35rem 0;}
.hero .overline {font-size:.78rem; letter-spacing:.12em; color:#c1dce4; font-weight:700;}
.hero .headline {font-size:1.7rem; line-height:1.22; font-weight:780; margin:.45rem 0 .4rem;}
.hero .subtitle {color:#d5e6e8; font-size:.93rem;}
[data-testid="stMetricValue"]{font-size:1.85rem;}
</style>
<div class="hero"><div class="overline">UNETI18 · BÚP SEN XANH · OLYMPIC AI 2026</div>
<div class="headline">🚕 Dự đoán thời gian di chuyển taxi New York</div>
<div class="subtitle">Dự đoán trực tiếp từ mô hình LightGBM + CatBoost đã huấn luyện · Kết quả tính bằng giây và phút</div></div>
''', unsafe_allow_html=True)

if not MANIFEST.is_file():
    st.error('Chưa tìm thấy bộ mô hình đã đóng gói. Ứng dụng sẽ không hiển thị dự đoán giả.')
    st.code('py UNETI18_APP/prepare_app.py', language='powershell')
    st.info('Chạy câu lệnh trên từ thư mục olympic vòng 2 sau khi đã hoàn thành notebook huấn luyện. Không cần huấn luyện lại.')
    st.stop()

@st.cache_resource(show_spinner='Đang tải mô hình đã huấn luyện...')
def load_model():
    return TaxiEnsemble(MANIFEST)

try:
    ensemble = load_model()
except Exception as error:
    st.error('Không thể nạp mô hình đã huấn luyện.')
    st.exception(error)
    st.stop()

with st.sidebar:
    st.subheader('Thông tin mô hình')
    st.success('Đã nạp mô hình AI thật')
    val = ensemble.manifest.get('validation_rmsle')
    if val is not None:
        st.metric('RMSLE Validation', f'{float(val):.6f}')
    st.caption('Đây là điểm kiểm định nội bộ, không phải RMSLE Test bí mật của ban tổ chức.')
    st.markdown('**Các thành phần được sử dụng**')
    for item in ensemble.items:
        st.write(f"{item['name']} — {item['weight']:.1%}")
    st.divider()
    st.caption('Ứng dụng sử dụng các trường đầu vào có thể biết khi bắt đầu chuyến xe. Không sử dụng trip_id hoặc thời gian kết thúc.')

PRESETS = {
    'Chuyến ngắn — Manhattan': {
        'vendor_id': 1, 'pickup_month': 3, 'pickup_dayofweek': 5, 'pickup_hour': 18,
        'pickup_minute_bucket': 45, 'passenger_count': 1,
        'pickup_lon_approx': -73.99, 'pickup_lat_approx': 40.72,
        'dropoff_lon_approx': -74.00, 'dropoff_lat_approx': 40.73,
        'straight_line_km_approx': 1.5, 'store_and_fwd_flag': 'N'
    },
    'Chuyến dài — Sân bay JFK': {
        'vendor_id': 2, 'pickup_month': 5, 'pickup_dayofweek': 1, 'pickup_hour': 8,
        'pickup_minute_bucket': 15, 'passenger_count': 2,
        'pickup_lon_approx': -73.78, 'pickup_lat_approx': 40.64,
        'dropoff_lon_approx': -73.98, 'dropoff_lat_approx': 40.76,
        'straight_line_km_approx': 21.5, 'store_and_fwd_flag': 'N'
    }
}

KEYS = {key:'taxi_'+key for key in next(iter(PRESETS.values()))}


def use_preset(name):
    row = PRESETS[name]
    for key, value in row.items():
        st.session_state[KEYS[key]] = value
    st.session_state.pop('taxi_result', None)


if 'taxi_vendor_id' not in st.session_state:
    use_preset('Chuyến ngắn — Manhattan')

st.markdown('#### 1. Chọn tình huống và nhập thông tin chuyến xe')
with st.container():
    c1, c2, c3 = st.columns([1.1, 1.1, 1.1])
    if c1.button('🚕 Ví dụ 1: Chuyến ngắn', use_container_width=True):
        use_preset('Chuyến ngắn — Manhattan')
    if c2.button('🛫 Ví dụ 2: Chuyến dài', use_container_width=True):
        use_preset('Chuyến dài — Sân bay JFK')
    c3.caption('Các ô bên dưới có thể sửa để dự đoán chuyến mới.')

with st.form('ride_form'):
    st.markdown('**Thời gian đón khách**')
    a,b,c,d = st.columns(4)
    a.selectbox('Tháng', list(range(1, 13)), key=KEYS['pickup_month'])
    b.selectbox('Thứ trong tuần (0 = Thứ Hai)', list(range(0, 7)), key=KEYS['pickup_dayofweek'])
    c.selectbox('Giờ đón (0–23)', list(range(24)), key=KEYS['pickup_hour'])
    d.selectbox('Mốc phút', [0,15,30,45], key=KEYS['pickup_minute_bucket'])

    st.markdown('**Thông tin chuyến đi**')
    a,b,c = st.columns(3)
    a.selectbox('Mã nhà vận tải', [1,2], key=KEYS['vendor_id'])
    b.selectbox('Số hành khách', [1,2,3,4,5,6], key=KEYS['passenger_count'])
    c.selectbox('Cờ lưu/đồng bộ', ['N','Y'], key=KEYS['store_and_fwd_flag'])

    st.markdown('**Tọa độ điểm đón và điểm đến**')
    a,b,c,d = st.columns(4)
    a.number_input('Kinh độ đón', min_value=-180.0, max_value=180.0, step=0.01,
                   format='%.2f', key=KEYS['pickup_lon_approx'])
    b.number_input('Vĩ độ đón', min_value=-90.0, max_value=90.0, step=0.01,
                   format='%.2f', key=KEYS['pickup_lat_approx'])
    c.number_input('Kinh độ đến', min_value=-180.0, max_value=180.0, step=0.01,
                   format='%.2f', key=KEYS['dropoff_lon_approx'])
    d.number_input('Vĩ độ đến', min_value=-90.0, max_value=90.0, step=0.01,
                   format='%.2f', key=KEYS['dropoff_lat_approx'])

    a,b = st.columns([1,2])
    a.number_input('Khoảng cách đường chim bay (km)', min_value=0.0,
                   max_value=1000.0, step=0.5, format='%.1f', key=KEYS['straight_line_km_approx'])
    b.caption('Khoảng cách trong đề được làm tròn theo mốc 0,5 km và có thể khác khoảng cách tính từ tọa độ đã làm tròn. Khi đổi điểm đón/đến, nhớ cập nhật cả khoảng cách.')

    submitted = st.form_submit_button('🔎 Dự đoán thời gian chuyến đi', type='primary', use_container_width=True)

if submitted:
    row = {column:st.session_state[KEYS[column]] for column in KEYS}
    try:
        seconds, components = ensemble.predict_details(row)
        estimate = float(seconds[0])
        gap = haversine_km(row['pickup_lon_approx'], row['pickup_lat_approx'],
                           row['dropoff_lon_approx'], row['dropoff_lat_approx'])
        st.session_state['taxi_result'] = {'seconds': estimate, 'inputs': row,
                                           'components': {n:float(v[0]) for n,v in components.items()},
                                           'haversine':gap}
        # Lịch sử chỉ lưu trong phiên Streamlit (không thay đổi mô hình/dữ liệu dự đoán).
        history = st.session_state.get('taxi_history', [])
        history.append({
            'Lần dự đoán': f'Lần {len(history) + 1}',
            'Thời gian (phút)': round(estimate / 60, 3),
            'Khoảng cách (km)': float(row['straight_line_km_approx']),
            'Giờ đón': int(row['pickup_hour']),
        })
        st.session_state['taxi_history'] = history
    except Exception as error:
        st.session_state.pop('taxi_result', None)
        st.error('Không thể dự đoán với đầu vào này. Kiểm tra đủ 12 trường và các mô hình.')
        st.exception(error)

if 'taxi_result' in st.session_state:
    r = st.session_state['taxi_result']
    t = r['seconds']
    st.divider()
    st.markdown('#### 2. Kết quả dự đoán bằng mô hình AI')
    st.success('Đã tính kết quả từ các tệp mô hình đã huấn luyện với thông tin đầu vào hiện tại.')
    a,b,c = st.columns(3)
    a.metric('Dự đoán (giây)', f'{t:,.1f} giây')
    b.metric('Dự đoán (phút)', f'{t / 60:,.2f} phút')
    rounded = int(round(t))
    c.metric('Hiển thị dễ đọc', f'{rounded // 60} phút {rounded % 60} giây')
    if r['inputs']['straight_line_km_approx'] <= 2:
        st.caption('Đây là chuyến ngắn (≤ 2 km). Nhóm này có độ biến động thời gian lớn, dự đoán chỉ mang tính ước lượng.')
    else:
        st.caption('Dự đoán dựa trên thời gian, khoảng cách, tọa độ và các đặc trưng đã huấn luyện.')
    st.markdown('##### Biểu đồ so sánh dự đoán từng mô hình')
    # Các cột đều là dự đoán thật của mô hình đã huấn luyện, không phải số liệu mẫu.
    chart_rows = [
        {'Mô hình': name, 'Thời gian dự đoán (phút)': value / 60}
        for name, value in r['components'].items()
    ]
    chart_rows.append({
        'Mô hình': 'Kết hợp (kết quả cuối)',
        'Thời gian dự đoán (phút)': t / 60
    })
    st.bar_chart(
        pd.DataFrame(chart_rows),
        x='Mô hình', y='Thời gian dự đoán (phút)',
        use_container_width=True,
    )
    st.caption('Kết quả kết hợp tính bằng trung bình có trọng số trong không gian log1p, '
               'không phải trung bình số học trực tiếp của các cột thời gian.')

    history = st.session_state.get('taxi_history', [])
    if len(history) >= 2:
        st.markdown('##### Biểu đồ so sánh những chuyến đã thử')
        st.bar_chart(
            pd.DataFrame(history), x='Lần dự đoán', y='Thời gian (phút)',
            use_container_width=True,
        )
        st.caption('Lịch sử chỉ lưu trong phiên hiện tại và giúp đối chiếu ít nhất 2 đầu vào khác nhau khi demo.')
        if st.button('Xóa lịch sử thử nghiệm'):
            st.session_state['taxi_history'] = []
            st.rerun()

    with st.expander('Xem chi tiết dự đoán từng mô hình và dữ liệu đầu vào'):
        st.dataframe(pd.DataFrame([{'Mô hình':name,'Dự đoán đơn lẻ (giây)':round(v,1)} for name,v in r['components'].items()]), hide_index=True, use_container_width=True)
        st.dataframe(pd.DataFrame([r['inputs']]).T.rename(columns={0:'Giá trị'}), use_container_width=True)
        st.caption(f"Khoảng cách Haversine tính từ tọa độ làm tròn: {r['haversine']:.2f} km — có thể khác cột khoảng cách của BTC.")

st.divider()
st.caption('Đội Búp sen xanh · UNETI18 · Yêu cầu 5 — Olympic Trí tuệ nhân tạo UNETI 2026. Dự đoán là ước lượng, không phải thời gian thực tế hay dữ liệu đã có sẵn.')
