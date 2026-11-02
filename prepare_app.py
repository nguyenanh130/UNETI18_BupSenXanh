"""Bundle the *already trained and selected* UNETI18 ensemble for YC5.
Run once from this folder or with: python UNETI18_APP/prepare_app.py
This script never retrains a model. It verifies against the submitted Test CSV.
"""

from pathlib import Path
import json
import pickle
import shutil
import sys

import numpy as np
import pandas as pd

from model_runtime import TaxiEnsemble, sha256

APP = Path(__file__).resolve().parent
ROOT = APP.parent
OUTPUT = ROOT / 'KET_QUA_UNETI18_CHOT_NOP'
TRAINED = OUTPUT / 'mo_hinh'
BUNDLE = APP / 'models'
REPORT = OUTPUT / '06_thong_tin_lan_chay.json'
SUBMISSION = OUTPUT / 'UNETI18_predictions_VONG2_CHOT.csv'
TEST_NAMES = ['UNETI05_TEST_3000.csv', 'UNETI05_TEST_3000(1).csv']


def fail(msg):
    raise RuntimeError(msg + '\nKHÔNG cần huấn luyện lại. Hãy kiểm tra các file mô hình đã lưu từ notebook.')


def find_test():
    for folder in (ROOT, ROOT/'data', ROOT/'DATA', ROOT/'dataset'):
        for name in TEST_NAMES:
            p = folder/name
            if p.is_file():
                return p
    fail('Không tìm thấy UNETI05_TEST_3000.csv cạnh notebook.')


def export():
    if not REPORT.is_file() or not SUBMISSION.is_file():
        fail('Thiếu 06_thong_tin_lan_chay.json hoặc CSV kết quả. Hãy chạy Ô 09 của notebook cũ.')
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    if report.get('fast_test_only'):
        fail('Notebook từng chạy ở chế độ FAST_TEST_ONLY, không thể làm ứng dụng nộp thi.')
    selection = report.get('selection', {})
    weights = selection.get('weights', {})
    if not weights:
        fail('Không thấy trọng số ensemble. Nếu đang dùng CACHE_V2, cần sử dụng mô hình V2 và bộ xuất riêng.')
    if report.get('source', '').startswith('Đã dùng V2 cache'):
        fail('Bản hiện tại dùng V2 cache, không phải ensemble mới. Dừng để tránh sai mô hình.')
    if not TRAINED.is_dir():
        fail('Không tìm thấy thư mục mô hình đã huấn luyện.')
    test_file = find_test()
    if 'test_sha256' in report and sha256(test_file) != report['test_sha256']:
        fail('Test CSV hiện tại khác file đã dùng để xuất kết quả nộp.')
    test = pd.read_csv(test_file)
    reference = pd.read_csv(SUBMISSION)
    if list(reference.columns) != ['trip_id', 'trip_duration'] or len(test) != 3000:
        fail('Test hoặc CSV nộp không có đúng định dạng.')
    if not reference.trip_id.equals(test.trip_id):
        fail('Thứ tự trip_id giữa Test và CSV nộp không khớp.')

    BUNDLE.mkdir(parents=True, exist_ok=True)
    manifest_items = []
    for name, weight in weights.items():
        matching = []
        for meta in TRAINED.glob('*.json'):
            try:
                x = json.loads(meta.read_text(encoding='utf-8'))
                sig = x['signature']
                if sig.get('name') != name:
                    continue
                if sig.get('data_sha256') != [report['train_sha256'], report['test_sha256']]:
                    continue
                kind, mode = name.split('_', 1)
                ext = 'txt' if kind == 'lgb' else 'cbm'
                model = meta.with_suffix('.' + ext)
                spec = meta.with_suffix('.pkl')
                if model.is_file() and spec.is_file():
                    matching.append((meta, model, spec, kind, mode))
            except (OSError, ValueError, TypeError, KeyError):
                continue
        if len(matching) != 1:
            fail(f'Tìm thấy {len(matching)} bộ mô hình {name} hợp lệ, cần đúng 1. Có thể đã dùng thư mục cache khác.')
        meta, source_model, source_spec, kind, mode = matching[0]
        target_model = BUNDLE / f'{name}.{source_model.suffix.lstrip(".")}'
        target_spec = BUNDLE / f'{name}.pkl'
        shutil.copyfile(source_model, target_model)
        shutil.copyfile(source_spec, target_spec)
        manifest_items.append({
            'name': name, 'kind': kind, 'mode': mode,
            'weight': float(weight),
            'file': f'models/{target_model.name}',
            'spec': f'models/{target_spec.name}',
            'sha256': sha256(target_model), 'spec_sha256': sha256(target_spec)
        })
        print(f'Đã đóng gói: {name}, trọng số {weight:.4f}')

    manifest = {
        'format': 'UNETI18_ENS_V1', 'team': 'UNETI18 - Búp sen xanh',
        'target': 'trip_duration (seconds)',
        'validation_rmsle': selection.get('validation_rmsle'),
        'tune_rmsle': selection.get('tune_rmsle'),
        'audit_rmsle': selection.get('audit_rmsle'),
        'training_data_sha256': report.get('train_sha256'),
        'models': manifest_items
    }
    # Ensure all components load, and exact predictions reproduce the prior CSV.
    manifest_path = APP / 'bundle.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    predictor = TaxiEnsemble(manifest_path)
    computed, _ = predictor.predict_details(test)
    original = reference.trip_duration.to_numpy(dtype=float)
    max_diff = float(np.max(np.abs(computed - original)))
    if not np.allclose(computed, original, rtol=1e-6, atol=5e-3):
        manifest_path.unlink(missing_ok=True)
        fail(f'Mô hình đóng gói KHÔNG tái tạo được CSV nộp. Sai lệch tối đa {max_diff:.6f} giây. KHÔNG dùng cho demo.')
    print(f'PASS: Dự đoán lại 3000 chuyến Test khớp CSV đã nộp. Max diff = {max_diff:.6f} giây.')
    print(f'Ứng dụng đã sẵn sàng: {APP / "app.py"}')
    return manifest_path


if __name__ == '__main__':
    try:
        export()
    except Exception as exc:
        print('LỖI ĐÓNG GÓI:', str(exc), file=sys.stderr)
        raise SystemExit(1)
