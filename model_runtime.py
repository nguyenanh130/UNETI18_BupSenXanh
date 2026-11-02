"""Load trained UNETI18 ensemble and make real predictions from fresh inputs.

LightGBM is loaded from its text using Python's UTF-8 reader to support
Windows paths containing Vietnamese characters. CatBoost models are first
copied to an ASCII-only temporary filename before loading.
"""

from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os
import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor

from preprocessing import RAW_FEATURES, make_features, transform, from_log


@contextmanager
def _ascii_catboost_copy(path):
    # CatBoost's native loader may have trouble with non-ASCII model paths.
    with tempfile.TemporaryDirectory(prefix="uneti18_cb_") as directory:
        short_path = Path(directory) / "model.cbm"
        shutil.copyfile(path, short_path)
        yield short_path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class TaxiEnsemble:
    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.directory = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
        if self.manifest.get('format') != 'UNETI18_ENS_V1':
            raise ValueError('Manifest mô hình không hợp lệ.')
        self.items = []
        for item in self.manifest.get('models', []):
            weight = float(item['weight'])
            if not (0 < weight <= 1):
                raise ValueError('Trọng số mô hình không hợp lệ')
            # Artifact filenames are fixed by exporter, never loaded from UI input.
            p = (self.directory / item['file']).resolve()
            sp = (self.directory / item['spec']).resolve()
            if not (p.is_relative_to(self.directory) and sp.is_relative_to(self.directory)):
                raise ValueError('Đường dẫn mô hình không hợp lệ')
            for path, expected_key in ((p, 'sha256'), (sp, 'spec_sha256')):
                if not path.is_file():
                    raise FileNotFoundError(f'Thiếu tệp: {path}')
                if sha256(path) != item[expected_key]:
                    raise ValueError(f'SHA-256 không đúng, tệp thay đổi: {path.name}')
            with sp.open('rb') as file:
                spec = pickle.load(file)  # Only load trusted model files made by this project!
            if item['kind'] == 'lgb':
                model = lgb.Booster(model_str=p.read_text(encoding='utf-8'))
            elif item['kind'] == 'cat':
                model = CatBoostRegressor()
                with _ascii_catboost_copy(p) as tmp:
                    model.load_model(str(tmp))
            else:
                raise ValueError('Loại mô hình không được hỗ trợ')
            self.items.append({**item, 'spec_obj': spec, 'model_obj': model})
        if not self.items:
            raise ValueError('Không có mô hình nào được chọn')
        if not np.isclose(sum(x['weight'] for x in self.items), 1.0, atol=1e-5):
            raise ValueError('Tổng trọng số mô hình khác 1')

    def predict_details(self, inputs):
        """Predict a fresh row or batch; return blend and component predictions."""
        raw = pd.DataFrame([inputs]) if isinstance(inputs, dict) else inputs.copy()
        if len(raw) == 0:
            raise ValueError('Không có chuyến đi đầu vào')
        missing = set(RAW_FEATURES).difference(raw.columns)
        if missing:
            raise ValueError('Thiếu trường: ' + ', '.join(sorted(missing)))
        blended_log = np.zeros(len(raw), dtype=np.float64)
        components = {}
        for item in self.items:
            fts, _ = make_features(raw, item['mode'])
            fts = transform(fts, item['spec_obj'], for_catboost=item['kind'] == 'cat')
            model = item['model_obj']
            if item['kind'] == 'lgb':
                pred_log = np.asarray(model.predict(fts, num_threads=min(6, os.cpu_count() or 1)), dtype=np.float64)
            else:
                pred_log = np.asarray(model.predict(fts, thread_count=min(6, os.cpu_count() or 1)), dtype=np.float64)
            if len(pred_log) != len(raw) or not np.isfinite(pred_log).all():
                raise ValueError(f'Mô hình {item["name"]} trả về dự đoán không hợp lệ')
            blended_log += item['weight'] * pred_log
            components[item['name']] = from_log(pred_log)
        seconds = from_log(blended_log)
        if not np.isfinite(seconds).all() or np.any(seconds <= 0):
            raise ValueError('Dự đoán không hợp lệ')
        return seconds, components


def haversine_km(lon1, lat1, lon2, lat2):
    l1, a1, l2, a2 = np.deg2rad(np.array([lon1, lat1, lon2, lat2], dtype=float))
    t = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((l2-l1)/2)**2
    return float(2*6371.0088*np.arcsin(np.sqrt(np.clip(t, 0, 1))))
