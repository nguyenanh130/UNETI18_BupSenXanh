"""Create a submission ZIP containing app code AND actual selected model files."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from model_runtime import TaxiEnsemble

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / 'bundle.json'
if not MANIFEST.is_file():
    raise RuntimeError('Chưa đóng gói mô hình. Hãy chạy CAI_DAT_LAN_DAU.bat trước.')
ensemble = TaxiEnsemble(MANIFEST)  # validates hashes and actual models
source_files = [HERE / x for x in (
    'app.py', 'model_runtime.py', 'preprocessing.py',
    'requirements.txt', 'README.md', 'bundle.json',
    'CHAY_UNG_DUNG.bat'
)]
for item in ensemble.items:
    source_files += [HERE / item['file'], HERE / item['spec']]
source_files = sorted(set(source_files))
assert all(f.is_file() for f in source_files)
output = HERE.parent / 'UNETI18_YeuCau5_UngDung_AI_Mini_NOP.zip'
with ZipFile(output, 'w', compression=ZIP_DEFLATED, compresslevel=5) as archive:
    for source in source_files:
        archive.write(source, arcname='UNETI18_APP/' + str(source.relative_to(HERE)).replace('\\','/'))
print('DA DONG GOI:', output)
print('THANH PHAN:',len(ensemble.items),'mo hinh da huan luyen; khong co CSV Train/Test')
print('VIDEO: Phai tu quay 2-3 phut voi 2 tinh huong, nop kem ZIP theo huong dan cua BTC.')
