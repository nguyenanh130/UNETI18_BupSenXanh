@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===== CAI DAT UNETI18 YEU CAU 5 =====
py -m pip install -r requirements.txt
if errorlevel 1 goto :error
py prepare_app.py
if errorlevel 1 goto :error
echo.
echo ===== THANH CONG. DANG MO UNG DUNG =====
py -m streamlit run app.py
pause
exit /b 0
:error
echo.
echo LOI: xem thong bao phia tren. Khong can train lai model.
pause
exit /b 1
