@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist bundle.json (
 echo Chua co bundle.json. Vui long chay CAI_DAT_LAN_DAU.bat truoc.
 pause
 exit /b 1
)
py -m streamlit run app.py
if errorlevel 1 (
 echo Chua cai streamlit hoac khong mo duoc ung dung.
 pause
)
