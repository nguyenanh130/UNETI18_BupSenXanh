@echo off
chcp 65001 >nul
cd /d "%~dp0"
py create_submission_zip.py
if errorlevel 1 (
 echo LOI DONG GOI. Hay chay CAI_DAT_LAN_DAU.bat truoc.
)
pause
