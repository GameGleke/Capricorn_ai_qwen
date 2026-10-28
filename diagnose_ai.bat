@echo off
rem Диагностика разума Капри: почему она «загугливает» вместо разговора.
rem Окно держим через pause — вывод также сохраняется в ai_diag.txt.
cd /d "%~dp0"
set "PY="
for %%v in (3.13 3.12 3.11 3.14) do if not defined PY py -%%v -c "" >nul 2>nul && set "PY=py -%%v"
if not defined PY where py >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo [Capri] Python not found. Run setup.bat first.
  pause
  exit /b 1
)
%PY% diagnose_ai.py
echo.
echo Полный вывод также в файле ai_diag.txt (папка Капри).
pause
