@echo off
rem Runs the whole CLePS pipeline. Double-click to start; safe to stop (close the window) and start again:
rem finished runs are kept and skipped, so it continues where it stopped. A run is redone only if its input data
rem or the computing code/settings changed. Progress: output\run_all_status.txt; console: output\run_all_console.txt.
cd /d "%~dp0"
if not exist output mkdir output
call .venv\Scripts\activate.bat
set LOG=output\run_all_console.txt
set ST=output\run_all_status.txt
echo ===== run_all started %date% %time% >> %ST%
python -m pip freeze > requirements-lock.txt

call :step "convert raw data" "python -u convert.py"
call :step "ridge + GBT, E1 + E2" "python -u cleps.py run --models ridge gbt"
call :step "report" "python -u cleps.py report"
call :step "MLP, E1 + E2" "python -u cleps.py run --models mlp"
call :step "report" "python -u cleps.py report"
call :step "LSTM, E1 + E2" "python -u cleps.py run --models lstm"
call :step "report" "python -u cleps.py report"
echo ===== ALL DONE %date% %time% >> %ST%
exit /b 0

:step
echo [%date% %time%] START %~1 >> %ST%
echo ===== %~1 >> %LOG%
%~2 >> %LOG% 2>&1
if errorlevel 1 (
  echo [%date% %time%] FAILED %~1 >> %ST%
  echo ===== FAILED, see run_all_console.txt >> %ST%
  exit 1
)
echo [%date% %time%] DONE  %~1 >> %ST%
exit /b 0
