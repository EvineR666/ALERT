@echo on
:loop
git pull
if %errorlevel% neq 0 (goto loop) else (echo success!)
pause