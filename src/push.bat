@echo on
:loop
git push
if %errorlevel% neq 0 (goto loop) else (echo success!)
pause
