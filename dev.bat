@echo off
chcp 65001 > nul
echo ========================================
echo   KFCS Attendance Pro - Dev Mode
echo ========================================
echo.
echo Watching for file changes...
echo Press Ctrl+C to stop the development server
echo.
pywatch "*.py modules/*.py" "python main.py"
pause