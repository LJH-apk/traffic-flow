@echo off
echo ========================================
echo   Traffic Dashboard - Build Script
echo ========================================
echo.

echo [Check] Verifying Python installation...
python --version
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ first.
    pause
    exit /b 1
)
echo.

echo [1/4] Upgrading pip and core tools...
python -m pip install --upgrade pip setuptools wheel
echo.

echo [2/4] Installing PyInstaller...
python -m pip install pyinstaller
if errorlevel 1 (
    echo [ERROR] Failed to install pyinstaller
    pause
    exit /b 1
)
echo.

echo [3/4] Installing project dependencies (dashboard only, no ML/vision)...
python -m pip install numpy scipy matplotlib openpyxl
if errorlevel 1 (
    echo [WARN] Some packages may have failed. Check above for details.
)
echo.

echo [4/4] Building EXE with PyInstaller...
echo   (this may take 5-10 minutes)...
echo.
python -m PyInstaller build_win.spec --noconfirm --log-level=WARN
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Build complete!
echo ========================================
echo.
echo   EXE location: dist\Traffic-Dashboard\Traffic-Dashboard.exe
echo.
echo   Before running, drop these folders into:
echo     dist\Traffic-Dashboard\_internal\
echo       - outputs\         (CSV + dashboard JSON data)
echo       - src\assets\data\     (video MP4 files)
echo       - src\assets\models\   (model .pt files)
echo.
pause
