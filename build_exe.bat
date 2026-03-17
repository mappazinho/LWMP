@echo off
cls
title LWMP EXE Builder

echo LWMP standalone EXE builder
echo.
echo [1/3] Installing PyInstaller...
python -m pip install --no-warn-script-location pyinstaller
if errorlevel 1 (
    echo Failed to install PyInstaller.
    pause
    exit /b 1
)

echo.
echo [2/3] Rebuilding Cython modules...
python setup.py build_ext --inplace
if errorlevel 1 (
    echo Failed to build Cython modules.
    pause
    exit /b 1
)

echo.
echo [3/3] Building one-file executable...
python -m PyInstaller --noconfirm --clean lwmp_dpg_onefile.spec
if errorlevel 1 (
    echo PyInstaller build failed.
    pause
    exit /b 1
)

echo.
echo Build complete.
echo Output: dist\LWMP.exe
pause
