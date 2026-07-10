@echo off
cls
title Compiler
echo LWMP Builder
echo Checking dependencies
pip install --no-warn-script-location numpy cython setuptools pyopengl pyopengl_accelerate pygame dearpygui tqdm

echo Dependencies cleared, move on?
pause

goto WIN

:WIN
echo Building corresponding cython files
python setup.py build_ext --inplace
echo Setup complete. (if .pyd not built, an error has occured,)

echo Building midi_batch_send.dll
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo Warning: Visual Studio C++ tools not found, skipping midi_batch_send.dll build
    goto DONE
)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
cl.exe /LD /O2 "%~dp0MIDI\midi_batch_send.c" /Fe:"%~dp0MIDI\midi_batch_send.dll" >nul 2>&1
if exist "%~dp0MIDI\midi_batch_send.dll" (
    echo midi_batch_send.dll built successfully
) else (
    echo Warning: Failed to build midi_batch_send.dll
)
del /q "%~dp0MIDI\midi_batch_send.obj" "%~dp0MIDI\midi_batch_send.lib" "%~dp0MIDI\midi_batch_send.exp" 2>nul

:DONE
pause
