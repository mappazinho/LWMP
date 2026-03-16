@echo off
cls
title Compiler
echo LWMP Builder
echo Checking dependencies
pip install --no-warn-script-location numpy Cython setuptools

echo Dependencies cleared, move on?
pause

goto WIN

:WIN
echo Building corresponding cython files
python setup.py build_ext --inplace
echo Setup complete. (if .pyd not built, an error has occured,)
pause
