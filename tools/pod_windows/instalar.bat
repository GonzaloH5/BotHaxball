@echo off
rem Instala Python portable, torch CPU, numpy y numba en C:\hxpod (PC prestada; cuenta hxpod, sin admin).
rem Todo queda en C:\hxpod: sin instalador, sin registro de Windows, sin ventanas. Se corre por SSH:
rem   ssh hxpod@<ip> "cmd /c C:\hxpod\instalar.bat"
call C:\hxpod\entorno.bat
cd /d %HX%
if not exist uv.exe (
  curl.exe -L -s -o uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
  tar -xf uv.zip
  del uv.zip
)
uv.exe --version
uv.exe python install 3.12
if not exist venv\Scripts\python.exe uv.exe venv venv --python 3.12
uv.exe pip install --python venv\Scripts\python.exe numpy numba
uv.exe pip install --python venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\python.exe -c "import torch, numpy, numba; print('torch', torch.__version__, 'numpy', numpy.__version__, 'numba', numba.__version__)"
echo INSTALADO
