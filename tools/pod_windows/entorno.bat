@echo off
rem Entorno de la cuenta hxpod: la sesión SSH no tiene perfil, así que todo (temporales, cachés, "home") va a C:\hxpod.
set HX=C:\hxpod
set TEMP=%HX%\tmp
set TMP=%HX%\tmp
set HOME=%HX%\home
set USERPROFILE=%HX%\home
set APPDATA=%HX%\home\appdata
set LOCALAPPDATA=%HX%\home\localappdata
set UV_CACHE_DIR=%HX%\cache
set UV_PYTHON_INSTALL_DIR=%HX%\py
set NUMBA_CACHE_DIR=%HX%\cache\numba
set PYTHONPATH=%HX%\repo
set PYTHONIOENCODING=utf-8
for %%d in (tmp home home\appdata home\localappdata cache cache\numba py repo runs) do if not exist %HX%\%%d mkdir %HX%\%%d
