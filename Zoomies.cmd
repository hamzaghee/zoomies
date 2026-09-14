@echo off
rem ---------------------------------------------------------------------
rem  Zoomies - double-click launcher.
rem
rem  Uses pythonw.exe so no black console window sits behind the app.
rem  If Python cannot be found, this window stays open and says so rather
rem  than flashing shut and leaving you guessing.
rem ---------------------------------------------------------------------
setlocal
set "HERE=%~dp0"

set "PYW="
for %%P in (
  "C:\Python314\pythonw.exe"
  "%LOCALAPPDATA%\Programs\Python\Python314\pythonw.exe"
  "%LOCALAPPDATA%\Programs\Python\Python313\pythonw.exe"
) do if not defined PYW if exist %%P set "PYW=%%~P"

if not defined PYW (
  for /f "delims=" %%P in ('where pythonw.exe 2^>nul') do if not defined PYW set "PYW=%%P"
)

if not defined PYW (
  echo.
  echo   Zoomies could not find Python on this machine.
  echo.
  echo   Install Python 3.11 or newer from https://www.python.org/downloads/
  echo   and tick "Add python.exe to PATH" during setup, then run this again.
  echo.
  pause
  exit /b 1
)

start "" "%PYW%" "%HERE%app.py"
endlocal
