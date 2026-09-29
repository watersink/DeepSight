@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "RTMP_BASE=rtmp://10.1.3.21:1935/live"
set "FFMPEG=ffmpeg"

where %FFMPEG% >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ffmpeg not found in PATH
  pause
  exit /b 1
)

echo ========================================
echo  DeepSight test push - start all streams
echo  base: %RTMP_BASE%
echo ========================================
echo.

echo [1/2] Killing previous live push ffmpeg ...
call :kill_all_pushes
:: ZLM may keep the old publisher session briefly; wait before republish
timeout /t 5 /nobreak >nul
echo [1/2] Done
echo.

echo [2/2] Starting streams ...
:: call :push "0c95571789bf65675ba644af86bea16d.mp4" "stream01"
call :push "rujing.mp4" "rujing"
:: call :push "monkeycar.mp4" "monkeycar"
:: call :push "guanlongin.mp4" "guanlongin"
call :push "guanlongin1.mp4" "guanlongin1"
:: call :push "huifengmian.mp4" "huifengmian"
call :push "rotate.mp4" "rotate"
call :push "nuoyi.mp4" "nuoyi"
call :push "zhedang.mp4" "zhedang"
call :push "mohu.mp4" "mohu"
call :push "guobao.mp4" "guobao"
call :push "guoan.mp4" "guoan"
call :push "freeze.mp4" "freeze"
call :push "doudong.mp4" "doudong"
call :push "diushi.mp4" "diushi"
call :push "fenbianlv.mp4" "fenbianlv"

echo.
echo All push windows started minimized.
echo Closing this console will NOT stop ffmpeg.
echo Press any key to exit this console.
pause >nul
exit /b 0

:: ------------------------------------------------------------
:kill_all_pushes
for %%S in (
  stream01 rujing monkeycar guanlongin guanlongin1 guanlongin2 huifengmian
  rotate nuoyi zhedang mohu guobao guoan freeze doudong diushi
) do (
  taskkill /FI "WINDOWTITLE eq push-%%S*" /F >nul 2>&1
)

powershell -NoProfile -Command ^
  "$procs = @(Get-CimInstance Win32_Process -Filter \"Name='ffmpeg.exe'\" | Where-Object { $_.CommandLine -match 'rtmp://.*/live/' });" ^
  "foreach ($p in $procs) { Write-Host ('  kill PID=' + $p.ProcessId); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }"
goto :eof

:: ------------------------------------------------------------
:push
set "FILE=%~1"
set "STREAM=%~2"
if not exist "%FILE%" (
  echo [SKIP] missing file: %FILE%
  goto :eof
)
echo [START] %FILE%  -^>  %RTMP_BASE%/%STREAM%
:: Same command line as manual: ffmpeg -re -stream_loop -1 -i ... -c copy -f flv ...
start "push-%STREAM%" /MIN %FFMPEG% -re -stream_loop -1 -i "%FILE%" -c copy -f flv "%RTMP_BASE%/%STREAM%"
:: gap between publishes; avoids ZLM Already publishing when restarting many streams
timeout /t 1 /nobreak >nul
goto :eof
