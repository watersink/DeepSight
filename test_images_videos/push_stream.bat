@echo off
chcp 65001 >nul
setlocal EnableExtensions

:: 切到脚本所在目录（视频文件同目录）
cd /d "%~dp0"

set "RTMP_BASE=rtmp://10.1.3.21:1935/live"
set "FFMPEG=ffmpeg"

where %FFMPEG% >nul 2>&1
if errorlevel 1 (
  echo [错误] 未找到 ffmpeg，请先安装并加入 PATH。
  pause
  exit /b 1
)

echo ========================================
echo  本地测试流：一键并行推送
echo  目标: %RTMP_BASE%/^<stream^>
echo  关闭各推流窗口或结束对应 ffmpeg 即可停推
echo ========================================
echo.

:: 格式: call :push 视频文件 流名
:: 同一 RTMP 地址只能有一路推流，勿重复 stream 名

:: call :push "0c95571789bf65675ba644af86bea16d.mp4" "stream01"
call :push "rujing.mp4" "rujing"
:: call :push "monkeycar.mp4" "monkeycar"
:: call :push "guanlongin.mp4" "guanlongin"
:: call :push "guanlongin1.mp4" "guanlongin1"
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

echo.
echo 已尝试启动全部推流窗口（最小化）。
echo 按任意键退出本控制台（不会自动杀掉已启动的 ffmpeg）。
pause >nul
exit /b 0

:: ------------------------------------------------------------
:push
set "FILE=%~1"
set "STREAM=%~2"
if not exist "%FILE%" (
  echo [跳过] 缺少文件: %FILE%
  goto :eof
)
echo [启动] %FILE%  -^>  %RTMP_BASE%/%STREAM%
start "push-%STREAM%" /MIN %FFMPEG% -hide_banner -loglevel warning -re -stream_loop -1 -i "%FILE%" -c copy -f flv "%RTMP_BASE%/%STREAM%"
goto :eof
