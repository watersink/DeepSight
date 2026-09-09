::ffmpeg -re -stream_loop -1 -i 0c95571789bf65675ba644af86bea16d.mp4 -c copy -f flv rtmp://10.1.3.21:1935/live/stream01
::ffmpeg -re -stream_loop -1 -i monkeycar.mp4 -c copy -f flv rtmp://10.1.3.21:1935/live/stream01
::ffmpeg -re -stream_loop -1 -i guanlongin.mp4 -c copy -f flv rtmp://10.1.3.21:1935/live/stream01
ffmpeg -re -stream_loop -1 -i guanlongin1.mp4 -c copy -f flv rtmp://10.1.3.21:1935/live/stream01
::ffmpeg -re -stream_loop -1 -i huifengmian.mp4 -c copy -f flv rtmp://10.1.3.21:1935/live/stream01