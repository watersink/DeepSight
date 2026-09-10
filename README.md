# Video Detection Stream

基于 FastAPI 的视频流检测推流服务。从 RTSP/RTMP/本地文件读取视频，通过可插拔技能（Skill）调用 [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) 进行 AI 推理，将标注后的画面经 FFmpeg 推送到 [ZLMediaKit](https://github.com/ZLMediaKit/ZLMediaKit) 流媒体服务器（RTMP/RTSP），支持人数告警，并可通过 HTTP API 截取事件视频与中间帧截图。

## 功能特性

- **HTTP API 管理 AI 视频识别任务**：启动、停止、查询任务状态
- **事件视频截取**：基于 ZLMediaKit `startRecordTask`，按流地址截取回溯 + 前向录像，并返回中间帧截图
- **子进程隔离**：每个推流任务在独立进程中运行，互不影响
- **Triton 推理**：AI 算法模型统一部署于 Triton Inference Server（TensorRT 等后端）
- **可插拔技能**：通过 `skill_name` 动态选择检测算法，与 Triton 模型解耦
- **异步处理管道**：检测、推流、告警分线程执行
- **ZLMediaKit 流媒体**：推流目标为 ZLMediaKit，支持 RTMP / RTSP 及 HTTP-FLV、HLS 等播放
- **多协议推流**：经 FFmpeg 向 ZLMediaKit 推流，由流媒体服务分发；任务响应返回 `out_url` 与对应 `flv_url`
- **远程 Worker**：支持 API/前端与编解码+Triton 分机部署；任务可绑定指定 Worker
- **硬件编码**：可选 NVENC（`h264_nvenc`），失败时回退软件编码
- **Swagger 文档**：内置 `/docs` 交互式 API 文档
- **自动化测试**：`app/auto_unit_test/run_tests.py` 一键执行技能、服务端、客户端、本地推流测试

## 系统架构

```
客户端 / 管理台前端
    │
    ▼
FastAPI (app.main)  ← CPU 机：鉴权 / CRUD / 调度 / 选 Worker
    ├── GET  /api/v1/workers
    ├── POST /api/v1/task-configs/{id}/start  ──► runtime_gateway
    └── POST /api/v1/streams/*                ──► runtime_gateway
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
              local 本机进程            Worker-1 (app.worker_main)  Worker-2
              stream_task_manager       解码+技能+FFmpeg编码         （同构副本）
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                           Triton / ZLMediaKit
```

### 远程 Worker 部署（可选）

同一仓库两个入口：

| 角色 | 启动 | 职责 |
|------|------|------|
| API | `python -m app.main` | 前后端、任务配置、调度，按 `worker_id` 调用算力节点 |
| Worker | `python -m app.worker_main` | 解码、调 Triton、编码推流 |

`.env` 示例（1 台 API + 2 台同构 GPU）：

```env
# API 机
WORKER_NODES=gpu1|GPU-1|http://10.1.3.21:8100;gpu2|GPU-2|http://10.1.3.22:8100
DEFAULT_WORKER_ID=gpu1
WORKER_TOKEN=change-me

# 每台 GPU 机同样配置 WORKER_TOKEN / TRITON_URL / ZLM_* / MySQL 等
# 并启动: python -m app.worker_main
```

单机全部署保持默认即可：

```env
WORKER_NODES=local|本机|local
DEFAULT_WORKER_ID=local
```

管理台「任务配置」可选择摄像头、Worker 与该 Worker 下的算法技能。

## 目录结构

```
code/
├── app/
│   ├── main.py                          # API / 前端入口
│   ├── worker_main.py                   # 远程 Worker 入口（编解码 + 调 Triton）
│   ├── core/
│   │   └── config.py                    # 环境配置（含 WORKER_NODES）
│   ├── api/
│   │   ├── schemas.py                   # 请求/响应模型
│   │   └── routes/
│   │       ├── streams.py               # AI 视频识别任务 API
│   │       ├── workers.py               # Worker 节点列表 / 技能
│   │       ├── worker_internal.py       # Worker 内部启停 API
│   │       ├── skills.py                # 技能查询 API
│   │       └── clip.py                  # 事件视频截取 API
│   ├── client_scripts/
│   │   ├── test_stream_client.py        # 推流任务 HTTP 测试客户端
│   │   ├── test_clip_client.py          # 事件截取 HTTP 测试客户端
│   │   └── test_stream_process.py       # 命令行直连推流（不经 API）
│   ├── auto_unit_test/
│   │   └── run_tests.py                 # 自动化测试入口
│   ├── plugins/
│   │   ├── skill_registry.py            # 技能注册与工厂
│   │   └── skills/
│   │       ├── person_count_detector_skill.py
│   │       └── person_count_detector26_skill.py
│   ├── services/
│   │   ├── runtime_gateway.py           # 本机 / 远程 Worker 统一启停
│   │   ├── worker_nodes.py              # Worker 节点配置解析
│   │   ├── stream_task_manager.py       # 多进程任务管理
│   │   ├── stream_worker.py             # 子进程入口
│   │   ├── stream_alert.py              # 默认告警处理
│   │   ├── triton_client.py             # Triton gRPC 客户端
│   │   ├── zlm_client.py                # ZLMediaKit HTTP API 客户端
│   │   ├── video_clip_service.py        # 事件录像截取与截图
│   │   └── video/
│   │       ├── pipeline.py
│   │       ├── frame_reader.py
│   │       ├── async_processor.py
│   │       ├── ffmpeg_streamer.py
│   │       └── types.py
│   └── skills/
│       └── skill_base.py                # 技能基类
├── .env                                 # 本地环境变量（可选，勿提交密钥）
├── requirements.txt
└── README.md
```

## 环境要求


| 依赖                                                                                  | 说明                                    |
| ----------------------------------------------------------------------------------- | ------------------------------------- |
| Python                                                                              | 3.9+                                  |
| [ZLMediaKit](https://github.com/ZLMediaKit/ZLMediaKit)                              | 流媒体引擎，接收推流并提供播放分发                     |
| FFmpeg                                                                              | 需在 `PATH` 中，向 ZLMediaKit 推送 RTMP/RTSP |
| [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) | AI 推理服务，承载检测/识别模型                     |
| NVIDIA GPU + 驱动                                                                     | Triton 容器推理及可选 NVENC 硬件编码             |
| OpenCV                                                                              | 通过 `opencv-python` 安装                 |
| Docker（可选）                                                                          | 用于快速部署 Triton、ZLMediaKit              |


## 安装

```bash
cd code
pip install -r requirements.txt
```

## 配置

在项目根目录 `code/` 下创建 `.env`（可选，均有默认值）：

```env
# 服务
REST_PORT=8000
REST_HOST=127.0.0.1
DEBUG=false
LOG_LEVEL=INFO

# Triton 推理地址
TRITON_URL=10.1.3.21:8201

# ZLMediaKit 流媒体引擎
ZLM_HOST=10.1.3.21
ZLM_RTMP_PORT=1935
ZLM_RTSP_PORT=8554
ZLM_HTTP_PORT=8080
ZLM_SECRET=请填写与 ZLMediaKit config.ini [api] secret 一致的密钥

# 推流默认参数
DEFAULT_OUTPUT_FORMAT=rtmp
DEFAULT_OUT_FPS=10
DEFAULT_SCENE_ID=scene_001
DEFAULT_SKILL_NAME=person_count_detector
DEFAULT_TEST_VIDEO=test_images_videos/person_gouzi.mp4
```


| 变量                        | 默认值                     | 说明                                         |
| ------------------------- | ----------------------- | ------------------------------------------ |
| `REST_PORT`               | `8000`                  | API 监听端口                                   |
| `REST_HOST`               | `127.0.0.1`             | 客户端默认 API 主机                               |
| `DEBUG`                   | `false`                 | 调试模式（热重载）                                  |
| `LOG_LEVEL`               | `INFO`                  | 日志级别                                       |
| `TRITON_URL`              | `10.1.3.21:8201`     | Triton gRPC 地址（对应容器内 `8001` 端口）            |
| `ZLM_HOST`                | `10.1.3.21`          | ZLMediaKit 主机                              |
| `ZLM_RTMP_PORT`           | `1935`                  | RTMP 端口                                    |
| `ZLM_RTSP_PORT`           | `8554`                  | RTSP 端口                                    |
| `ZLM_HTTP_PORT`           | `8080`                  | HTTP 端口（HTTP-FLV / 录像访问 / API）             |
| `ZLM_SECRET`              | （需配置）                   | ZLMediaKit HTTP API 密钥，与 `config.ini` 中一致 |
| `ZLM_DEFAULT_VHOST`       | `__defaultVhost__`      | 默认虚拟主机                                     |
| `ZLM_RECORD_WAIT_BUFFER_MS` | `2000`                | 事件录制结束后额外等待写入完成的毫秒数                        |
| `DEFAULT_SCENE_ID`        | `scene_001`             | 默认场景 ID                                    |
| `DEFAULT_SKILL_NAME`      | `person_count_detector` | 默认技能名称                                     |
| `DEFAULT_OUTPUT_FORMAT`   | `rtmp`                  | 默认推流协议（`rtmp` / `rtsp`）                    |


推流地址默认按 `{协议}://{ZLM_HOST}:{端口}/{scene_id}/{skill_name}` 拼装，例如：

```text
rtmp://10.1.3.21:1935/scene_001/person_count_detector
```

对应 HTTP-FLV 播放地址（任务响应中的 `flv_url`）：

```text
http://10.1.3.21:8080/scene_001/person_count_detector.live.flv
```

API 与客户端在未指定 `out_url` 时，根据请求中的 `scene_id`、`skill_name` 自动生成。

## AI 推理服务（Triton Inference Server）

本项目的 AI 算法模型基于 [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) 部署。各技能（Skill）通过 `app/services/triton_client.py` 以 **gRPC** 方式调用 Triton 上的模型完成推理。

### Docker 部署

将宿主机模型目录挂载到容器 `/models`，并按需指定 GPU 设备。

#### Linux

```bash
docker run --gpus '"device=4"' -itd \
  -p 8200:8000 -p 8201:8001 -p 8202:8002 \
  -v "/home/user/AI-platform/Deploy-models-using-Triton/yolo11_TensorRT/":/models \
  nvcr.io/nvidia/tritonserver:24.06-py3 \
  tritonserver --model-repository=/models \
  --model-control-mode=explicit \
  --load-model=*
```

#### Windows（PowerShell）

将模型仓库路径改为你本机实际目录（示例为 `yolo26` 模型仓库）：

```powershell
docker run --gpus "device=0" -itd `
  -p 8200:8000 `
  -p 8201:8001 `
  -p 8202:8002 `
  -v "C:\Users\Administrator\Desktop\yolo26\Deploy-models-using-Triton\yolo26_TensorRT:/models" `
  nvcr.io/nvidia/tritonserver:24.06-py3 `
  tritonserver `
  --model-repository=/models `
  --model-control-mode=explicit `
  --load-model=*
```

> **说明**
>
> - `-v ...:/models`：模型仓库路径，需包含符合 Triton 规范的模型目录结构（Linux 示例为 `yolo11_TensorRT/`，Windows 示例为 `yolo26/`）
> - `--model-control-mode=explicit`：显式加载模型；`--load-model=`* 加载仓库中全部模型
> - `--gpus "device=0"` / `--gpus '"device=4"'`：绑定指定 GPU，多卡环境请替换为实际设备编号
> - Windows 下路径使用 `-v "C:\path\to\models:/models"`，PowerShell 换行用反引号 ```
> - 应用侧 `TRITON_URL` 应指向宿主机映射的 gRPC 端口，即 `主机IP:8201`

### 端口说明


| 宿主机端口  | 容器端口   | 协议   | 用途                          |
| ------ | ------ | ---- | --------------------------- |
| `8200` | `8000` | HTTP | Triton HTTP API / 健康检查      |
| `8201` | `8001` | gRPC | **本项目推理调用端口**（`TRITON_URL`） |
| `8202` | `8002` | HTTP | Metrics 监控                  |


### 已部署模型


| 模型名称            | 用途          | 关联技能                      | 状态  |
| --------------- | ----------- | ------------------------- | --- |
| `yolo11_person` | 行人检测 / 人数统计 | `person_count_detector`   | 已部署 |
| `yolo26_person` | 行人检测 / 人数统计 | `person_count_detector26` | 按环境部署 |


后续将随业务需求在 Triton 模型仓库中**持续增加**更多模型；新增模型后，在 `app/plugins/skills/` 实现对应技能并在 `skill_registry.py` 注册即可。

### 扩展新模型

1. 将新模型按 Triton 规范放入模型仓库（与 `yolo11_TensorRT` 同级或同仓库目录）
2. 重启 Triton 容器，或使用 Triton 模型控制 API 加载新模型
3. 在技能 `DEFAULT_CONFIG["required_models"]` 中声明模型名
4. 注册技能并在推流请求中指定 `skill_name`

## 流媒体服务（ZLMediaKit）

本项目使用 [ZLMediaKit](https://github.com/ZLMediaKit/ZLMediaKit) 作为流媒体引擎。检测推流任务将标注后的视频经 FFmpeg 推送到 ZLMediaKit，再由 ZLMediaKit 提供 RTMP、RTSP、HTTP-FLV、HLS 等多种协议的播放能力。

### Docker 部署

```bash
docker run -id \
  -p 1935:1935 \
  -p 8080:80 \
  -p 8443:443 \
  -p 8554:554 \
  -p 10000:10000 \
  -p 10000:10000/udp \
  -p 8000:8000/udp \
  -p 9000:9000/udp \
  zlmediakit/zlmediakit:master
```

### 端口说明


| 端口      | 协议      | 用途                        |
| ------- | ------- | ------------------------- |
| `1935`  | TCP     | RTMP 推流/拉流                |
| `8080`  | TCP     | HTTP 服务（含 HTTP-FLV、HLS 等） |
| `8443`  | TCP     | HTTPS                     |
| `8554`  | TCP     | RTSP                      |
| `10000` | TCP/UDP | WebRTC 等                  |
| `8000`  | UDP     | RTP                       |
| `9000`  | UDP     | SRT                       |


### 推流地址格式

推送到 ZLMediaKit 时，`out_url` 默认采用如下格式（也可在请求中显式指定完整地址）：

```
rtmp://<ZLMediaKit主机>:1935/<scene_id>/<skill_name>
```

示例（使用 config 默认值）：

```
rtmp://10.1.3.21:1935/scene_001/person_count_detector
```

- `scene_001`：场景 ID（`scene_id`）
- `person_count_detector`：技能名称（`skill_name`）

推流成功后，常用播放地址：

| 协议 | 地址格式 |
| ---- | -------- |
| RTMP | `rtmp://<主机>:1935/<app>/<stream>` |
| RTSP | `rtsp://<主机>:8554/<app>/<stream>` |
| HTTP-FLV | `http://<主机>:8080/<app>/<stream>.live.flv` |
| HLS | `http://<主机>:8080/<app>/<stream>/hls.m3u8` |
| 事件录像 MP4 | `http://<主机>:8080/record/<app>/<stream>/events/<文件名>.mp4` |

更多说明见 [ZLMediaKit 文档](https://github.com/ZLMediaKit/ZLMediaKit)。

### HTTP API 鉴权（事件截取必配）

调用 ZLMediaKit HTTP API（如 `getMediaList`、`startRecordTask`）需携带 `secret`。请在本服务 `.env` 中配置 `ZLM_SECRET`，与 MediaServer `config.ini` 中 `[api] secret` 保持一致。未配置时 API 可能返回 `Please login first`（`code: -100`）。

### 事件录像与 GOP 缓存（设计说明）

本服务的事件截取底层调用 ZLMediaKit `startRecordTask`，时长由两个参数共同决定：

| 参数 | 含义 | 数据来源 |
| ---- | ---- | -------- |
| `back_ms` | 回溯录制（当前时刻之前） | 只能取自内存中的 **GOP 缓存**，缓存没有的画面无法补录 |
| `forward_ms` | 前向录制（当前时刻之后） | 从调用时刻起继续收流并写入，一般能录满 |

因此：

```text
实际录像时长 ≈ min(回溯时长, GOP 缓存中已有历史) + forward_ms
```

`back_ms + forward_ms` 是**请求上限**，不是保证值。常见现象：请求 `back_ms=10000`、`forward_ms=10000`（期望约 20s），实际只有约 10s——多半是回溯几乎没录上，只录了前向 10s。

#### 为何 `config.ini` 里 `gop_cache` 太小会导致回溯不足

ZLMediaKit 配置文件（`config.ini`）中与事件回溯相关的关键项大致为：

```ini
# startSendRtp、startRecord 等相关功能是否提前开启 GOP 缓存
# 默认开启，并缓存 1 个 GOP；若需回溯更长时间，应加大该参数
gop_cache=1
```

设计要点：

1. **默认 `gop_cache=1`**：通常只缓存约 **1 个 GOP**（从一个关键帧到下一个关键帧之间的数据）。
2. **关键帧间隔决定单 GOP 时长**：若关键帧间隔只有 **1～2 秒**，则默认缓存历史大约只有 1～2 秒，远达不到 `back_ms=10000`（10 秒）的回溯目标。
3. **`back_ms` 不能凭空造历史**：ZLM 不会从磁盘去“倒推”未缓存的画面；只能吐出当前仍留在 GOP 缓存里的帧。
4. **另有帧数上限**：GOP 缓存还受内部总帧数限制（量级约 1024 帧）。帧率很高时，即便 `gop_cache` 较大，过旧的帧也可能被挤掉。本项目默认推流约 10fps 时，10 秒约 100 帧，一般不是主因；优先检查 `gop_cache`。

粗略估算需要的 `gop_cache`：

```text
建议 gop_cache ≥ ceil(期望回溯秒数 / 单个 GOP 秒数)
```

示例：期望回溯 10 秒，关键帧间隔约 2 秒 → 至少 `gop_cache=5`（可略放大到 `8`～`10` 留余量）。

修改后需**重启 ZLMediaKit**，并让目标流再稳定推送一段时间（建议 ≥ 回溯时长，如 15～20 秒），再调用本服务的截取接口。

#### 推荐配置与验证

1. 在 ZLMediaKit `config.ini` 中增大 `gop_cache`（按上式估算），保存并重启 MediaServer。  
2. 启动 AI 推流任务，等待缓存攒够后再截取（`test_clip_client.py demo` 中的 warmup 即为此设计）。  
3. 对比验证：
   - `--back_ms 0 --forward_ms 10000` → 时长约 10s，说明前向正常  
   - `--back_ms 10000 --forward_ms 10000` → 若仍接近 10s，说明回溯仍不足，继续加大 `gop_cache` 或检查源流关键帧间隔  

## 快速开始

### 1. 启动 Triton Inference Server（若尚未部署）

参考上文 [AI 推理服务](#ai-推理服务triton-inference-server) 启动 Triton 容器，确认 `yolo11_person` 模型已加载，且 `8201` 端口（gRPC）可达。在 `code/.env` 中配置：

```env
TRITON_URL=<Triton主机IP>:8201
```

### 2. 启动 ZLMediaKit（若尚未部署）

参考上文 [流媒体服务](#流媒体服务zlmediakit) 启动流媒体服务，确保 `1935` / `8080` 端口可达，并配置好 `ZLM_SECRET`。

### 3. 启动 API 服务

```bash
cd code
python -m app.main
```

或使用 uvicorn：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

启动后访问：

- Swagger UI：[http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc：[http://localhost:8000/redoc](http://localhost:8000/redoc)
- 健康检查：[http://localhost:8000/health](http://localhost:8000/health)

### 4. 通过 API 客户端启动 AI 视频识别任务

```bash
# 查看可用技能
python ./app/client_scripts/test_stream_client.py skills

# 本地文件推流
python ./app/client_scripts/test_stream_client.py start \
  --in_url ./test_images_videos/person_gouzi.mp4 \
  --scene_id scene_001 \
  --skill_name person_count_detector26

# 从已有直播源拉流并推检测结果
python ./app/client_scripts/test_stream_client.py start \
  --in_url rtmp://10.1.3.21:1935/live/stream01 \
  --scene_id scene_001 \
  --skill_name person_count_detector26

# 查询 / 停止 / 列表
python ./app/client_scripts/test_stream_client.py status --task_id <task_id>
python ./app/client_scripts/test_stream_client.py stop --task_id <task_id>
python ./app/client_scripts/test_stream_client.py list

# 演示模式（启动后轮询状态，Ctrl+C 自动停止）
python ./app/client_scripts/test_stream_client.py demo
```

启动成功后响应包含：

- `out_url`：RTMP/RTSP 推流地址  
- `flv_url`：对应 HTTP-FLV 播放地址（由 `out_url` 推导）

### 5. 事件视频截取

需目标流已在线（可先完成步骤 4），且已配置 `ZLM_SECRET`：

```bash
# 推流稳定约 15s 后再截取，便于积累 GOP 缓存（回溯）
python ./app/client_scripts/test_clip_client.py capture \
  --video_url rtmp://10.1.3.21:1935/scene_001/person_count_detector26 \
  --back_ms 10000 \
  --forward_ms 10000

# 一键：启动推流 → 等待 → 截取 → 停止
python ./app/client_scripts/test_clip_client.py demo
```

请求仅需三个业务参数：`video_url`、`back_ms`、`forward_ms`。服务端解析流地址中的 `app`/`stream`，调用 ZLMediaKit 事件录像，同步返回 MP4 / 截图 Base64 及 `video_url`（录像 HTTP 地址）。

### 6. 命令行直连推流（不经 API）

适合本地调试，不经过 FastAPI 和子进程管理：

```bash
python ./app/client_scripts/test_stream_process.py \
  --in_url ./test_images_videos/person_gouzi.mp4 \
  --scene_id scene_001 \
  --skill_name person_count_detector
```

## 自动化测试（auto_unit_test）

`app/auto_unit_test/run_tests.py` 将常用手工验证命令封装为一条脚本，按顺序执行并输出 `PASS / FAIL / SKIP` 汇总。

### 运行

在 `code/` 目录下：

```bash
python app/auto_unit_test/run_tests.py
```

PowerShell：

```powershell
python .\app\auto_unit_test\run_tests.py
```

仅运行部分测试：

```bash
python app/auto_unit_test/run_tests.py --only skill
python app/auto_unit_test/run_tests.py --only server
python app/auto_unit_test/run_tests.py --only client
python app/auto_unit_test/run_tests.py --only stream
```

可多次指定 `--only`，例如 `--only server --only client`。

### 测试项与等价手工命令


| 序号  | 自动化测试  | 等价手工命令                                                                                                                                                                  |
| --- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 技能测试   | `python app/plugins/skills/person_count_detector_skill.py`                                                                                                              |
| 2   | 服务端测试  | 启动 `python -m app.main`，检查 `GET /health`                                                                                                                                |
| 3   | 客户端测试  | `python app/client_scripts/test_stream_client.py start --in_url ./test_images_videos/person_gouzi.mp4 --scene_id scene_001 --skill_name person_count_detector`（并自动 stop） |
| 4   | 本地推流测试 | `python app/client_scripts/test_stream_process.py`（运行约 8 秒后自动停止）                                                                                                        |


### 依赖与说明

- 技能 / 推流 / 客户端测试需要 **Triton** 可达，且存在测试资源：
  - `test_images_videos/person_gouzi.mp4`（视频流测试）
  - `test_images_videos/9-74-0004-000880.jpg`（技能图片测试）
- 本地推流与客户端测试默认加 `--no_hw_encode`，避免无 NVENC 环境失败。
- Triton 或测试文件不可用时，对应用例会标记为 **SKIP**，不会导致整批中断。
- 推流测试为短时运行后自动终止进程，无需手动 Ctrl+C。

### 输出示例

```
工作目录: C:\...\code
计划执行: skill, server, client, stream

>>> ...

============================================================
测试汇总
============================================================
[PASS] 技能测试 (person_count_detector_skill.py)
[PASS] 服务端测试 (app.main /health)
[PASS] 客户端测试 (test_stream_client.py start)
[PASS] 本地推流测试 (test_stream_process.py)
------------------------------------------------------------
通过: 4  失败: 0  跳过: 0  合计: 4
```

## API 接口

前缀：`/api/v1`（`/health` 除外）。完整交互文档见 `/docs`。


| 方法     | 路径                        | 说明                         |
| ------ | ------------------------- | -------------------------- |
| `GET`  | `/health`                 | 服务健康检查                     |
| `GET`  | `/skills`                 | 列出可用技能                     |
| `POST` | `/streams/start`          | 启动 AI 视频识别任务               |
| `POST` | `/streams/stop/{task_id}` | 停止 AI 视频识别任务               |
| `GET`  | `/streams`                | 查询所有 AI 视频识别任务             |
| `GET`  | `/streams/{task_id}`      | 查询单个任务状态                   |
| `POST` | `/clip/capture`           | 截取事件视频并返回中间帧截图（Base64）     |


### 启动推流请求示例

`POST /api/v1/streams/start`

```json
{
  "in_url": "rtmp://10.1.3.21:1935/live/stream01",
  "out_fps": 10,
  "alarm_interval": 1,
  "scene_id": "scene_001",
  "skill_name": "person_count_detector26",
  "output_format": "rtmp",
  "queue_size": 2,
  "use_hardware_encoding": true,
  "bitrate": "2M",
  "buffer_size": "2M",
  "reconnect_delay": 2.0,
  "fence_config": null,
  "skill_config": null
}
```

未传 `out_url` 时自动生成为：`rtmp://{ZLM_HOST}:1935/{scene_id}/{skill_name}`。

### 启动推流响应示例

```json
{
  "task_id": "a1b2c3d4-...",
  "status": "running",
  "scene_id": "scene_001",
  "skill_name": "person_count_detector26",
  "in_url": "rtmp://10.1.3.21:1935/live/stream01",
  "out_url": "rtmp://10.1.3.21:1935/scene_001/person_count_detector26",
  "flv_url": "http://10.1.3.21:8080/scene_001/person_count_detector26.live.flv",
  "pid": 12345,
  "created_at": "2026-07-14T11:38:45",
  "started_at": "2026-07-14T11:38:45",
  "message": "视频AI检测任务已启动"
}
```

### 事件视频截取请求示例

`POST /api/v1/clip/capture`

仅需三个参数（其余如 `app`/`stream`/`vhost` 由服务端从 `video_url` 解析）：

```json
{
  "video_url": "rtmp://10.1.3.21:1935/scene_001/person_count_detector26",
  "back_ms": 10000,
  "forward_ms": 10000
}
```

`video_url` 支持 RTMP / RTSP / HTTP-FLV / HLS 等路径中含 `app/stream` 的地址。

### 事件视频截取响应要点

| 字段 | 说明 |
| ---- | ---- |
| `video_url` | 录像 MP4 的 HTTP 访问地址（`/record/...`） |
| `video_base64` | 录像文件 Base64 |
| `image_base64` | 视频中间时刻 JPEG 截图 Base64 |
| `duration_sec` | 实际录像时长（秒，由 ffprobe 探测） |
| `back_ms` / `forward_ms` | 请求的回溯 / 前向时长 |

本接口为**同步阻塞**，耗时大约为 `forward_ms + 写入等待`（默认额外约 2s）。

## 内置技能


| skill_name                 | 中文名              | 模型              | 说明                 |
| -------------------------- | ---------------- | --------------- | ------------------ |
| `person_count_detector`    | 人流量检测            | `yolo11_person` | 人数统计，超上限触发告警       |
| `person_count_detector26`  | 人流量检测(YOLO26)    | `yolo26_person` | 基于 YOLO26 的人数检测技能 |

技能默认参数（可通过 `skill_config` 覆盖）：

```json
{
  "params": {
    "conf_thres": 0.5,
    "iou_thres": 0.45,
    "max_det": 300,
    "input_size": [640, 640],
    "enable_default_sort_tracking": true,
    "tracking_algorithm": "sort",
    "person_limit": 6
  }
}
```

跟踪算法（`tracking_algorithm`）可选：


| 值            | 说明                                           | 实现位置                                     |
| ------------ | -------------------------------------------- | ---------------------------------------- |
| `sort`       | SORT 算法（默认）                                  | `app/services/trackers/sort.py`          |
| `bytetrack`  | ByteTrack                                    | `app/services/trackers/byte_tracker.py`  |
| `botsort`    | BoT-SORT（含 GMC 相机运动补偿）                       | `app/services/trackers/bot_sort.py`      |
| `ocsort`     | OC-SORT（观测中心关联：ORU / OCM / OCR）              | `app/services/trackers/oc_sort.py`       |
| `fasttrack`  | FastTracker（遮挡感知 ByteTrack 扩展）               | `app/services/trackers/fast_tracker.py`  |
| `deepocsort` | Deep OC-SORT（OC-SORT + GMC + ReID，ReID 默认关闭） | `app/services/trackers/deep_oc_sort.py`  |
| `tracktrack` | TrackTrack（多 cue 代价 + 迭代匹配）                  | `app/services/trackers/track_tracker.py` |


别名也支持，例如 `oc_sort`、`fast_tracker`、`deep_oc_sort`、`track_track`、`BoT-SORT` 等，详见 `TrackerService` 中的 `_ALGORITHM_ALIASES`。

关闭跟踪：设置 `"enable_default_sort_tracking": false`。

ByteTrack / BoT-SORT / OC-SORT / FastTracker / Deep OC-SORT / TrackTrack 通用参数（可选）：

```json
{
  "tracking_frame_rate": 30,
  "track_high_thresh": 0.5,
  "track_low_thresh": 0.1,
  "track_buffer": 30,
  "match_thresh": 0.8,
  "new_track_thresh": 0.6,
  "fuse_score": true
}
```

BoT-SORT / Deep OC-SORT / TrackTrack 相机运动补偿与 ReID（可选）：

```json
{
  "gmc_method": "sparseOptFlow",
  "proximity_thresh": 0.5,
  "appearance_thresh": 0.8,
  "with_reid": false,
  "reid_model": "auto"
}
```

OC-SORT 额外参数（可选）：

```json
{
  "delta_t": 3,
  "inertia": 0.2,
  "use_byte": false
}
```

FastTracker 遮挡处理参数（可选）：

```json
{
  "reset_velocity_offset_occ": 5,
  "reset_pos_offset_occ": 3,
  "enlarge_bbox_occ": 1.1,
  "dampen_motion_occ": 0.5,
  "active_occ_to_lost_thresh": 10,
  "occ_cover_thresh": 0.7,
  "occ_reappear_window": 40,
  "init_iou_suppress": 0.7
}
```

Deep OC-SORT 外观特征参数（可选，`with_reid: true` 时生效）：

```json
{
  "alpha_fixed_emb": 0.95
}
```

TrackTrack 额外参数（可选）：

```json
{
  "lost_match_thr": 0.0,
  "penalty_p": 0.2,
  "penalty_q": 0.4,
  "reduce_step": 0.05,
  "iou_weight": 0.5,
  "reid_weight": 0.5,
  "conf_weight": 0.1,
  "angle_weight": 0.05,
  "tai_thr": 0.55,
  "min_track_len": 3
}
```

SORT 额外参数（可选）：

```json
{
  "tracking_max_age": 30,
  "tracking_min_hits": 1,
  "tracking_iou_threshold": 0.3
}
```

轨迹历史（可选，用于技能画图绘制运动轨迹）：

```json
{
  "enable_track_history": true,
  "tracking_history_frames": 20
}
```

- `tracking_history_frames`：每个 track_id 保留最近 N 帧的框中心点坐标（如 10、20）
- 检测结果中会附加 `track_history` 字段：`[[cx, cy], ...]`
- 技能画图时可调用 `BaseSkill.draw_track_trajectories()` 绘制轨迹线

## 告警

当检测到人数超过 `person_limit` 且距上次告警超过 `alarm_interval` 秒时，触发告警回调。

默认告警处理（`app/services/stream_alert.py`）会输出日志：

```
触发人数告警 scene=scene_001 count=3 pic=person_count/scene_001_....jpg shape=(1080, 1920, 3)
```

可在 `stream_worker.py` 中替换 `default_alert_handler`，扩展为写库、上传对象存储、推送消息等。

## 扩展新技能

1. 在 Triton 中部署并加载对应模型（见 [扩展新模型](#扩展新模型)）
2. 在 `app/plugins/skills/` 下新建技能类，继承 `BaseSkill`
3. 定义 `DEFAULT_CONFIG`（含 `name`、`required_models` 等）
4. 实现 `_initialize()` 和 `process(frame, fence_config)` 方法
5. 在 `app/plugins/skill_registry.py` 的 `_SKILL_CLASSES` 中注册

```python
_SKILL_CLASSES = {
    PersonCountDetectorSkill.DEFAULT_CONFIG["name"]: PersonCountDetectorSkill,
    "my_new_skill": MyNewSkill,
}
```

启动推流时传入 `"skill_name": "my_new_skill"` 即可。

## 视频处理管道说明

`VideoStreamPipeline` 是核心编排类，数据流如下：

1. **FrameReader**：OpenCV 读取视频源，支持降帧、断线重连
2. **AsyncFrameProcessor**（三线程）：
  - 检测线程：调用 `skill.process()`，经 Triton 推理并画框
  - 推流线程：将标注帧送入 FFmpeg
  - 告警线程：按间隔检查并触发 `on_alert` 回调
3. **FFmpegStreamer**：将原始 BGR 帧编码后推送到 ZLMediaKit（RTMP/RTSP）

## 常见问题

**无法连接 Triton**

- 确认 Triton 容器已启动，端口映射为 `8201:8001`（gRPC）
- 确认 `TRITON_URL` 指向正确的主机 IP 与 `8201` 端口
- 确认所用技能对应模型（如 `yolo11_person` / `yolo26_person`）已在 Triton 中加载
- 检查 GPU 驱动与 `nvidia-container-toolkit` 是否正常
- 可用 `curl http://<主机>:8200/v2/health/ready` 检查 Triton HTTP 健康状态

**RTMP 推流失败或播放器无法播放**

- 确认 ZLMediaKit 已启动，Docker 容器端口映射正确（尤其 `1935`、`8080`）
- 确认 FFmpeg 已安装且在 `PATH` 中
- 确认 `out_url` 格式正确：`rtmp://<主机>:1935/<scene_id>/<skill_name>`
- 无 NVIDIA GPU 时加 `--no_hw_encode` 使用软件编码
- 网页播放优先使用响应中的 `flv_url`（HTTP-FLV），或 HLS / RTSP 地址

**ZLMediaKit 返回 `Please login first`（code: -100）**

- 在 `.env` 中配置 `ZLM_SECRET`，与 MediaServer `config.ini` 的 `[api] secret` 一致
- 修改后重启本 API 服务

**事件截取时长明显短于 `back_ms + forward_ms`**

- `back_ms` 依赖 ZLM **GOP 缓存**；缓存不足时回溯不到满额度
- 在 `config.ini` 中增大 `gop_cache`，并在推流稳定后再截取
- 可用 `--back_ms 0 --forward_ms 10000` 对比：若约 10s 说明后向正常、问题在回溯

**`ModuleNotFoundError: No module named 'app'`**

- 请在 `code/` 目录下运行，或使用 `python -m app.main`
- 客户端脚本路径为 `app/client_scripts/`，不要从其他目录直接 `python xxx.py` 除非已设置 `PYTHONPATH`

**任务状态为 `failed`**

- 查看 API 服务与子进程日志
- 检查输入视频源 `in_url` 是否可访问
- 检查 `skill_name` 是否在注册表中（`GET /api/v1/skills`）

## 许可证

内部项目，按需使用。