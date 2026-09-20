# DeepSight Docker 一键部署

按机器角色拆成两套 Compose（参考 [Dify docker](https://github.com/langgenius/dify/tree/main/docker) 的「目录 + `.env` + volumes」思路）：

| 目录 | 机器 | 服务 |
|------|------|------|
| [`cpu/`](./cpu/) | **CPU 机** | `mysql` `redis` `rabbitmq` `minio` `zlm` `api`（含前端） |
| [`gpu/`](./gpu/) | **GPU 机** | `triton` `worker` |

无 nginx：管理台直连 `http://CPU:8000`，播流直连 `http://CPU:8080`。

```text
摄像头 / 文件
    → GPU worker 解码 → Triton 推理 → FFmpeg
    → 推 rtmp://CPU:1935/...
    → ZLM 分发；截图/录像进 MinIO；告警可走 RabbitMQ / MySQL
管理台
    → CPU api → HTTP 调 GPU:8100 启停任务
```

---

## 前置条件

**CPU 机**

- Docker / Docker Compose v2
- 开放端口：`8000`（API）、`8080/1935/8554`（ZLM）、`9000/9001`（MinIO）、按需 `3306/6379/5672/15672`

**GPU 机**

- Docker / Compose v2
- NVIDIA Driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 模型按 Triton 规范放到 `gpu/volumes/triton/models/`（或 `.env` 里改 `TRITON_MODELS_HOST_PATH`）
- 开放：`8100`（Worker，给 CPU API 调）、可选 `8200/8201`（Triton 调试）

两机 `WORKER_TOKEN`、`ZLM_SECRET`、中间件账号必须一致。

构建镜像默认使用**阿里云 Debian / PyPI / npm 镜像**（避免国内访问 `deb.debian.org` 失败）。可在对应 `.env` 中覆盖：

```env
DEBIAN_MIRROR=https://mirrors.aliyun.com
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
PIP_TRUSTED_HOST=mirrors.aliyun.com
# CPU 前端构建可选
NPM_REGISTRY=https://registry.npmmirror.com
```

---

## 1. 启动 CPU 机

```bash
cd docker/cpu
cp .env.example .env
# 必改: PUBLIC_HOST、GPU_HOST、密码
docker compose up -d --build
```

访问：

- 管理台 / API：`http://<PUBLIC_HOST>:8000`
- MinIO Console：`http://<PUBLIC_HOST>:9001`
- RabbitMQ 管理台：`http://<PUBLIC_HOST>:15672`
- ZLM HTTP-FLV：`http://<PUBLIC_HOST>:8080/<app>/<stream>.live.flv`

持久化目录：`docker/cpu/volumes/{mysql,redis,rabbitmq,minio,zlm}/`

---

## 2. 启动 GPU 机

```bash
cd docker/gpu
cp .env.example .env
# 必改: CPU_HOST（= CPU 的 PUBLIC_HOST）、WORKER_TOKEN；放入模型后启动
docker compose up -d --build
```

模型示例结构：

```text
volumes/triton/models/
  yolo26_person/
    1/
      model.plan   # 或其它 Triton 后端文件
    config.pbtxt
```

Triton 使用 `--model-control-mode=explicit`，管理台「模型管理」的加载/卸载才可用。

---

## 3. 联通检查

1. CPU 管理台 → Worker 节点：应能看到 GPU（`WORKER_NODES` 配置的 url）
2. 新建任务，选择该 Worker + 技能，启动推流
3. 播放 `flv_url`；告警图应出现在 MinIO bucket

防火墙：

| 方向 | 端口 |
|------|------|
| 用户 → CPU | 8000, 8080, 1935, 9000 |
| CPU API → GPU | **8100** |
| GPU Worker → CPU | 1935, 8080, 9000, 5672, 3306（告警落库） |

---

## 环境变量（IP 只改 `.env`）

| 机器 | 文件 | 必改项 |
|------|------|--------|
| CPU | `docker/cpu/.env` | `PUBLIC_HOST`（本机对外 IP）、`GPU_HOST`（GPU 机 IP） |
| GPU | `docker/gpu/.env` | `CPU_HOST`（须等于 CPU 的 `PUBLIC_HOST`）、`WORKER_TOKEN` |

`docker-compose.yml` **不写死局域网 IP**；Compose 启动时自动读取同目录 `.env`。  
改完 `.env` 后执行 `docker compose up -d`（需要时加 `--force-recreate`）即可生效。

多 GPU 时在 CPU 的 `.env` 覆盖整行，例如：

```env
WORKER_NODES=gpu1|GPU-1|http://10.1.3.21:8100;gpu2|GPU-2|http://10.1.3.22:8100
```

**改 ZLM secret**：同时改 `cpu/conf/zlm/config.ini` 的 `[api] secret` 与两边 `.env` 的 `ZLM_SECRET`。

---

## 常用命令

```bash
# CPU
cd docker/cpu && docker compose ps && docker compose logs -f api

# GPU
cd docker/gpu && docker compose ps && docker compose logs -f worker triton

# 停服务（数据保留在 volumes/）
docker compose down
```

---

## 镜像说明

| 镜像 | Dockerfile |
|------|-----------|
| `deepsight-api:local` | [`Dockerfile.api`](./Dockerfile.api)（多阶段：前端 build + FastAPI） |
| `deepsight-worker:local` | [`Dockerfile.worker`](./Dockerfile.worker)（含 FFmpeg） |
| Triton | `nvcr.io/nvidia/tritonserver:24.06-py3`（可在 gpu `.env` 改版本） |
