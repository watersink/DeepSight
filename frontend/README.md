# 前端开发与一体演示

## 开发（前后端分离）

```bash
# 终端1：后端
pip install -r requirements.txt
# 确保 MySQL 已建库: CREATE DATABASE yingjiting DEFAULT CHARSET utf8mb4;
python -m app.main

# 终端2：前端
cd frontend
npm install
npm run dev
```

浏览器打开 http://127.0.0.1:5173 （Vite 代理 /api → :8000）

## 一体演示

```bash
cd frontend
npm install
npm run build
cd ..
python -m app.main
```

打开 http://127.0.0.1:8000/ 即可使用管理台（同时提供 /docs API）。

## 环境变量（.env）

```
MYSQL_HOST=10.1.3.21
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=root
MYSQL_DB=yingjiting
REDIS_HOST=10.1.3.21
REDIS_PORT=6379
REDIS_DB=1
```
