"""MinIO 对象存储客户端"""
from __future__ import annotations

import json
import logging
from io import BytesIO
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = logging.getLogger(__name__)


class MinioClientError(RuntimeError):
    """MinIO 操作失败"""


class MinioStorage:
    def __init__(self):
        self._client: Optional[Minio] = None

    @property
    def client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ROOT_USER,
                secret_key=settings.MINIO_ROOT_PASSWORD,
                secure=settings.MINIO_SECURE,
            )
        return self._client

    def ensure_bucket(self, bucket: Optional[str] = None) -> str:
        """若桶不存在则自动创建；并按配置设置匿名可读策略。"""
        bucket_name = bucket or settings.MINIO_BUCKET_NAME
        try:
            if self.client.bucket_exists(bucket_name):
                logger.debug("MinIO 存储桶已存在: %s", bucket_name)
            else:
                self.client.make_bucket(bucket_name)
                logger.info("已自动创建 MinIO 存储桶: %s", bucket_name)

            if settings.MINIO_PUBLIC_READ:
                self._ensure_public_read_policy(bucket_name)
        except S3Error as e:
            raise MinioClientError(f"检查/创建存储桶失败: {e}") from e
        return bucket_name

    def _ensure_public_read_policy(self, bucket_name: str) -> None:
        """允许匿名 GetObject，使返回的 HTTP 直链可在浏览器直接访问。"""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }
        try:
            self.client.set_bucket_policy(bucket_name, json.dumps(policy))
            logger.info("已设置 MinIO 桶匿名可读策略: %s", bucket_name)
        except S3Error as e:
            raise MinioClientError(f"设置桶公开读策略失败: {e}") from e

    def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        content_type: str,
        bucket: Optional[str] = None,
    ) -> str:
        """
        上传字节到 MinIO，返回可访问 HTTP URL。
        """
        if not data:
            raise MinioClientError("上传内容为空")

        bucket_name = self.ensure_bucket(bucket)
        key = object_name.lstrip("/")
        try:
            self.client.put_object(
                bucket_name,
                key,
                BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
        except S3Error as e:
            raise MinioClientError(f"上传对象失败 ({key}): {e}") from e

        url = settings.build_minio_object_url(key, bucket_name)
        logger.info("已上传至 MinIO bucket=%s object=%s url=%s", bucket_name, key, url)
        return url


minio_storage = MinioStorage()
