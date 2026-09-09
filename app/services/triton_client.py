import tritonclient.grpc as grpcclient
import logging
import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from app.core.config import settings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TritonClient:
    """
    Triton推理服务器通用客户端
    支持服务器健康检查、模型信息查询和推理调用
    包含自动重连和懒加载机制
    """
    def __init__(self, url=None, verbose=False):
        """
        初始化Triton客户端
        
        参数:
            url (str): Triton服务器的URL地址，格式为"host:port"
            verbose (bool): 是否启用详细日志
        """
        if url is None:
            url = settings.TRITON_URL
        self.url = url
        self.verbose = verbose
        self._client = None  # 懒加载，不在初始化时连接
        self._last_connection_attempt = 0
        self._connection_retry_interval = 5  # 重连间隔（秒）
        self._live_check_interval = 30.0  # 连接存活探测间隔（秒）
        self._last_live_check = 0.0
        self._metadata_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        logger.info(f"初始化Triton客户端配置，目标服务器: {url}")

    @staticmethod
    def _metadata_cache_key(model_name: str, model_version: str = "") -> Tuple[str, str]:
        return model_name, model_version or ""

    def clear_metadata_cache(
        self,
        model_name: Optional[str] = None,
        model_version: str = "",
    ) -> None:
        """清除模型 metadata 缓存；不传 model_name 时清空全部。"""
        if model_name is None:
            self._metadata_cache.clear()
            return
        self._metadata_cache.pop(self._metadata_cache_key(model_name, model_version), None)

    def warm_metadata_cache(self, model_name: str, model_version: str = "") -> bool:
        """预拉取并缓存模型 metadata（启动阶段调用）。"""
        return self.get_model_metadata(model_name, model_version, use_cache=True) is not None
    
    def _get_client(self):
        """
        懒加载获取Triton客户端连接
        包含自动重连机制
        """
        current_time = time.time()
        
        # 如果客户端存在且连接正常，直接返回
        if self._client is not None:
            if current_time - self._last_live_check < self._live_check_interval:
                return self._client
            try:
                # 周期性探测连接是否有效，避免每帧推理都发起 RPC
                self._client.is_server_live()
                self._last_live_check = current_time
                return self._client
            except Exception:
                logger.warning("检测到Triton连接失效，准备重新连接")
                self._client = None
                self.clear_metadata_cache()
        
        # 限制重连频率
        if current_time - self._last_connection_attempt < self._connection_retry_interval:
            raise Exception(f"Triton服务连接失败，请等待{self._connection_retry_interval}秒后重试")
        
        # 尝试创建新连接
        self._last_connection_attempt = current_time
        try:
            logger.info(f"尝试连接Triton服务器: {self.url}")
            self._client = grpcclient.InferenceServerClient(url=self.url, verbose=self.verbose)
            
            # 测试连接
            if self._client.is_server_live():
                logger.info(f"成功连接到Triton服务器: {self.url}")
                self._last_live_check = current_time
                return self._client
            else:
                logger.error("Triton服务器无响应")
                self._client = None
                raise Exception("Triton服务器无响应")
                
        except Exception as e:
            logger.error(f"连接Triton服务器失败: {str(e)}")
            self._client = None
            raise Exception(f"无法连接到Triton服务器 {self.url}: {str(e)}")
    
    @property
    def client(self):
        """获取客户端连接（自动重连）"""
        return self._get_client()
    
    def is_connected(self) -> bool:
        """
        检查当前是否连接到Triton服务器
        """
        try:
            return self._client is not None and self._client.is_server_live()
        except Exception:
            return False
    
    def reconnect(self) -> bool:
        """
        强制重新连接到Triton服务器
        """
        try:
            self._client = None
            self._last_connection_attempt = 0  # 重置重连时间限制
            self._last_live_check = 0.0
            self.clear_metadata_cache()
            self._get_client()
            return True
        except Exception as e:
            logger.error(f"强制重连失败: {str(e)}")
            return False
    
    def is_server_live(self) -> bool:
        """
        检查Triton服务器是否在线
        
        返回:
            bool: 服务器是否在线
        """
        try:
            return self.client.is_server_live()
        except grpcclient.InferenceServerException as e:
            logger.error(f"检查服务器状态失败: {e}")
            return False
    
    def is_server_ready(self) -> bool:
        """
        检查Triton服务器是否就绪
        
        返回:
            bool: 服务器是否就绪
        """
        try:
            return self.client.is_server_ready()
        except grpcclient.InferenceServerException as e:
            logger.error(f"检查服务器就绪状态失败: {e}")
            return False
    
    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        """
        检查指定模型是否就绪
        
        参数:
            model_name (str): 模型名称
            model_version (str): 模型版本，默认为空字符串（使用最新版本）
            
        返回:
            bool: 模型是否就绪
        """
        try:
            return self.client.is_model_ready(model_name, model_version)
        except grpcclient.InferenceServerException as e:
            logger.error(f"检查模型 {model_name} 就绪状态失败: {e}")
            return False
    
    def get_model_metadata(
        self,
        model_name: str,
        model_version: str = "",
        use_cache: bool = True,
    ) -> Optional[Dict]:
        """
        获取模型元数据（默认使用进程内缓存，避免每帧 RPC）
        
        参数:
            model_name (str): 模型名称
            model_version (str): 模型版本，默认为空字符串（使用最新版本）
            use_cache (bool): 是否使用/写入缓存
            
        返回:
            Dict: 模型元数据，如果获取失败则返回None
        """
        cache_key = self._metadata_cache_key(model_name, model_version)
        if use_cache and cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        try:
            metadata = self.client.get_model_metadata(model_name, model_version, as_json=True)
            if metadata and use_cache:
                self._metadata_cache[cache_key] = metadata
            return metadata
        except grpcclient.InferenceServerException as e:
            logger.error(f"获取模型 {model_name} 元数据失败: {e}")
            return None
    
    def get_model_config(self, model_name: str, model_version: str = "") -> Optional[Dict]:
        """
        获取模型配置
        
        参数:
            model_name (str): 模型名称
            model_version (str): 模型版本，默认为空字符串（使用最新版本）
            
        返回:
            Dict: 模型配置，如果获取失败则返回None
        """
        try:
            config = self.client.get_model_config(model_name, model_version, as_json=True)
            return config
        except grpcclient.InferenceServerException as e:
            logger.error(f"获取模型 {model_name} 配置失败: {e}")
            return None
    
    def get_server_metadata(self) -> Optional[Dict]:
        """
        获取服务器元数据
        
        返回:
            Dict: 服务器元数据，如果获取失败则返回None
        """
        try:
            metadata = self.client.get_server_metadata(as_json=True)
            return metadata
        except grpcclient.InferenceServerException as e:
            logger.error(f"获取服务器元数据失败: {e}")
            return None
    
    def get_model_repository_index(self) -> Optional[Dict]:
        """
        获取模型仓库索引
        
        返回:
            Dict: 模型仓库索引，如果获取失败则返回None
        """
        try:
            index = self.client.get_model_repository_index(as_json=True)
            return index
        except grpcclient.InferenceServerException as e:
            logger.error(f"获取模型仓库索引失败: {e}")
            return None
    
    def load_model(self, model_name: str, config: Optional[str] = None, 
                  files: Optional[Dict[str, bytes]] = None) -> bool:
        """
        请求Triton服务器加载或重新加载指定模型
        
        参数:
            model_name (str): 要加载的模型名称
            config (Optional[str]): 可选的模型配置JSON字符串，如果提供，将使用此配置加载模型
            files (Optional[Dict[str, bytes]]): 可选的文件字典，指定加载模型的文件内容
            
        返回:
            bool: 模型是否成功加载
        """
        try:
            self.client.load_model(model_name, config=config, files=files)
            logger.info(f"模型 {model_name} 加载请求已发送")
            
            # 验证模型是否成功加载
            if self.is_model_ready(model_name):
                logger.info(f"模型 {model_name} 已成功加载并就绪")
                self.clear_metadata_cache(model_name)
                return True
            else:
                logger.warning(f"模型 {model_name} 加载请求已发送，但模型尚未就绪")
                return False
        except grpcclient.InferenceServerException as e:
            logger.error(f"加载模型 {model_name} 失败: {e}")
            return False
    
    def unload_model(self, model_name: str, unload_dependents: bool = False) -> bool:
        """
        请求Triton服务器卸载指定模型
        
        参数:
            model_name (str): 要卸载的模型名称
            unload_dependents (bool): 是否同时卸载依赖的模型，默认为False
            
        返回:
            bool: 模型是否成功卸载
        """
        try:
            self.client.unload_model(model_name, unload_dependents=unload_dependents)
            logger.info(f"模型 {model_name} 卸载请求已发送")
            
            # 验证模型是否成功卸载
            if not self.is_model_ready(model_name):
                logger.info(f"模型 {model_name} 已成功卸载")
                self.clear_metadata_cache(model_name)
                return True
            else:
                logger.warning(f"模型 {model_name} 卸载请求已发送，但模型仍然就绪")
                return False
        except grpcclient.InferenceServerException as e:
            logger.error(f"卸载模型 {model_name} 失败: {e}")
            return False
    
    def infer(self, model_name: str, inputs: Dict[str, np.ndarray], 
              model_version: str = "", request_id: str = "", 
              timeout: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        执行同步推理
        
        参数:
            model_name (str): 模型名称
            inputs (Dict[str, np.ndarray]): 模型输入，键为输入名称，值为输入数据
            model_version (str): 模型版本，默认为空字符串（使用最新版本）
            request_id (str): 请求ID，用于追踪请求
            timeout (Optional[int]): 超时时间，单位为毫秒
            
        返回:
            Dict[str, np.ndarray]: 推理结果，键为输出名称，值为输出数据，如果推理失败则返回None
        """
        try:
            metadata = self.get_model_metadata(model_name, model_version, use_cache=True)
            if not metadata:
                logger.error(f"获取模型 {model_name} 元数据失败，无法执行推理")
                return None
            
            # 创建输入对象
            infer_inputs = []
            for input_name, input_data in inputs.items():
                # 查找匹配的输入元数据
                input_metadata = next((inp for inp in metadata['inputs'] if inp['name'] == input_name), None)
                if not input_metadata:
                    logger.error(f"模型 {model_name} 没有名为 {input_name} 的输入")
                    return None
                
                # 创建输入对象
                infer_input = grpcclient.InferInput(
                    input_name, 
                    input_data.shape, 
                    input_metadata['datatype']
                )
                infer_input.set_data_from_numpy(input_data)
                infer_inputs.append(infer_input)
            
            # 创建输出对象
            infer_outputs = []
            for output_metadata in metadata['outputs']:
                infer_output = grpcclient.InferRequestedOutput(output_metadata['name'])
                infer_outputs.append(infer_output)
            
            # 执行推理
            result = self.client.infer(
                model_name=model_name,
                inputs=infer_inputs,
                outputs=infer_outputs,
                model_version=model_version,
                request_id=request_id,
                timeout=timeout
            )
            
            # 处理结果
            outputs = {}
            for output_metadata in metadata['outputs']:
                output_name = output_metadata['name']
                outputs[output_name] = result.as_numpy(output_name)
            
            return outputs
            
        except grpcclient.InferenceServerException as e:
            logger.error(f"推理失败: {e}")
            return None
        except Exception as e:
            logger.error(f"推理过程中发生错误: {e}")
            return None
    
    # 移除了原来的async_infer方法，可以根据需要添加

# 全局单例
triton_client = TritonClient()

# 测试代码
if __name__ == "__main__":
    # 测试服务器连接
    client = TritonClient()
    print(f"服务器是否在线: {client.is_server_live()}")
    print(f"服务器是否就绪: {client.is_server_ready()}")
    
    # 获取可用模型
    models = client.get_model_repository_index()
    if models:
        print("可用模型:")
        for model in models.get("models", []):
            model_name = model.get("name")
            print(f"- {model_name}")
            print(f"  状态: {'就绪' if client.is_model_ready(model_name) else '未就绪'}") 