import numpy as np
from typing import Dict, Type, Optional, Any
from abc import ABC, abstractmethod
import random
from functools import wraps


class PolicyRegistry:
    """
    Policy注册器，用于管理所有的Policy类
    """
    _policies: Dict[str, Type['Policy']] = {}
    
    @classmethod
    def register(cls, name: Optional[str] = None):
        """
        装饰器，用于注册Policy
        
        Args:
            name: Policy的名称，如果不提供则使用类的name属性
        
        Usage:
            @PolicyRegistry.register("my_policy")
            class MyPolicy(Policy):
                ...
        """
        def decorator(policy_cls: Type['Policy']) -> Type['Policy']:
            policy_name = name or policy_cls.name.fget(policy_cls) if hasattr(policy_cls.name, 'fget') else policy_cls.__name__.lower()
            
            if policy_name in cls._policies:
                raise ValueError(f"Policy '{policy_name}' is already registered")
            
            cls._policies[policy_name] = policy_cls
            return policy_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str, **kwargs) -> 'Policy':
        """
        获取并实例化一个Policy
        
        Args:
            name: Policy的名称
            **kwargs: 传递给Policy构造函数的参数
        
        Returns:
            Policy实例
        """
        if name not in cls._policies:
            raise ValueError(f"Policy '{name}' is not registered. Available policies: {list(cls._policies.keys())}")
        
        policy_cls = cls._policies[name]
        return policy_cls(**kwargs)
    
    @classmethod
    def list_policies(cls) -> list:
        """
        列出所有已注册的Policy名称
        """
        return list(cls._policies.keys())
    
    @classmethod
    def get_policy_class(cls, name: str) -> Type['Policy']:
        """
        获取Policy类（不实例化）
        """
        if name not in cls._policies:
            raise ValueError(f"Policy '{name}' is not registered")
        return cls._policies[name]
    
    @classmethod
    def clear(cls):
        """
        清空注册器（主要用于测试）
        """
        cls._policies.clear()


class Policy(ABC):
    """
    基础Policy抽象类
    """
    def __init__(self, model=None):
        self.model = model
    
    def reset_instruction(self, instruction):
        """
        重置Policy的指令
        """
        self.instruction = instruction
    
    def predict(self, obs, **kwargs):
        """
        预测动作
        
        Args:
            obs: 观察值
            **kwargs: 额外参数
        
        Returns:
            动作
        """
        pass
    
    def _process_observation(self, obs, **kwargs):
        """
        处理观察值,对齐到Policy输入格式
        """
        pass
    
    def _process_action(self, output):
        """
        处理输出，对齐到动作格式
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """
        Policy名称
        """
        pass
    
    @property
    def control_mode(self):
        """
        控制模式，如 "ee" 或 "joint"
        """
        return "ee"