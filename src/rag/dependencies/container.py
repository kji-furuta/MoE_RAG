"""
依存性注入（DI）コンテナの実装

Phase 2: サービスの依存性注入とライフサイクル管理を提供する
DIコンテナを実装します。
"""

from typing import Dict, Any, Optional, Type, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect
import threading
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import asyncio
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ServiceScopeEnum(Enum):
    """サービスのスコープ"""
    SINGLETON = "singleton"      # アプリケーション全体で1インスタンス
    TRANSIENT = "transient"      # 要求ごとに新規インスタンス
    SCOPED = "scoped"           # スコープごとに1インスタンス


class ServiceLifecycle(Enum):
    """サービスのライフサイクル状態"""
    REGISTERED = "registered"    # 登録済み
    CREATING = "creating"       # 作成中
    CREATED = "created"         # 作成済み
    DISPOSING = "disposing"     # 破棄中
    DISPOSED = "disposed"       # 破棄済み


class ServiceNotFoundError(Exception):
    """サービスが見つからない場合のエラー"""
    pass


class CircularDependencyError(Exception):
    """循環依存が検出された場合のエラー"""
    pass


class ServiceCreationError(Exception):
    """サービスの作成に失敗した場合のエラー"""
    pass


@dataclass
class ServiceDescriptor:
    """サービスの記述子"""
    service_type: Type                           # サービスの型
    implementation: Union[Type, Callable, Any]   # 実装（クラス、ファクトリ、インスタンス）
    scope: ServiceScopeEnum = ServiceScopeEnum.SINGLETON # スコープ
    dependencies: Dict[str, str] = field(default_factory=dict)  # 依存関係
    metadata: Dict[str, Any] = field(default_factory=dict)      # メタデータ
    lifecycle_hooks: Dict[str, Callable] = field(default_factory=dict)  # ライフサイクルフック
    

class IServiceProvider(ABC):
    """サービスプロバイダーのインターフェース"""
    
    @abstractmethod
    def get_service(self, service_type: Type) -> Any:
        """サービスを取得"""
        pass
    
    @abstractmethod
    def get_required_service(self, service_type: Type) -> Any:
        """必須サービスを取得（見つからない場合は例外）"""
        pass


class ServiceProvider(IServiceProvider):
    """サービスプロバイダーの実装"""
    
    def __init__(self, services: Dict[Type, Any]):
        self._services = services
    
    def get_service(self, service_type: Type) -> Optional[Any]:
        """サービスを取得"""
        return self._services.get(service_type)
    
    def get_required_service(self, service_type: Type) -> Any:
        """必須サービスを取得"""
        service = self.get_service(service_type)
        if service is None:
            raise ServiceNotFoundError(f"Service {service_type} not found")
        return service


class ServiceScope:
    """サービススコープ"""
    
    def __init__(self, container: 'DIContainer'):
        self._container = container
        self._scoped_instances: Dict[Type, Any] = {}
        self._disposed = False
    
    def get_service(self, service_type: Type) -> Any:
        """スコープ内でサービスを取得"""
        if self._disposed:
            raise RuntimeError("Scope has been disposed")
        
        if service_type in self._scoped_instances:
            return self._scoped_instances[service_type]
        
        # コンテナから作成
        service = self._container.resolve(service_type, scope=self)
        self._scoped_instances[service_type] = service
        return service
    
    def dispose(self):
        """スコープを破棄"""
        if self._disposed:
            return
        
        # 全てのスコープ付きインスタンスを破棄
        for service in self._scoped_instances.values():
            if hasattr(service, 'dispose'):
                try:
                    service.dispose()
                except Exception as e:
                    logger.error(f"Error disposing service: {e}")
        
        self._scoped_instances.clear()
        self._disposed = True


class DIContainer:
    """依存性注入コンテナ"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._resolving: List[Type] = []  # 循環依存検出用
        self._lock = threading.RLock()
        self._scopes: List[ServiceScope] = []
    
    def register(
        self,
        service_type: Type,
        implementation: Union[Type, Callable, Any] = None,
        scope: ServiceScopeEnum = ServiceScopeEnum.SINGLETON,
        dependencies: Dict[str, Union[Type, str]] = None,
        **metadata
    ) -> 'DIContainer':
        """サービスを登録"""
        
        if implementation is None:
            implementation = service_type
        
        # 依存関係を文字列に変換
        deps = {}
        if dependencies:
            for param_name, dep_type in dependencies.items():
                if isinstance(dep_type, type):
                    deps[param_name] = dep_type
                else:
                    deps[param_name] = dep_type
        
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=implementation,
            scope=scope,
            dependencies=deps,
            metadata=metadata
        )
        
        self._services[service_type] = descriptor
        logger.debug(f"Registered service: {service_type.__name__} with scope {scope.value}")
        
        return self
    
    def register_singleton(
        self,
        service_type: Type,
        implementation: Union[Type, Callable, Any] = None,
        **kwargs
    ) -> 'DIContainer':
        """シングルトンサービスを登録"""
        return self.register(service_type, implementation, ServiceScopeEnum.SINGLETON, **kwargs)
    
    def register_transient(
        self,
        service_type: Type,
        implementation: Union[Type, Callable, Any] = None,
        **kwargs
    ) -> 'DIContainer':
        """トランジェントサービスを登録"""
        return self.register(service_type, implementation, ServiceScopeEnum.TRANSIENT, **kwargs)
    
    def register_scoped(
        self,
        service_type: Type,
        implementation: Union[Type, Callable, Any] = None,
        **kwargs
    ) -> 'DIContainer':
        """スコープ付きサービスを登録"""
        return self.register(service_type, implementation, ServiceScopeEnum.SCOPED, **kwargs)
    
    def register_instance(
        self,
        service_type: Type,
        instance: Any
    ) -> 'DIContainer':
        """インスタンスを登録"""
        self._singletons[service_type] = instance
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=instance,
            scope=ServiceScopeEnum.SINGLETON
        )
        return self
    
    def resolve(
        self,
        service_type: Type,
        scope: Optional[ServiceScope] = None
    ) -> Any:
        """サービスを解決"""
        
        with self._lock:
            # 循環依存チェック
            if service_type in self._resolving:
                cycle = " -> ".join(t.__name__ for t in self._resolving) + f" -> {service_type.__name__}"
                raise CircularDependencyError(f"Circular dependency detected: {cycle}")
            
            self._resolving.append(service_type)
            
            try:
                # サービス記述子を取得
                if service_type not in self._services:
                    raise ServiceNotFoundError(f"Service {service_type.__name__} is not registered")
                
                descriptor = self._services[service_type]
                
                # スコープに応じて処理
                if descriptor.scope == ServiceScopeEnum.SINGLETON:
                    if service_type in self._singletons:
                        return self._singletons[service_type]
                    
                    instance = self._create_instance(descriptor)
                    self._singletons[service_type] = instance
                    return instance
                
                elif descriptor.scope == ServiceScopeEnum.TRANSIENT:
                    return self._create_instance(descriptor)
                
                elif descriptor.scope == ServiceScopeEnum.SCOPED:
                    if scope:
                        return scope.get_service(service_type)
                    else:
                        # スコープがない場合はトランジェントとして扱う
                        return self._create_instance(descriptor)
                
            finally:
                self._resolving.remove(service_type)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """インスタンスを作成"""
        
        implementation = descriptor.implementation
        
        # すでにインスタンスの場合
        if not inspect.isclass(implementation) and not callable(implementation):
            return implementation
        
        # 依存関係を解決
        resolved_deps = {}
        for param_name, dep_type in descriptor.dependencies.items():
            if isinstance(dep_type, type):
                resolved_deps[param_name] = self.resolve(dep_type)
            else:
                # 文字列の場合は型を検索
                for service_type in self._services.keys():
                    if service_type.__name__ == dep_type:
                        resolved_deps[param_name] = self.resolve(service_type)
                        break
        
        # インスタンスを作成
        try:
            if inspect.isclass(implementation):
                # コンストラクタの引数を自動解決
                sig = inspect.signature(implementation.__init__)
                auto_resolved = {}
                
                for param_name, param in sig.parameters.items():
                    if param_name == 'self':
                        continue
                    
                    # 明示的な依存関係がある場合
                    if param_name in resolved_deps:
                        auto_resolved[param_name] = resolved_deps[param_name]
                    # 型ヒントから自動解決
                    elif param.annotation != inspect.Parameter.empty:
                        try:
                            auto_resolved[param_name] = self.resolve(param.annotation)
                        except ServiceNotFoundError:
                            if param.default == inspect.Parameter.empty:
                                raise
                
                instance = implementation(**auto_resolved)
                
            elif callable(implementation):
                # ファクトリ関数
                instance = implementation(**resolved_deps)
                
            else:
                instance = implementation
            
            # ライフサイクルフックの実行
            if hasattr(instance, 'on_created'):
                instance.on_created()
            
            return instance
            
        except Exception as e:
            raise ServiceCreationError(f"Failed to create {descriptor.service_type.__name__}: {e}")
    
    def get_service(self, service_type: Type) -> Optional[Any]:
        """サービスを取得"""
        try:
            return self.resolve(service_type)
        except ServiceNotFoundError:
            return None
    
    def get_required_service(self, service_type: Type) -> Any:
        """必須サービスを取得"""
        return self.resolve(service_type)
    
    def create_scope(self) -> ServiceScope:
        """新しいスコープを作成"""
        scope = ServiceScope(self)
        self._scopes.append(scope)
        return scope
    
    @contextmanager
    def scope(self):
        """スコープのコンテキストマネージャー"""
        scope = self.create_scope()
        try:
            yield scope
        finally:
            scope.dispose()
            self._scopes.remove(scope)
    
    def build_service_provider(self) -> ServiceProvider:
        """サービスプロバイダーを構築"""
        services = {}
        for service_type in self._services.keys():
            try:
                services[service_type] = self.resolve(service_type)
            except Exception as e:
                logger.warning(f"Failed to resolve {service_type}: {e}")
        
        return ServiceProvider(services)
    
    def dispose(self):
        """コンテナを破棄"""
        # スコープを破棄
        for scope in self._scopes[:]:
            scope.dispose()
        
        # シングルトンを破棄
        for service in self._singletons.values():
            if hasattr(service, 'dispose'):
                try:
                    service.dispose()
                except Exception as e:
                    logger.error(f"Error disposing service: {e}")
        
        self._singletons.clear()
        self._services.clear()
        self._scopes.clear()


# グローバルコンテナインスタンス
_global_container = DIContainer()


def get_container() -> DIContainer:
    """グローバルコンテナを取得"""
    return _global_container


def configure_services(configure_func: Callable[[DIContainer], None]):
    """サービスを設定"""
    configure_func(_global_container)


# 便利なデコレーター
def injectable(scope: ServiceScopeEnum = ServiceScopeEnum.SINGLETON):
    """クラスを注入可能にするデコレーター"""
    def decorator(cls):
        _global_container.register(cls, cls, scope)
        return cls
    return decorator


def inject(**dependencies):
    """依存関係を注入するデコレーター"""
    def decorator(cls):
        original_init = cls.__init__
        
        def new_init(self, *args, **kwargs):
            # 依存関係を注入
            for name, service_type in dependencies.items():
                if name not in kwargs:
                    kwargs[name] = _global_container.resolve(service_type)
            original_init(self, *args, **kwargs)
        
        cls.__init__ = new_init
        return cls
    
    return decorator
