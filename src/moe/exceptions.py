"""
MoE System Exceptions
MoEシステムのカスタム例外クラス
"""


class MoEException(Exception):
    """MoEシステムの基底例外クラス"""
    pass


class MoEConfigError(MoEException):
    """設定エラー"""
    pass


class MoEModelError(MoEException):
    """モデル関連エラー"""
    pass


class MoETrainingError(MoEException):
    """トレーニング関連エラー"""
    pass


class MoEInferenceError(MoEException):
    """推論関連エラー"""
    pass


class MoEDataError(MoEException):
    """データ関連エラー"""
    pass


class MoERouterError(MoEException):
    """ルーター関連エラー"""
    pass


class MoEExpertError(MoEException):
    """エキスパート関連エラー"""
    pass