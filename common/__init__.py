"""共享模块：数据集 / 指标 / 随机种子 / 可视化 / 训练引擎。

所有对比实验模型复用本包，保证数据、指标、评估流程完全一致。
"""


def _patch_torchvision_compat() -> None:
    """兼容旧写法 `from torchvision.models.utils import load_state_dict_from_url`。

    torchvision >= 0.13 删除了 `torchvision.models.utils`，而不少老仓库仍按旧路径导入，
    会报 ModuleNotFoundError。这里注册一个兼容模块转发到 torch 的官方实现，
    让各原仓库代码无需改动即可运行。
    """
    try:
        import sys
        import types

        import torchvision.models as tv_models

        if hasattr(tv_models, "utils"):
            return

        mod = types.ModuleType("torchvision.models.utils")
        try:
            from torch.hub import load_state_dict_from_url
        except Exception:  # noqa: BLE001
            from torch.utils.model_zoo import load_state_dict_from_url  # type: ignore
        mod.load_state_dict_from_url = load_state_dict_from_url

        try:  # 极老版本还用过 load_url
            from torch.utils.model_zoo import load_url
            mod.load_url = load_url
        except Exception:  # noqa: BLE001
            pass

        sys.modules["torchvision.models.utils"] = mod
        tv_models.utils = mod
    except Exception:  # noqa: BLE001
        pass


_patch_torchvision_compat()
