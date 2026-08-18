from __future__ import annotations


def device_not_found() -> None:
    raise RuntimeError(
        "FlagRAND could not detect a supported accelerator. Set FLAGRAND_VENDOR "
        "to one of: nvidia, metax, iluvatar, ascend, thead."
    )


__all__ = ["device_not_found"]
