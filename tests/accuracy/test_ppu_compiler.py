from __future__ import annotations

import pytest

from flagrand.runtime.ppu_compiler import _normalize_inline_assembly

pytestmark = [pytest.mark.accuracy, pytest.mark.ppu, pytest.mark.runtime]


def test_ppu_compiler_normalizes_namespaced_inline_assembly() -> None:
    source = 'tail call void asm sideeffect "ppu.st.global.b32"\n' 'tail call float asm "ppu.ex2.approx.ftz.f32"\n'

    normalized = _normalize_inline_assembly(source)

    assert "ppu." not in normalized
    assert "st.global.b32" in normalized
    assert "ex2.approx.ftz.f32" in normalized
