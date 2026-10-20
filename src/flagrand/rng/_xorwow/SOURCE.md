# XORWOW source provenance

Imported on 2026-10-20 from the existing `xorwow_kernels` implementation. The table below records the imported source snapshot. The modules have subsequently received host integration, backend adaptations and formatting changes; these hashes are provenance records, not checksums of the current files. Host integration is in `../xorwow.py`.

The original implementation was measured between 2026-09-29 and 2026-10-01. Those historical measurements do not establish current FlagRAND API performance. Profile stream counts are historical choices; the FlagRAND wrapper keeps its own stream count fixed. Current results are produced by the repository's accuracy and performance runners.

| File | SHA256 of imported source |
|---|---|
| `api.py` | `63d248db41749ba829f1440d79fb3706dfc048afb4334ffc3c1e7fe405791a1f` |
| `reference.py` | `3055dec18f6428bec1c3affdbab45e6df30d0eee07c77b19d5d7462c00e48dbe` |
| `profiles.py` | `18999d889dcf2bb2855cb5d773725f2c05a0667648dfa38e1ecacfb46c399f5b` |
| `_binding.py` | `f71f52c174b73577feefb13f95f017c01b9ab8258174ceb72bb86f5e188d79c9` |
| `kernels.py` | `ff276388840fd1203d5970a43d824f17912ab47806f3137110fec3c282ad899e` |
| `native_math.py` | `30c38b21175f0f6ee771607eeb2c3042fbf4cbf970613386c25d7eebb6dd8b92` |
| `range_math64.py` | `1a39c38d0bc31cde0eab08a6cb217b4ac12aad34fd6389fe3ea70dd60f4cba43` |
| `metax_math.py` | `fd7894231eb234ac2de14cc8469b8856866e2b390accf6fcc1e0f37b6a4640ba` |
| `mthreads_math.py` | `d5bbe8d8df87d4c7215117cb45ffabc99e0f1056c9910da7a71afb8bd76207bf` |
| `ascend_kernels.py` | `7bfbfdd73cbf3592b93a5a2a14e5f359023a800c5eecf0949bdee6ce2f89339a` |
| `corex_packed64.py` | `4d6b1948a3284e1760093f0651baad386d683574393ce6d9abac0227cd1cff9f` |
| `packed64.py` | `1ec3318f3acccdcbd3f14c3a2c1a92a8a27533df420c864919eb29a9f0b2f585` |
| `soft64_kernels.py` | `9d49b3157628c3408b0743c48f0d9c21e97f3b36dcc44a369c3e712028c4a463` |
| `soft64_math.py` | `7edae16fab0d763692cffe72a9e596dabbf7ad46bd07804f467c7f0cf077f765` |
| `ascend_soft64_math.py` | `1e5bb166b18b5f49b0c868567931fb2de43e6a5485a868348ab46135360277c6` |
| `ascend_soft64_loop_math.py` | `385437f1b4b5ff4ce510e271b23a1590b051cf814dc5d3e7a779116b0ab2e58a` |
| `ascend_soft64_coefficients.py` | `39aa0d4c977220793c802b1a7bbfc158f87ecd9d8a71fd95bad9e88efece0beb` |
| `ascend_simple_product.py` | `5bd90ef0f4aacb19f6f855ee7dc0c9d5acce9d127d5968968ed24242a8f9aa3a` |
| `xpu_soft64_math.py` | `c360c3fc3d624b0cd7a88b869e3146aeea8e12001a3805a18a66ded300709370` |
| `xpu_soft64_loop_math.py` | `396ba10c5d986a50beea85e386fda6c7762d60cff8cc801042341f840b7695c0` |
| `xpu_product.py` | `97fc9283ac6d9af343f02b12ffea822efcea36d421ce3ccc431a4444529bff1b` |
| `xpu_compiler.py` | `df3378be5e4a3105bf5a04e0f85c2dfad38e860dc89659d2929608b69877c883` |
