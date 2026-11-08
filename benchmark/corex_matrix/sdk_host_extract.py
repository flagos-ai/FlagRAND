#!/usr/bin/env python3
"""Extract unchanged SDK MRG host helpers, then build an independent probe."""

import hashlib
import json
import re
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def block(text, match):
    opening = text.index("{", match.start())
    depth = 0
    for index in range(opening, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[match.start() : index + 1], match.start(), index + 1
    raise ValueError("Unclosed SDK block")


def extract_host(include, output):
    kernel_path = include / "curand_kernel.h"
    table_path = include / "curand_mrg32k3a.h"
    kernel, tables = kernel_path.read_text(), table_path.read_text()
    parts, records, matrices = [], [], {}

    def keep(label, source_path, text, match, suffix=""):
        code, first, last = block(text, match)
        records.append(
            {
                "label": label,
                "source": str(source_path),
                "start_line": text.count("\n", 0, first) + 1,
                "end_line": text.count("\n", 0, last) + 1,
                "verbatim_sha256": digest(code.encode()),
                "bytes": len(code.encode()),
            }
        )
        parts.append(code + suffix)
        return code

    keep("MRG state", kernel_path, kernel, re.search(r"struct curandStateMRG32k3a\s*\{", kernel), ";")
    parts.append("typedef struct curandStateMRG32k3a curandStateMRG32k3a_t;")
    for match in re.finditer(r"^#define MRG32K3A_[^\n]+", kernel, re.M):
        parts.append(match.group())
    for name in ("mrg32k3aM1Host", "mrg32k3aM2Host", "mrg32k3aM1SubSeqHost", "mrg32k3aM2SubSeqHost"):
        match = re.search(r"static unsigned int " + name + r"\[.*?=\s*\{", tables)
        code = keep(name, table_path, tables, match, ";")
        values = [int(x) for x in re.findall(r"\b(\d+)u\b", code)]
        matrices[name] = [[values[i + j : i + j + 3] for j in (0, 3, 6)] for i in range(0, len(values), 9)]
    names = [
        "curand_MRGmod",
        "curand_MRGmodMul",
        "curand_MRGmatMul3x3",
        "curand_MRGmatVecMul3x3",
        "curand_MRGmatPow3x3",
        "skipahead",
        "skipahead_subsequence",
        "curand_init",
    ]
    for name in names:
        matches = list(re.finditer(r"QUALIFIERS\s+[\w\s:*]+\b" + name + r"\s*\([^;{}]*\)\s*\{", kernel))
        if name in ("skipahead", "skipahead_subsequence", "curand_init"):
            matches = [m for m in matches if "curandStateMRG32k3a_t" in m.group()]
        if len(matches) != 1:
            raise ValueError(f"Expected one SDK MRG {name}, found {len(matches)}")
        keep(name, kernel_path, kernel, matches[0])
    license_text = kernel[: kernel.index("#if !defined(CURAND_KERNEL_H_)")]
    source = (
        license_text + "\n#include <cmath>\n#include <cstdint>\n#define QUALIFIERS static inline\nusing std::floor;\n"
    )
    source += "\n\n".join(parts)
    source += """
extern "C" int probe_host_init(uint64_t seed, unsigned n, uint32_t *out) {
    for (unsigned i = 0; i < n; ++i) {
        curandStateMRG32k3a_t state{};
        curand_init(seed, i, 0, &state);
        for (unsigned j = 0; j < 3; ++j) {
            out[i * 6 + j] = state.s1[j];
            out[i * 6 + 3 + j] = state.s2[j];
        }
    }
    return 0;
}
"""
    target = output / "sdk_mrg_host_init.cpp"
    target.write_text(source)
    (output / "sdk-host-matrices.json").write_text(json.dumps(matrices) + "\n")
    observation = {
        "sdk_blocks": records,
        "qualifiers": "QUALIFIERS=static inline, only host compilation attribute changed",
        "generated_source_sha256": digest(target.read_bytes()),
        "sdk_headers": [{"path": str(p), "sha256": digest(p.read_bytes())} for p in (kernel_path, table_path)],
    }
    (output / "host-extraction.json").write_text(json.dumps(observation, indent=2) + "\n")
    return target
