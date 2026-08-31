from __future__ import annotations

from benchmark.run_muxi import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            default_results_dir_name="ppu_results",
            default_skip_mcrand=True,
            default_skip_ppu_curand=False,
            default_skip_ppu_acrand=False,
            default_acceptance_baseline="ppu_acrand",
            platform_name="PPU",
        )
    )
