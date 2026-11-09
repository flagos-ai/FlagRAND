"""Cached host planning for standard Ascend MRG raw and FP32 distributions."""

import torch
import triton

from flagrand.runtime.specialization import integer_specialization_key

from ._mrg32k3a_ascend_active import tile_count


class RawPlan:

    def __init__(self, engine, out, distribution, config, launcher, joint_min_count):
        self.state = engine.ensure_state(out.device)
        self.config = config
        self.n = out.numel()
        self.streams = engine.nstreams
        self.block = 1024
        self.total_tiles = (self.streams + self.block - 1) // self.block
        self.fixed_grid = (self.total_tiles,) if self.n >= self.streams else None
        self.full_tiles = self.streams % self.block == 0
        self.launcher = launcher
        full_rounds = self.n >= self.streams and self.n % self.streams == 0
        output_block = triton.next_power_of_2(min(self.n, self.block))
        joint = self.n >= max(self.streams, joint_min_count)
        self.constants = tuple(
            (
                (self.streams, self.block, distribution, False, aligned, False, full_rounds, output_block, joint)
                for aligned in (False, True)
            )
        )
        self.keys = {}
        self.options = (4, False, False)
        self.advance = self.n % self.streams

    def call(self, engine, out):
        cursor = engine._cursor
        aligned = cursor % self.block == 0
        alignment = out.data_ptr() % 16
        variant = (alignment, integer_specialization_key(cursor), aligned)
        key = self.keys.get(variant)
        if key is None:
            key = (self.config, variant)
            self.keys[variant] = key
        grid = self.fixed_grid
        if grid is None:
            if self.full_tiles:
                grid = (min(self.total_tiles, (cursor % self.block + self.n + self.block - 1) // self.block),)
            else:
                grid = (tile_count(self.streams, cursor, self.n, self.block),)
        self.launcher.launch(
            grid,
            (self.state, out, self.state, self.n, cursor, 0, 0.0, 1.0),
            self.constants[aligned],
            self.options,
            specialization_key=key,
        )
        engine.offset += self.n
        engine._state_offset = engine.offset
        engine._cursor = (cursor + self.advance) % self.streams
        return out


class NormalPlan:

    def __init__(self, engine, out, distribution, config, launcher, joint_min_count):
        self.state = engine.ensure_state(out.device)
        self.config = config
        (self.n, self.streams) = (out.numel(), engine.nstreams)
        self.block = 256 if self.n <= 1024 else 1024
        self.total_tiles = (self.streams + self.block - 1) // self.block
        self.full_tiles = self.streams % self.block == 0
        self.launcher = launcher
        self.options = (4, False, False)
        self.counts = ((self.n + 1) // 2, self.n // 2)
        self.emit_spares = (self.n % 2 != 0, (self.n - 1) % 2 != 0)
        self.offset_advances = tuple((2 * count for count in self.counts))
        self.cursor_advances = tuple((count % self.streams for count in self.counts))
        self.fixed_grids = tuple(
            ((1,) if count == 0 else (self.total_tiles,) if count >= self.streams else None for count in self.counts)
        )
        output_block = triton.next_power_of_2(min(self.n, 2 * self.block))
        self.constants = tuple(
            (
                tuple(
                    (
                        (
                            self.streams,
                            self.block,
                            distribution,
                            has_spare,
                            aligned,
                            self.emit_spares[has_spare],
                            count >= self.streams and count % self.streams == 0,
                            output_block,
                            count >= 32768,
                        )
                        for aligned in (False, True)
                    )
                )
                for (has_spare, count) in zip((False, True), self.counts)
            )
        )
        self.keys = {}
        if False not in engine._normal_cache:
            engine._normal_cache[False] = torch.empty(2, dtype=out.dtype, device=out.device)
        if self.n % 2 == 0:
            self.even_count = self.counts[False]
            self.even_constants = self.constants[False]
            self.even_fixed_grid = self.fixed_grids[False]
            self.even_offset_advance = self.offset_advances[False]
            self.even_cursor_advance = self.cursor_advances[False]
            self.call = self._call_even

    def call(self, engine, out):
        cursor = engine._cursor
        has_spare = engine._normal_has_spare.get(False, False)
        spare_index = engine._normal_spare_index.get(False, 0)
        spare = engine._normal_cache[False]
        aligned = cursor % self.block == 0
        variant = (out.data_ptr() % 16, integer_specialization_key(cursor), spare_index, has_spare, aligned)
        key = self.keys.get(variant)
        if key is None:
            key = (self.config, variant)
            self.keys[variant] = key
        grid = self.fixed_grids[has_spare]
        if grid is None:
            count = self.counts[has_spare]
            if self.full_tiles:
                grid = (min(self.total_tiles, (cursor % self.block + count + self.block - 1) // self.block),)
            else:
                grid = (tile_count(self.streams, cursor, count, self.block),)
        self.launcher.launch(
            grid,
            (self.state, out, spare, self.n, cursor, spare_index, 0.0, 1.0),
            self.constants[has_spare][aligned],
            self.options,
            specialization_key=key,
        )
        emit_spare = self.emit_spares[has_spare]
        engine._normal_has_spare[False] = emit_spare
        if emit_spare:
            engine._normal_spare_index[False] = 1 - spare_index
        engine.offset += self.offset_advances[has_spare]
        engine._state_offset = engine.offset
        engine._cursor = (cursor + self.cursor_advances[has_spare]) % self.streams
        return out

    def _call_even(self, engine, out):
        if engine._normal_has_spare.get(False, False):
            return NormalPlan.call(self, engine, out)
        cursor = engine._cursor
        spare = engine._normal_cache[False]
        aligned = cursor % self.block == 0
        variant = (out.data_ptr() % 16, integer_specialization_key(cursor), aligned)
        key = self.keys.get(variant)
        if key is None:
            key = (self.config, variant)
            self.keys[variant] = key
        grid = self.even_fixed_grid
        if grid is None:
            if self.full_tiles:
                grid = (min(self.total_tiles, (cursor % self.block + self.even_count + self.block - 1) // self.block),)
            else:
                grid = (tile_count(self.streams, cursor, self.even_count, self.block),)
        self.launcher.launch(
            grid,
            (self.state, out, spare, self.n, cursor, 0, 0.0, 1.0),
            self.even_constants[aligned],
            self.options,
            specialization_key=key,
        )
        # Retain insertion on the first call and the original post-launch
        # write order, including when a launch hook observes the engine.
        engine._normal_has_spare[False] = False
        engine.offset += self.even_offset_advance
        engine._state_offset = engine.offset
        engine._cursor = (cursor + self.even_cursor_advance) % self.streams
        return out
