# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for model-agnostic activation-checkpoint block discovery."""

import unittest
from functools import partial
from unittest.mock import MagicMock, call, patch

import torch
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaForCausalLM


from hyper_parallel.core.activation_memory.wrapper import ckpt_wrapper as _checkpoint_wrapper
from hyper_parallel.distributed.activation_checkpoint import (
    _apply_activation_checkpointing,
    _find_transformer_block_modules,
    _find_transformer_layer_container_infos,
    _wrap_layer_containers,
    apply_submodule_checkpointing,
    normalize_activation_checkpoint_layers,
)


_ACTIVATION_CHECKPOINT_MODULE = (
    "hyper_parallel.distributed.activation_checkpoint"
)


class _DiscoveryBlock(nn.Module):
    """Minimal block used to test model-agnostic layer discovery."""

    def __init__(self) -> None:
        """Create one minimal transformer block."""
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the fixture's linear layer."""
        return self.linear(inputs)


class _DiscoveryOwner(nn.Module):
    """HF-style owner whose repeated block path has an arbitrary name."""

    gradient_checkpointing = False

    def __init__(self) -> None:
        """Create an owner with a non-contiguous repeated block container."""
        super().__init__()
        self.decoder = nn.ModuleDict({"2": _DiscoveryBlock(), "7": _DiscoveryBlock()})

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply every block in registration order."""
        for block in self.decoder.values():
            inputs = block(inputs)
        return inputs


class _DiscoveryModel(nn.Module):
    """Model with multiple marked towers and no architecture-specific class name."""

    def __init__(self) -> None:
        """Create independent text and image towers."""
        super().__init__()
        self.text_tower = _DiscoveryOwner()
        self.image_tower = _DiscoveryOwner()

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply both towers and combine their outputs."""
        return self.text_tower(inputs) + self.image_tower(inputs)


class _UnmarkedDiscoveryModel(nn.Module):
    """Repeated layers without the HF discovery marker."""

    def __init__(self) -> None:
        """Create repeated blocks without a discovery marker."""
        super().__init__()
        self.layers = nn.ModuleList([_DiscoveryBlock(), _DiscoveryBlock()])

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply every unmarked block in order."""
        for block in self.layers:
            inputs = block(inputs)
        return inputs


class _CheckpointableSubmodules(nn.Module):
    """Minimal transformer block exposing all supported checkpoint targets."""

    def __init__(self) -> None:
        """Create submodules recognized by the checkpointing fallback."""
        super().__init__()
        self.mlp = nn.Linear(2, 2)
        self.self_attn = nn.Linear(2, 2)
        self.input_layernorm = nn.LayerNorm(2)
        self.post_attention_layernorm = nn.LayerNorm(2)


class _NamedBlocksOwner(nn.Module):
    """HF-style owner whose blocks are registered under names, not indices."""

    gradient_checkpointing = False

    def __init__(self) -> None:
        """Create two blocks registered by name."""
        super().__init__()
        self.decoder = nn.ModuleDict({"first": _DiscoveryBlock(), "second": _DiscoveryBlock()})

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply both blocks in registration order."""
        for block in self.decoder.values():
            inputs = block(inputs)
        return inputs


class _IndexedOwner(nn.Module):
    """HF-style owner whose blocks are registered under their layer index."""

    gradient_checkpointing = False

    def __init__(self, num_layers: int = 4) -> None:
        """Create ``num_layers`` blocks in a ``ModuleList``."""
        super().__init__()
        self.layers = nn.ModuleList(_DiscoveryBlock() for _ in range(num_layers))

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply every block in order."""
        for block in self.layers:
            inputs = block(inputs)
        return inputs


def _tiny_llama(num_layers: int) -> LlamaForCausalLM:
    """Build a CPU Llama whose decoder layers are HF ``GradientCheckpointingLayer``s."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    return LlamaForCausalLM(config)


def _count_forward(calls: list[int], index: int, module: nn.Module, args: tuple) -> None:
    """Count one forward call of decoder layer ``index``."""
    del module, args
    calls[index] += 1


def _forward_calls_per_layer(model: LlamaForCausalLM) -> list[int]:
    """Run one training step and count each decoder layer's forward calls.

    A recomputed layer runs its forward again during backward, so it counts
    two calls where a layer that keeps its activations counts one.
    """
    calls = [0] * len(model.model.layers)
    for index, layer in enumerate(model.model.layers):
        # A layer may be wrapped; the hook belongs on the decoder layer itself.
        inner = getattr(layer, "_wrapped_module", layer)
        inner.register_forward_pre_hook(partial(_count_forward, calls, index))
    model.train()
    input_ids = torch.randint(0, 32, (1, 8))
    model(input_ids=input_ids, labels=input_ids).loss.backward()
    return calls


class TestTransformerBlockDiscovery(unittest.TestCase):
    """Tests for model-agnostic transformer block discovery."""

    def test_discovery_uses_marker_and_not_model_paths(self):
        """Marked arbitrary towers should all be discovered."""
        blocks, _ = _find_transformer_block_modules(_DiscoveryModel())

        self.assertEqual(
            [block.fqn for block in blocks],
            [
                "text_tower.decoder.2",
                "text_tower.decoder.7",
                "image_tower.decoder.2",
                "image_tower.decoder.7",
            ],
        )

    def test_container_info_preserves_registered_keys(self):
        """Container metadata should retain non-contiguous ModuleDict keys."""
        containers = _find_transformer_layer_container_infos(_DiscoveryModel())

        self.assertEqual(
            [container.path for container in containers],
            ["text_tower.decoder", "image_tower.decoder"],
        )
        self.assertEqual(
            [[block.child_name for block in container.blocks] for container in containers],
            [["2", "7"], ["2", "7"]],
        )

    def test_unmarked_layers_are_not_selected(self):
        """A conventional layers attribute is insufficient without the marker."""
        blocks, _ = _find_transformer_block_modules(_UnmarkedDiscoveryModel())

        self.assertEqual(blocks, [])

    def test_shared_block_is_selected_once(self):
        """Aliased block objects should not be wrapped more than once."""
        owner = _DiscoveryOwner()
        shared = _DiscoveryBlock()
        owner.decoder["2"] = shared
        owner.decoder["7"] = shared

        blocks, _ = _find_transformer_block_modules(owner)

        self.assertEqual([block.child_name for block in blocks], ["2"])

    def test_wrapping_uses_actual_container_child_names(self):
        """Wrapping should preserve arbitrary ModuleDict keys."""
        model = _DiscoveryModel()
        containers = _find_transformer_layer_container_infos(model)

        wrapped_count = _wrap_layer_containers(
            containers,
            nn.Sequential,
        )

        self.assertEqual(wrapped_count, 4)
        for tower in (model.text_tower, model.image_tower):
            self.assertTrue(all(isinstance(block, nn.Sequential) for block in tower.decoder.values()))

    def test_missing_marker_has_clear_activation_checkpoint_error(self):
        """Activation checkpointing should fail instead of guessing a container."""
        with self.assertRaisesRegex(ValueError, "gradient_checkpointing"):
            _apply_activation_checkpointing(_UnmarkedDiscoveryModel(), "selective")


class TestActivationCheckpointSwapInputs(unittest.TestCase):
    """Tests for activation checkpoint input-swapping configuration."""

    def setUp(self) -> None:
        """Keep swap-input tests independent of optional and platform adapters."""
        hf_checkpointing_patch = patch(
            f"{_ACTIVATION_CHECKPOINT_MODULE}._should_use_hf_native_gradient_checkpointing",
            return_value=False,
        )
        hf_checkpointing_patch.start()
        self.addCleanup(hf_checkpointing_patch.stop)

        checkpoint_wrapper_patch = patch(
            f"{_ACTIVATION_CHECKPOINT_MODULE}.checkpoint_wrapper",
            new=_checkpoint_wrapper,
        )
        checkpoint_wrapper_patch.start()
        self.addCleanup(checkpoint_wrapper_patch.stop)

    @staticmethod
    def _wrapped_blocks(model):
        """Return all checkpoint-wrapped blocks from the discovery fixture."""
        return [
            block
            for tower in (model.text_tower, model.image_tower)
            for block in tower.decoder.values()
        ]

    def test_eager_checkpoint_wrappers_receive_swap_inputs(self):
        """Eager full and selective wrappers should receive the configured value."""
        for mode in ("full", "selective"):
            for swap_inputs in (False, True):
                with self.subTest(mode=mode, swap_inputs=swap_inputs):
                    model = _DiscoveryModel()
                    with (
                        patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_profiler_ops_sac_ignored"),
                        patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_fsdp_ops_sac_ignored"),
                    ):
                        _apply_activation_checkpointing(
                            model,
                            mode,
                            enable_compile=False,
                            swap_inputs=swap_inputs,
                        )

                    for block in self._wrapped_blocks(model):
                        self.assertIs(block.checkpoint_kwargs["swap_inputs"], swap_inputs)

    def test_compile_checkpoint_wrappers_omit_swap_inputs(self):
        """Compile wrappers should omit swap_inputs and report that it is disabled."""
        expected_warning = (
            "activation_checkpoint.swap_inputs is not supported with torch.compile; "
            "input swapping will be disabled."
        )
        for mode in ("full", "selective"):
            with self.subTest(mode=mode):
                model = _DiscoveryModel()
                with (
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_profiler_ops_sac_ignored"),
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_fsdp_ops_sac_ignored"),
                    self.assertLogs(_ACTIVATION_CHECKPOINT_MODULE, level="WARNING") as log_context,
                ):
                    _apply_activation_checkpointing(
                        model,
                        mode,
                        enable_compile=True,
                        swap_inputs=True,
                    )

                self.assertIn(expected_warning, "\n".join(log_context.output))
                for block in self._wrapped_blocks(model):
                    self.assertNotIn("swap_inputs", block.checkpoint_kwargs)

    def test_hf_native_checkpointing_warns_when_swap_inputs_enabled(self):
        """HF-native checkpointing should warn that input swapping is disabled."""
        model = _DiscoveryModel()
        model.gradient_checkpointing_enable = MagicMock()
        expected_warning = (
            "activation_checkpoint.swap_inputs is not supported by Hugging Face native "
            "gradient checkpointing for now; input swapping will be disabled."
        )

        with (
            patch(
                f"{_ACTIVATION_CHECKPOINT_MODULE}._should_use_hf_native_gradient_checkpointing",
                return_value=True,
            ),
            self.assertLogs(_ACTIVATION_CHECKPOINT_MODULE, level="WARNING") as log_context,
        ):
            result = _apply_activation_checkpointing(
                model,
                "full",
                swap_inputs=True,
            )

        self.assertIs(result, model)
        self.assertIn(expected_warning, "\n".join(log_context.output))
        model.gradient_checkpointing_enable.assert_called_once_with(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

    def test_rejects_non_boolean_swap_inputs(self):
        """The component boundary should reject ambiguous swap_inputs values."""
        with self.assertRaisesRegex(
            ValueError,
            "activation_checkpoint.swap_inputs must be bool",
        ):
            _apply_activation_checkpointing(
                _DiscoveryModel(),
                "full",
                swap_inputs=1,
            )

    def test_submodule_checkpointing_handles_swap_inputs_by_compile_mode(self):
        """Submodule wrappers should pass swap_inputs only during eager execution."""
        for enable_compile in (False, True):
            with self.subTest(enable_compile=enable_compile):
                block = _CheckpointableSubmodules()

                wrapped_count = apply_submodule_checkpointing(
                    [block],
                    has_kv_sharing=False,
                    enable_compile=enable_compile,
                    swap_inputs=True,
                )

                self.assertEqual(wrapped_count, 4)
                for attr_name in (
                    "mlp",
                    "self_attn",
                    "input_layernorm",
                    "post_attention_layernorm",
                ):
                    checkpoint_kwargs = getattr(block, attr_name).checkpoint_kwargs
                    if enable_compile:
                        self.assertNotIn("swap_inputs", checkpoint_kwargs)
                    else:
                        self.assertIs(checkpoint_kwargs["swap_inputs"], True)

    def test_swap_inputs_registers_prefetch_within_each_container(self):
        """Eager input swapping should connect adjacent wrapped transformer blocks."""
        for mode in ("full", "selective"):
            with self.subTest(mode=mode):
                model = _DiscoveryModel()
                swap_manager = MagicMock()
                with (
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.SwapManager", return_value=swap_manager),
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_profiler_ops_sac_ignored"),
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.ensure_fsdp_ops_sac_ignored"),
                ):
                    _apply_activation_checkpointing(
                        model,
                        mode,
                        swap_inputs=True,
                    )

                swap_manager.set_forward_prefetch_layer.assert_has_calls(
                    [
                        call(model.text_tower.decoder["2"], model.text_tower.decoder["7"]),
                        call(model.image_tower.decoder["2"], model.image_tower.decoder["7"]),
                    ]
                )
                self.assertEqual(swap_manager.set_forward_prefetch_layer.call_count, 2)

    def test_submodule_checkpointing_registers_matching_prefetch_chains(self):
        """KV-shared fallback should connect matching non-attention wrappers."""
        model = _DiscoveryOwner()
        model.decoder["2"] = _CheckpointableSubmodules()
        model.decoder["7"] = _CheckpointableSubmodules()
        swap_manager = MagicMock()

        with (
            patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.SwapManager", return_value=swap_manager),
            patch(
                f"{_ACTIVATION_CHECKPOINT_MODULE}._detect_kv_sharing_and_maybe_disable_cache",
                return_value=True,
            ),
        ):
            _apply_activation_checkpointing(
                model,
                "full",
                swap_inputs=True,
            )

        first_block = model.decoder["2"]
        second_block = model.decoder["7"]
        expected_calls = [
            call(getattr(first_block, attr_name), getattr(second_block, attr_name))
            for attr_name in (
                "mlp",
                "input_layernorm",
                "post_attention_layernorm",
            )
        ]
        swap_manager.set_forward_prefetch_layer.assert_has_calls(expected_calls)
        self.assertEqual(swap_manager.set_forward_prefetch_layer.call_count, 3)
        self.assertIsInstance(first_block.self_attn, nn.Linear)
        self.assertIsInstance(second_block.self_attn, nn.Linear)

    def test_prefetch_registration_requires_effective_swap_inputs(self):
        """Disabled or compile-only input swapping should not register prefetch hooks."""
        for enable_compile, swap_inputs in ((False, False), (True, True)):
            with self.subTest(enable_compile=enable_compile, swap_inputs=swap_inputs):
                model = _DiscoveryModel()
                swap_manager = MagicMock()
                with (
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.SwapManager", return_value=swap_manager),
                    patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.logger.warning"),
                ):
                    _apply_activation_checkpointing(
                        model,
                        "full",
                        enable_compile=enable_compile,
                        swap_inputs=swap_inputs,
                    )

                swap_manager.set_forward_prefetch_layer.assert_not_called()


class TestActivationCheckpointLayerPlan(unittest.TestCase):
    """Tests for running some layers in another mode than the default one."""

    def setUp(self) -> None:
        """Keep the wrapper-path tests independent of optional operators."""
        for name in ("ensure_profiler_ops_sac_ignored", "ensure_fsdp_ops_sac_ignored"):
            ignore_patch = patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.{name}")
            ignore_patch.start()
            self.addCleanup(ignore_patch.stop)

    @staticmethod
    def _wrapper_path(model: nn.Module, mode: str, layers: dict, **kwargs) -> nn.Module:
        """Apply a plan with Hyper Parallel's own wrappers, which record their kwargs."""
        with (
            patch(
                f"{_ACTIVATION_CHECKPOINT_MODULE}._should_use_hf_native_gradient_checkpointing",
                return_value=False,
            ),
            patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.checkpoint_wrapper", new=_checkpoint_wrapper),
        ):
            return _apply_activation_checkpointing(model, mode, layers=layers, **kwargs)

    @staticmethod
    def _block_kinds(model: _IndexedOwner) -> list[str]:
        """Name what runs each block: ``plain``, ``full`` or ``selective``."""
        kinds = []
        for block in model.layers:
            if not hasattr(block, "checkpoint_kwargs"):
                kinds.append("plain")
            else:
                kinds.append("selective" if "context_fn" in block.checkpoint_kwargs else "full")
        return kinds

    def test_hf_native_full_plan_recomputes_only_the_other_layers(self):
        """A layer the plan keeps runs its forward once in a training step, a recomputed one twice."""
        model = _tiny_llama(num_layers=3)

        _apply_activation_checkpointing(model, "full", layers={"1": "off"})

        flags = [layer.gradient_checkpointing for layer in model.model.layers]
        self.assertEqual(flags, [True, False, True], f"flags={flags}")
        calls = _forward_calls_per_layer(model)
        self.assertEqual(calls, [2, 1, 2], f"calls={calls}")

    def test_hf_native_full_without_plan_recomputes_every_layer(self):
        """Without a plan the HF-native path still recomputes every layer."""
        model = _tiny_llama(num_layers=3)

        _apply_activation_checkpointing(model, "full")

        calls = _forward_calls_per_layer(model)
        self.assertEqual(calls, [2, 2, 2], f"calls={calls}")

    def test_hf_native_full_plan_wraps_selective_layers_itself(self):
        """A selective layer beside HF-native full layers gets the selective wrapper and no HF flag."""
        model = _tiny_llama(num_layers=3)

        with patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.checkpoint_wrapper", new=_checkpoint_wrapper):
            _apply_activation_checkpointing(model, "full", layers={"1": "selective", "2": "off"})

        layers = model.model.layers
        self.assertTrue(layers[0].gradient_checkpointing, "layer 0 should use HF-native checkpointing")
        self.assertIn("context_fn", layers[1].checkpoint_kwargs, f"kwargs={layers[1].checkpoint_kwargs}")
        self.assertFalse(
            layers[1]._wrapped_module.gradient_checkpointing,
            "layer 1 must not be checkpointed twice",
        )
        self.assertFalse(layers[2].gradient_checkpointing, "layer 2 keeps its activations")
        self.assertFalse(hasattr(layers[2], "checkpoint_kwargs"), "layer 2 must not be wrapped")

    def test_wrapper_path_wraps_only_the_layers_the_plan_recomputes(self):
        """Every block runs the default mode unless a range of the plan names it."""
        cases = (
            ("full", {"2-3": "off"}, ["full", "full", "plain", "plain"]),
            ("full", {1: "selective", "3": "off"}, ["full", "selective", "full", "plain"]),
            ("selective", {"0": "off", "3": "full"}, ["plain", "selective", "selective", "full"]),
            ("selective", {"0-3": "off"}, ["plain", "plain", "plain", "plain"]),
        )
        for mode, layers, expected in cases:
            with self.subTest(mode=mode, layers=layers):
                model = self._wrapper_path(_IndexedOwner(), mode, layers)

                kinds = self._block_kinds(model)
                self.assertEqual(kinds, expected, f"mode={mode}, layers={layers}, kinds={kinds}")

    def test_plan_is_logged_as_ranges_per_mode(self):
        """The log names each mode's layers, so a run shows the plan it ran."""
        with self.assertLogs(_ACTIVATION_CHECKPOINT_MODULE, level="INFO") as log_context:
            self._wrapper_path(_IndexedOwner(), "full", {"1": "off", "3": "selective"})

        expected = "Activation checkpointing per layer in layers: full 0, 2; off 1; selective 3"
        self.assertIn(expected, "\n".join(log_context.output))

    def test_swap_prefetch_chain_skips_layers_left_off(self):
        """Input-swap prefetch connects each wrapped block to the next wrapped one."""
        model = _IndexedOwner(num_layers=3)
        swap_manager = MagicMock()

        with patch(f"{_ACTIVATION_CHECKPOINT_MODULE}.SwapManager", return_value=swap_manager):
            self._wrapper_path(model, "full", {"1": "off"}, swap_inputs=True)

        swap_manager.set_forward_prefetch_layer.assert_called_once_with(model.layers[0], model.layers[2])

    def test_plan_must_name_layers_the_model_holds(self):
        """A plan naming a missing layer, or a model it cannot index, is refused before any wrapping."""
        cases = (
            (_IndexedOwner(), {"3-4": "off"}, r"names layer 4, but layers holds layers 0-3"),
            (_DiscoveryModel(), {"2": "off"}, r"needs one repeated block container, but the model has 2"),
            (_NamedBlocksOwner(), {"0": "off"}, r"needs blocks registered under their index"),
        )
        for model, layers, message in cases:
            with self.subTest(layers=layers, model=type(model).__name__):
                with self.assertRaisesRegex(ValueError, message):
                    self._wrapper_path(model, "full", layers)
                wrapped = [
                    child for module in model.modules() for child in module.children()
                    if hasattr(child, "checkpoint_kwargs")
                ]
                self.assertEqual(wrapped, [], "a refused plan must leave the model unwrapped")

    def test_plan_needs_a_mode_that_recomputes(self):
        """A plan is a set of exceptions to full or selective recompute."""
        for mode in ("off", None):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "must then be 'full' or 'selective'"):
                    normalize_activation_checkpoint_layers(mode, {"0": "full"})

    def test_normalized_plan_is_written_one_way(self):
        """Keys become strings in layer order and an unquoted YAML ``off`` becomes ``"off"``."""
        normalized = normalize_activation_checkpoint_layers(
            "full", {"6-7": False, 0: "selective", " 3 - 4 ": "off"}
        )
        self.assertEqual(
            list(normalized.items()),
            [("0", "selective"), (" 3 - 4 ", "off"), ("6-7", "off")],
            f"normalized={normalized}",
        )
        for empty in (None, {}):
            with self.subTest(layers=empty):
                self.assertIsNone(normalize_activation_checkpoint_layers("full", empty))

    def test_malformed_plans_are_refused(self):
        """Each malformed entry is refused with the entry named."""
        cases = (
            ({"2-1": "off"}, r"range '2-1' ends before it starts"),
            ({"0-3": "off", "2": "selective"}, r"entries '0-3' and '2' overlap"),
            ({"first": "off"}, r"must be a layer index or a range 'first-last', but got 'first'"),
            ({-1: "off"}, r"must not be negative, but got -1"),
            ({True: "off"}, r"must be a layer index or a range"),
            ({"1": True}, r"activation_checkpoint.layers\['1'\] must be one of"),
            ({"1": "swap"}, r"activation_checkpoint.layers\['1'\] must be one of"),
            (["0", "off"], r"must be a mapping, but got list"),
        )
        for layers, message in cases:
            with self.subTest(layers=layers):
                with self.assertRaisesRegex(ValueError, message):
                    normalize_activation_checkpoint_layers("full", layers)
