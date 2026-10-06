from __future__ import annotations

import json
from typing import Any

from mcp import types as mcp_types
import pytest

from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    GlobalSettingsConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState


class _DummyEvent:
    def __init__(self, origin: str) -> None:
        self.unified_msg_origin = origin


class _DummyProvider:
    def __init__(self, modalities: list[str] | None = None) -> None:
        self.provider_config = {"modalities": modalities} if modalities is not None else {}


class _DummyContext:
    def __init__(self, provider: Any = None) -> None:
        self._provider = provider or _DummyProvider(["image"])

    def get_using_provider(self, *args: Any, **kwargs: Any) -> Any:
        return self._provider


def _build_plugin(
    *,
    image_caption: bool,
    image_caption_provider_id: str = "caption-provider",
    modalities: list[str] | None = None,
) -> tuple[Main, _DummyEvent]:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()
    plugin.context = _DummyContext(_DummyProvider(modalities if modalities is not None else ["image"]))
    cfg = PluginConfig(
        group_history=GroupHistoryEnhancementConfig(
            enable=True,
            image_caption=image_caption,
            image_caption_provider_id=image_caption_provider_id if image_caption else "",
        ),
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        global_settings=GlobalSettingsConfig(),
    )
    plugin._cfg = lambda: cfg
    return plugin, _DummyEvent("origin-1")


def _payload_from_results(
    results: list[mcp_types.CallToolResult],
) -> dict[str, object]:
    return json.loads(results[-1].content[0].text)


@pytest.mark.asyncio
async def test_use_image_caption_disabled_attaches_image_without_calling_caption() -> None:
    plugin, event = _build_plugin(image_caption=False)

    async def should_not_be_called(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("_get_image_caption should not be called when caption is disabled")

    async def resolve_local_path(_image_ref: str) -> str:
        return "/tmp/fake-image.png"

    plugin._get_image_caption = should_not_be_called
    plugin._resolve_image_ref_to_local_path = resolve_local_path
    plugin._encode_image_file = lambda _path: ("ZmFrZQ==", "image/png")

    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
    ):
        results.append(item)

    assert len(results) == 2
    image_content = results[0].content[0]
    assert isinstance(image_content, mcp_types.ImageContent)
    assert image_content.mimeType == "image/png"
    assert image_content.data == "ZmFrZQ=="
    payload = _payload_from_results(results)
    assert payload["status"] == "ok"
    assert payload["mode"] == "attach_image"
    assert payload["attach_success"] is True


@pytest.mark.asyncio
async def test_use_image_caption_enabled_writes_history_without_attaching_image() -> None:
    plugin, event = _build_plugin(image_caption=True)
    applied: dict[str, object] = {}

    async def get_image_caption(*args: Any, **kwargs: Any) -> str:
        return "A test caption"

    async def should_not_resolve(_image_ref: str) -> str:
        raise AssertionError("_resolve_image_ref_to_local_path should not be called in caption mode")

    def apply_caption_to_history(**kwargs: Any) -> bool:
        applied.update(kwargs)
        return True

    plugin._get_image_caption = get_image_caption
    plugin._resolve_image_ref_to_local_path = should_not_resolve
    plugin._apply_image_caption_to_history = apply_caption_to_history

    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(event=event, message_id="123", image_index=1):
        results.append(item)

    # 仅返回文本 payload，不 yield ImageContent
    assert len(results) == 1
    assert isinstance(results[0].content[0], mcp_types.TextContent)
    payload = _payload_from_results(results)
    assert payload["status"] == "ok"
    assert payload["mode"] == "caption"
    assert payload["description"] == "A test caption"
    assert payload["write_to_history_success"] is True
    assert (
        plugin.runtime.image_message_registry[event.unified_msg_origin]["123"]["captions"][0]
        == "A test caption"
    )
    assert applied["message_id"] == "123"
    assert applied["image_index"] == 0
    assert applied["caption"] == "A test caption"


@pytest.mark.asyncio
async def test_use_image_caption_disabled_fails_when_model_does_not_support_image() -> None:
    plugin, event = _build_plugin(image_caption=False, modalities=["text"])
    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
    ):
        results.append(item)

    assert len(results) == 1
    assert "当前对话模型不支持图像输入" in results[0].content[0].text


@pytest.mark.asyncio
async def test_use_image_returns_not_found_when_message_id_is_missing() -> None:
    plugin, event = _build_plugin(image_caption=False)

    results = []
    async for item in plugin.use_image(event=event, message_id="not-exist", image_index=1):
        results.append(item)

    assert len(results) == 1
    assert "not found in current runtime history" in results[0].content[0].text


@pytest.mark.asyncio
async def test_use_image_returns_error_when_image_index_out_of_range() -> None:
    plugin, event = _build_plugin(image_caption=False)
    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(event=event, message_id="123", image_index=2):
        results.append(item)

    assert len(results) == 1
    assert "`image_index` out of range" in results[0].content[0].text
