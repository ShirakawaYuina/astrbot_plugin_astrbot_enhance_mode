from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest

from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    ActiveReplyConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState


class _DummyGroupEvent:
    def __init__(
        self,
        *,
        message_id: str,
        active_reply_triggered: bool = False,
        active_reply_mode: str = "",
    ) -> None:
        self.unified_msg_origin = "test:group:100"
        self.message_obj = SimpleNamespace(message_id=message_id)
        self._extras = {
            "_enhance_active_reply_triggered": active_reply_triggered,
            "_enhance_active_reply_mode": active_reply_mode,
        }
        self.stopped = False

    def get_message_type(self) -> MessageType:
        return MessageType.GROUP_MESSAGE

    def get_extra(self, key: str, default=None):  # noqa: ANN001
        return self._extras.get(key, default)

    def stop_event(self) -> None:
        self.stopped = True


def _build_plugin(*, active_reply_mode: str = "probability") -> tuple[Main, PluginConfig]:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()
    cfg = PluginConfig(
        group_history=GroupHistoryEnhancementConfig(enable=True),
        active_reply=ActiveReplyConfig(enable=True, mode=active_reply_mode),
        group_features=GroupFeatureEnhancementConfig(
            react_mode_enable=True,
            mention_parse=True,
        ),
    )
    plugin._cfg = lambda: cfg
    return plugin, cfg


@pytest.mark.asyncio
async def test_normal_reply_injects_current_once_and_excludes_future_messages() -> None:
    plugin, _cfg = _build_plugin()
    event = _DummyGroupEvent(message_id="2")
    plugin.runtime.session_chats[event.unified_msg_origin].extend(
        [
            "[旧用户/100/10:00:00] #msg1: 旧消息",
            "[当前用户/200/10:00:01] #msg2: 当前消息",
            "[后续用户/300/10:00:02] #msg3: 后续消息",
        ]
    )
    req = ProviderRequest(prompt="当前消息", contexts=[{"role": "user", "content": "旧会话"}])

    await plugin.inject_group_context(event, req)

    assert req.prompt.count("当前消息") == 1
    assert req.prompt.count("#msg2") == 1
    assert "旧消息" in req.prompt
    assert "后续消息" not in req.prompt
    assert req.contexts == []


@pytest.mark.asyncio
async def test_model_choice_sees_history_through_trigger_but_not_future_messages() -> None:
    plugin, _cfg = _build_plugin(active_reply_mode="model_choice")
    event = _DummyGroupEvent(
        message_id="2",
        active_reply_triggered=True,
        active_reply_mode="model_choice",
    )
    plugin.runtime.session_chats[event.unified_msg_origin].extend(
        [
            "[旧用户/100/10:00:00] #msg1: 旧消息",
            "[触发用户/200/10:00:01] #msg2: 触发消息",
            "[后续用户/300/10:00:02] #msg3: 后续消息",
        ]
    )
    req = ProviderRequest(prompt="触发消息", contexts=[])

    await plugin.inject_group_context(event, req)

    assert "旧消息" in req.prompt
    assert "触发消息" in req.prompt
    assert "后续消息" not in req.prompt


@pytest.mark.asyncio
async def test_boundary_error_does_not_inject_unbounded_history() -> None:
    plugin, _cfg = _build_plugin()
    event = _DummyGroupEvent(message_id="missing")
    plugin.runtime.session_chats[event.unified_msg_origin].append(
        "[其他用户/100/10:00:00] #msg1: 不应注入的消息"
    )
    req = ProviderRequest(prompt="原始提示词", contexts=[])

    await plugin.inject_group_context(event, req)

    assert req.prompt == "原始提示词"
    assert "不应注入的消息" not in req.prompt
    assert event.stopped is True
