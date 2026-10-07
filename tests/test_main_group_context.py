from __future__ import annotations

from types import SimpleNamespace

import pytest

from astrbot.api.message_components import Image
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


class _DummyProvider:
    def __init__(self, modalities: list[str] | None = None) -> None:
        self.provider_config = {"modalities": modalities} if modalities is not None else {}


class _DummyContext:
    def __init__(self, provider: object | None = None) -> None:
        self._provider = provider or _DummyProvider(["image"])

    def get_using_provider(self, *args, **kwargs) -> object:
        return self._provider


def _build_plugin(
    *,
    active_reply_mode: str = "probability",
    image_caption: bool = False,
    image_caption_provider_id: str = "",
    max_attached_images: int = 3,
    attached_images_scan_messages: int = 10,
    filter_memes: bool = True,
    meme_max_dimension: int = 400,
    modalities: list[str] | None = None,
) -> tuple[Main, PluginConfig]:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()
    plugin.context = _DummyContext(_DummyProvider(modalities if modalities is not None else ["image"]))
    cfg = PluginConfig(
        group_history=GroupHistoryEnhancementConfig(
            enable=True,
            image_caption=image_caption,
            image_caption_provider_id=image_caption_provider_id,
            max_attached_images=max_attached_images,
            attached_images_scan_messages=attached_images_scan_messages,
            filter_memes=filter_memes,
            meme_max_dimension=meme_max_dimension,
        ),
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


@pytest.mark.asyncio
async def test_auto_attach_recent_images_when_caption_disabled() -> None:
    plugin, _cfg = _build_plugin(
        image_caption=False,
        max_attached_images=2,
    )
    event = _DummyGroupEvent(message_id="3")
    origin = event.unified_msg_origin
    plugin.runtime.session_chats[origin].extend(
        [
            "[用户A/100/10:00:00] #msg1: 早先图片 [Image]",
            "[用户B/200/10:00:01] #msg2: 次新图片 [Image]",
            "[用户C/300/10:00:02] #msg3: 最新图片 [Image]",
        ]
    )
    plugin.runtime.image_message_registry[origin]["1"] = {
        "urls": ["https://example.com/img1.png"],
        "captions": {},
    }
    plugin.runtime.image_message_registry[origin]["2"] = {
        "urls": ["https://example.com/img2.png"],
        "captions": {},
    }
    plugin.runtime.image_message_registry[origin]["3"] = {
        "urls": ["https://example.com/img3.png"],
        "captions": {},
    }

    async def fake_resolve(ref: str) -> str:
        return f"/local/{ref.split('/')[-1]}"

    plugin._resolve_image_ref_to_local_path = fake_resolve

    req = ProviderRequest(prompt="最新图片 [Image]", contexts=[])
    await plugin.inject_group_context(event, req)

    # 应该只挂载最新的 2 张（img2, img3）
    assert req.image_urls == ["/local/img2.png", "/local/img3.png"]
    # 在注入的 prompt 中，早先图片依然是 [Image]，次新和最新标记为已挂载
    assert "早先图片 [Image]" in req.prompt
    assert "[Image: 原图已挂载#1]" in req.prompt
    assert "[Image: 原图已挂载#2]" in req.prompt


@pytest.mark.asyncio
async def test_no_auto_attach_when_caption_enabled() -> None:
    plugin, _cfg = _build_plugin(
        image_caption=True,
        image_caption_provider_id="caption-provider",
        max_attached_images=3,
    )
    event = _DummyGroupEvent(message_id="2")
    origin = event.unified_msg_origin
    plugin.runtime.session_chats[origin].extend(
        [
            "[用户A/100/10:00:00] #msg1: 图片1 [Image]",
            "[用户B/200/10:00:01] #msg2: 图片2 [Image]",
        ]
    )
    plugin.runtime.image_message_registry[origin]["1"] = {
        "urls": ["https://example.com/img1.png"],
        "captions": {},
    }
    plugin.runtime.image_message_registry[origin]["2"] = {
        "urls": ["https://example.com/img2.png"],
        "captions": {},
    }

    req = ProviderRequest(prompt="图片2 [Image]", contexts=[])
    await plugin.inject_group_context(event, req)

    # 转述模式下，不会自动挂载原图到 req.image_urls
    assert req.image_urls is None or req.image_urls == []
    # 历史消息中的 [Image] 保持占位符原样
    assert "原图已挂载" not in req.prompt
    assert "图片1 [Image]" in req.prompt


@pytest.mark.asyncio
async def test_attached_images_scan_messages_window() -> None:
    # 模拟历史中有 5 条消息：
    # msg1 (有图片)
    # msg2 (纯文字)
    # msg3 (纯文字)
    # msg4 (纯文字)
    # msg5 (当前消息，纯文字)
    plugin, _ = _build_plugin(
        image_caption=False,
        max_attached_images=3,
        attached_images_scan_messages=3,
    )
    event = _DummyGroupEvent(message_id="5")
    origin = event.unified_msg_origin
    plugin.runtime.session_chats[origin].extend(
        [
            "[用户A/100/10:00:00] #msg1: 旧图片消息 [Image]",
            "[用户B/200/10:00:01] #msg2: 纯文字2",
            "[用户C/300/10:00:02] #msg3: 纯文字3",
            "[用户D/400/10:00:03] #msg4: 纯文字4",
            "[用户E/500/10:00:04] #msg5: 当前文字5",
        ]
    )
    plugin.runtime.image_message_registry[origin]["1"] = {
        "urls": ["https://example.com/img1.png"],
        "captions": {},
    }

    async def fake_resolve(ref: str) -> str:
        return f"/local/{ref.split('/')[-1]}"

    plugin._resolve_image_ref_to_local_path = fake_resolve

    req = ProviderRequest(prompt="当前文字5", contexts=[])
    await plugin.inject_group_context(event, req)

    # 1. 扫描最近 3 条消息（msg3, msg4, msg5），无图片，因此不挂载任何图片
    assert req.image_urls == []
    assert "原图已挂载" not in req.prompt
    # 2. 但是全部 5 条消息依然完整注入 Prompt（历史文本注入不受 scan_messages 影响）
    assert "旧图片消息 [Image]" in req.prompt
    assert "纯文字2" in req.prompt
    assert "纯文字3" in req.prompt
    assert "纯文字4" in req.prompt
    assert "当前文字5" in req.prompt

    # 3. 如果把扫描窗口扩大到 5 条（包含 msg1）：
    plugin2, _ = _build_plugin(
        image_caption=False,
        max_attached_images=3,
        attached_images_scan_messages=5,
    )
    plugin2.runtime.session_chats[origin].extend(
        plugin.runtime.session_chats[origin]
    )
    plugin2.runtime.image_message_registry[origin]["1"] = {
        "urls": ["https://example.com/img1.png"],
        "captions": {},
    }
    plugin2._resolve_image_ref_to_local_path = fake_resolve

    req2 = ProviderRequest(prompt="当前文字5", contexts=[])
    await plugin2.inject_group_context(event, req2)

    # msg1 处于最近 5 条窗口内，图片被成功挂载
    assert req2.image_urls == ["/local/img1.png"]
    assert "[Image: 原图已挂载#1]" in req2.prompt


@pytest.mark.asyncio
async def test_filter_memes_in_history_and_attach() -> None:
    plugin, _ = _build_plugin(
        image_caption=False,
        max_attached_images=2,
        attached_images_scan_messages=10,
        filter_memes=True,
    )
    event = _DummyGroupEvent(message_id="3")
    origin = event.unified_msg_origin
    # 模拟历史：
    # msg1: 截图 (真实大图，非表情包)
    # msg2: 表情包 (被标记为 [Meme])
    # msg3: 当前文本消息
    plugin.runtime.session_chats[origin].extend(
        [
            "[用户A/100/10:00:00] #msg1: 发了张错误截图 [Image]",
            "[用户B/200/10:00:01] #msg2: 发了个搞笑表情包 [Meme]",
            "[用户C/300/10:00:02] #msg3: 帮我看下报错",
        ]
    )
    plugin.runtime.image_message_registry[origin]["1"] = {
        "urls": ["https://example.com/screenshot.png"],
        "resolved_paths": ["/local/screenshot.png"],
        "is_memes": [False],
        "captions": {},
    }
    plugin.runtime.image_message_registry[origin]["2"] = {
        "urls": ["https://example.com/meme.gif"],
        "resolved_paths": ["/local/meme.gif"],
        "is_memes": [True],
        "captions": {},
    }

    async def fake_resolve(ref: str) -> str:
        return f"/local/{ref.split('/')[-1]}"

    plugin._resolve_image_ref_to_local_path = fake_resolve

    req = ProviderRequest(prompt="帮我看下报错", contexts=[])
    await plugin.inject_group_context(event, req)

    # 1. 自动跳过表情包 msg2，只挂载截图 msg1
    assert req.image_urls == ["/local/screenshot.png"]
    # 2. 截图替换为 [Image: 原图已挂载#1]，而表情包保持 [Meme] 标记
    assert "[Image: 原图已挂载#1]" in req.prompt
    assert "发了个搞笑表情包 [Meme]" in req.prompt


def test_is_meme_image_detection() -> None:
    plugin, cfg = _build_plugin(filter_memes=True, meme_max_dimension=400)

    # 1. NapCat subType = 1 (自定义表情包)
    event_meme = _DummyGroupEvent(message_id="10")
    event_meme.message_obj.raw_message = {
        "message": [{"type": "image", "data": {"file": "meme1.jpg", "subType": "1"}}]
    }
    comp_meme = Image(file="meme1.jpg")
    assert plugin._is_meme_image(comp_meme, "", event_meme, cfg) is True

    # 2. NapCat summary = [动画表情]
    event_anim = _DummyGroupEvent(message_id="11")
    event_anim.message_obj.raw_message = {
        "message": [{"type": "image", "data": {"file": "anim.jpg", "subType": "0", "summary": "[动画表情]"}}]
    }
    comp_anim = Image(file="anim.jpg")
    assert plugin._is_meme_image(comp_anim, "", event_anim, cfg) is True

    # 3. NapCat subType = 0 (普通图片 / 截图)
    event_real = _DummyGroupEvent(message_id="12")
    event_real.message_obj.raw_message = {
        "message": [{"type": "image", "data": {"file": "screen.png", "subType": "0", "summary": "[图片]"}}]
    }
    comp_real = Image(file="screen.png")
    # 无本地文件时，仅根据协议判定为非表情包
    assert plugin._is_meme_image(comp_real, "", event_real, cfg) is False

    # 4. 本地文件为 .gif 动图
    assert plugin._is_meme_image(comp_real, "/path/to/funny.gif", event_real, cfg) is True

