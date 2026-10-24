from __future__ import annotations

import pytest

from astrbot_plugin_astrbot_enhance_mode.history_context import (
    HistoryBoundaryError,
    build_current_message_reply_prompt,
    format_quote_for_history,
    history_retained_before_append,
    split_history_at_message,
)


def test_split_history_excludes_current_and_following_messages() -> None:
    history_lines = [
        "[旧用户/100/10:00:00] #msg1: 旧消息",
        "[当前用户/200/10:00:01] #msg2: 当前消息",
        "[后续用户/300/10:00:02] #msg3: 后续消息 B",
        "[后续用户/400/10:00:03] #msg4: 后续消息 C",
    ]

    boundary = split_history_at_message(history_lines, "2")

    assert boundary.history_before == (history_lines[0],)
    assert boundary.current_message == history_lines[1]
    assert boundary.history_through_current == tuple(history_lines[:2])
    assert boundary.excluded_after_count == 2


def test_split_history_uses_latest_duplicate_message_id() -> None:
    history_lines = [
        "[用户/100/10:00:00] #msg7: 首次记录",
        "[You/10:00:01]: 中间回复",
        "[用户/100/10:00:02] #msg7: 最近记录",
        "[其他用户/200/10:00:03] #msg8: 后续消息",
    ]

    boundary = split_history_at_message(history_lines, "#msg7")

    assert boundary.history_before == tuple(history_lines[:2])
    assert boundary.current_message == history_lines[2]
    assert boundary.excluded_after_count == 1


@pytest.mark.parametrize("message_id", ["", "missing"])
def test_split_history_rejects_missing_boundary(message_id: str) -> None:
    history_lines = ["[用户/100/10:00:00] #msg1: 消息"]

    with pytest.raises(HistoryBoundaryError):
        split_history_at_message(history_lines, message_id)


def test_current_message_reply_prompt_contains_target_once() -> None:
    prompt = build_current_message_reply_prompt(
        history_text="=== CHAT_HISTORY_BEGIN ===\n旧消息\n=== CHAT_HISTORY_END ===",
        current_message="[当前用户/200/10:00:01] #msg2: 唯一目标正文",
        interaction_instructions="\n交互规则",
    )

    assert prompt.count("唯一目标正文") == 1
    assert prompt.count("#msg2") == 1
    assert "Current message to reply to:" in prompt
    assert "Reply to the current message above." in prompt
    assert "[Context Rules]" in prompt
    assert "[You/...]" in prompt


def test_format_quote_omits_text_when_target_is_in_history() -> None:
    history_lines = ["[被引用用户/100/10:00:00] #msg1: 被引用原文"]

    quote = format_quote_for_history(
        quote_message_id="1",
        quote_nickname="被引用用户",
        quote_text="被引用原文",
        history_lines=history_lines,
    )

    assert quote == " [Quote #msg1]"


def test_format_quote_keeps_text_when_target_is_not_in_history() -> None:
    quote = format_quote_for_history(
        quote_message_id="1",
        quote_nickname="被引用用户",
        quote_text="被引用原文",
        history_lines=[],
    )

    assert quote == " [Quote #msg1 被引用用户: 被引用原文]"


def test_format_quote_without_message_id_keeps_text() -> None:
    quote = format_quote_for_history(
        quote_message_id="",
        quote_nickname="被引用用户",
        quote_text="被引用原文",
        history_lines=[],
    )

    assert quote == " [Quote 被引用用户: 被引用原文]"


def test_quote_keeps_text_when_target_will_be_evicted_by_current_message() -> None:
    history_lines = [
        "[最旧用户/100/10:00:00] #msg1: 即将淘汰的原文",
        "[最近用户/200/10:00:01] #msg2: 最近消息",
    ]
    retained_history = history_retained_before_append(
        history_lines,
        max_messages=2,
    )

    quote = format_quote_for_history(
        quote_message_id="1",
        quote_nickname="最旧用户",
        quote_text="即将淘汰的原文",
        history_lines=retained_history,
    )

    assert retained_history == (history_lines[1],)
    assert quote == " [Quote #msg1 最旧用户: 即将淘汰的原文]"


def test_interaction_instructions_no_refuse_and_natural_quote() -> None:
    from astrbot_plugin_astrbot_enhance_mode.tag_utils import build_interaction_instructions
    instructions = build_interaction_instructions(mention_parse=True, include_sender_id=True)
    assert "Refuse" not in instructions
    assert "<refuse/>" not in instructions
    assert "DO NOT quote every single message" in instructions
    assert "DO NOT mention the user" in instructions


def test_normalize_quote_id() -> None:
    from astrbot_plugin_astrbot_enhance_mode.tag_utils import normalize_quote_id
    assert normalize_quote_id("#msg57871096") == "57871096"
    assert normalize_quote_id("[#msg57871096]") == "57871096"
    assert normalize_quote_id("msg12345") == "12345"
    assert normalize_quote_id("#12345") == "12345"
    assert normalize_quote_id("12345") == "12345"
    assert normalize_quote_id("") == ""
    assert normalize_quote_id(None) == ""


def test_transform_result_chain_citation_compatibility() -> None:
    from astrbot.api.message_components import Plain, Reply
    from astrbot_plugin_astrbot_enhance_mode.tag_utils import transform_result_chain

    # 1. <citation>#msg12345</citation> 容器标签转换
    chain = [Plain(text="<citation>#msg57871096</citation> 好的收到")]
    transformed = transform_result_chain(chain, parse_mention=True)
    assert transformed is not None
    assert len(transformed) == 2
    assert isinstance(transformed[0], Reply)
    assert transformed[0].id == "57871096"
    assert isinstance(transformed[1], Plain)
    assert transformed[1].text == "好的收到"

    # 2. <citation id="57871096"/> 自闭合标签转换
    chain2 = [Plain(text='<citation id="57871096"/>这也算引用')]
    transformed2 = transform_result_chain(chain2, parse_mention=True)
    assert transformed2 is not None
    assert isinstance(transformed2[0], Reply)
    assert transformed2[0].id == "57871096"
    assert transformed2[1].text == "这也算引用"


def test_transform_result_chain_strips_meta_chip() -> None:
    from astrbot.api.message_components import Plain
    from astrbot_plugin_astrbot_enhance_mode.tag_utils import transform_result_chain

    chain = [Plain(text='摸摸头，别慌～<meta chip="img3"/> 挨批也是暂时的')]
    transformed = transform_result_chain(chain, parse_mention=True)
    assert transformed is not None
    assert len(transformed) == 1
    assert transformed[0].text == "摸摸头，别慌～ 挨批也是暂时的"
    assert "<meta" not in transformed[0].text
    assert "img3" not in transformed[0].text


def test_clean_response_text_strips_citation_and_meta() -> None:
    from astrbot_plugin_astrbot_enhance_mode.tag_utils import clean_response_text_for_history

    raw = '<citation>#msg57871096</citation> 摸摸头～<meta chip="img3"/> 别慌'
    cleaned = clean_response_text_for_history(raw)
    assert cleaned == "摸摸头～ 别慌"
    assert "citation" not in cleaned
    assert "meta" not in cleaned
    assert "img3" not in cleaned
