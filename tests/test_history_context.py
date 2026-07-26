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
