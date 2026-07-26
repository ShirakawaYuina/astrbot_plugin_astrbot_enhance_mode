import re
from dataclasses import dataclass

MESSAGE_ID_PATTERN = re.compile(r"#msg([^:]+):")


class HistoryBoundaryError(ValueError):
    """表示无法根据平台消息 ID 建立可靠的群聊历史边界。"""


@dataclass(frozen=True, slots=True)
class HistoryBoundary:
    """保存一次 LLM 请求对应的群聊历史边界切分结果。"""

    history_before: tuple[str, ...]
    current_message: str
    history_through_current: tuple[str, ...]
    excluded_after_count: int


def normalize_message_id(raw_message_id: str | int | None) -> str:
    """统一平台消息 ID 与历史标签 ID 的外观，供精确比较使用。"""

    message_id = str(raw_message_id or "").strip()
    if message_id.startswith("#"):
        message_id = message_id[1:]
    if message_id.lower().startswith("msg"):
        message_id = message_id[3:]
    if message_id.endswith(":"):
        message_id = message_id[:-1]
    return message_id.strip()


def extract_message_id_from_history_line(history_line: str) -> str:
    """从历史消息头的 ``#msgID:`` 标记中提取消息 ID。"""

    matched = MESSAGE_ID_PATTERN.search(str(history_line or ""))
    if not matched:
        return ""
    return normalize_message_id(matched.group(1))


def split_history_at_message(
    history_lines: list[str] | tuple[str, ...],
    current_message_id: str | int | None,
) -> HistoryBoundary:
    """按当前消息 ID 切分历史，排除本轮请求之后到达的消息。

    从后向前定位是为了处理平台重投或异常重复记录：同一消息 ID 多次出现时，
    最近一次记录才是当前请求应使用的时间边界。
    """

    normalized_current_id = normalize_message_id(current_message_id)
    if not normalized_current_id:
        raise HistoryBoundaryError("当前消息缺少 message_id，无法建立群聊历史边界。")

    history_snapshot = tuple(history_lines)
    for current_index in range(len(history_snapshot) - 1, -1, -1):
        history_message_id = extract_message_id_from_history_line(
            history_snapshot[current_index]
        )
        if history_message_id != normalized_current_id:
            continue

        return HistoryBoundary(
            history_before=history_snapshot[:current_index],
            current_message=history_snapshot[current_index],
            history_through_current=history_snapshot[: current_index + 1],
            excluded_after_count=len(history_snapshot) - current_index - 1,
        )

    raise HistoryBoundaryError(
        f"群聊历史中找不到当前消息 #msg{normalized_current_id}，无法建立请求边界。"
    )


def history_contains_message_id(
    history_lines: list[str] | tuple[str, ...],
    message_id: str | int | None,
) -> bool:
    """判断指定消息 ID 的原始消息是否仍位于当前会话历史窗口内。"""

    normalized_message_id = normalize_message_id(message_id)
    if not normalized_message_id:
        return False
    return any(
        extract_message_id_from_history_line(history_line) == normalized_message_id
        for history_line in history_lines
    )


def history_retained_before_append(
    history_lines: list[str] | tuple[str, ...],
    max_messages: int,
) -> tuple[str, ...]:
    """返回追加一条新消息后仍会保留的既有历史。

    当前消息尚未追加，但引用去重必须按追加完成后的最终窗口判断；否则被引用原消息
    恰好即将淘汰时，省略引用全文会造成模型只看到一个无法解析的引用 ID。
    """

    retained_existing_count = max(0, int(max_messages) - 1)
    if retained_existing_count == 0:
        return ()
    return tuple(history_lines[-retained_existing_count:])


def format_quote_for_history(
    *,
    quote_message_id: str | int | None,
    quote_nickname: str,
    quote_text: str,
    history_lines: list[str] | tuple[str, ...],
) -> str:
    """格式化引用组件，并避免重复注入仍在历史中的原消息全文。"""

    normalized_quote_id = normalize_message_id(quote_message_id)
    if normalized_quote_id and history_contains_message_id(
        history_lines, normalized_quote_id
    ):
        return f" [Quote #msg{normalized_quote_id}]"
    if normalized_quote_id:
        return f" [Quote #msg{normalized_quote_id} {quote_nickname}: {quote_text}]"
    return f" [Quote {quote_nickname}: {quote_text}]"


def build_current_message_reply_prompt(
    *,
    history_text: str,
    current_message: str,
    interaction_instructions: str,
) -> str:
    """构造普通回复提示词，确保当前回复目标只出现一次。"""

    return (
        "You are now in a chatroom. "
        f"The chat history before the current message is as follows:\n{history_text}\n\n"
        f"Current message to reply to:\n{current_message}\n\n"
        "Reply to the current message above. Your entire output is your reply to this message. "
        "Quote the current message in most cases. "
        "Only output your response and do not output any other information. "
        "You MUST use the SAME language as the chatroom is using."
        f"{interaction_instructions}"
    )
