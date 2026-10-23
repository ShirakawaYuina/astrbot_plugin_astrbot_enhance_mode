import re

from astrbot.api.message_components import At, Plain, Reply

MENTION_RE = re.compile(
    r"""<mention\s+id\s*=\s*['"]([^'"]+)['"]\s*/?>""",
    re.IGNORECASE,
)
QUOTE_RE = re.compile(
    r"""<quote\s+id\s*=\s*['"]([^'"]+)['"]\s*/?>""",
    re.IGNORECASE,
)
MENTION_CLOSE_RE = re.compile(r"</mention\s*>", re.IGNORECASE)
QUOTE_CLOSE_RE = re.compile(r"</quote\s*>", re.IGNORECASE)


def normalize_quote_id(raw_id: str | None) -> str:
    """标准化平台引用消息 ID。

    Args:
        raw_id: 原始消息 ID 字符串或整数。

    Returns:
        去除前缀与两端空白后的标准化 ID 字符串。
    """
    if not raw_id:
        return ""
    quote_id = str(raw_id).strip()
    if quote_id.startswith("#"):
        quote_id = quote_id[1:]
    if quote_id.lower().startswith("msg"):
        quote_id = quote_id[3:]
    return quote_id.strip()


def transform_result_chain(chain: list, parse_mention: bool) -> list | None:
    """解析回复链中的控制标签（Mention 和 Quote），转换为平台原生消息组件。

    Args:
        chain: 待转换的消息组件列表。
        parse_mention: 是否启用 Mention 标签解析。

    Returns:
        转换后的新消息组件列表；若无相关标签则返回 None。
    """
    quote_msg_id = None
    if any(isinstance(comp, Plain) and QUOTE_RE.search(comp.text) for comp in chain):
        for comp in chain:
            if isinstance(comp, Plain):
                matched = QUOTE_RE.search(comp.text)
                if matched:
                    quote_msg_id = normalize_quote_id(matched.group(1))
                    break

    has_tags = any(
        isinstance(comp, Plain)
        and (
            QUOTE_RE.search(comp.text)
            or QUOTE_CLOSE_RE.search(comp.text)
            or MENTION_CLOSE_RE.search(comp.text)
            or (parse_mention and MENTION_RE.search(comp.text))
        )
        for comp in chain
    )
    if not has_tags and not quote_msg_id:
        return None

    new_chain = []
    for comp in chain:
        if not isinstance(comp, Plain):
            new_chain.append(comp)
            continue

        text = QUOTE_RE.sub("", comp.text)
        text = QUOTE_CLOSE_RE.sub("", text)

        if parse_mention and MENTION_RE.search(text):
            parts = MENTION_RE.split(text)
            for idx, part in enumerate(parts):
                if idx % 2 == 0:
                    cleaned_text = MENTION_CLOSE_RE.sub("", part)
                    if cleaned_text.strip():
                        new_chain.append(Plain(text=cleaned_text))
                else:
                    new_chain.append(At(qq=part))
        else:
            text = MENTION_CLOSE_RE.sub("", text)
            if text.strip():
                new_chain.append(Plain(text=text))

    if quote_msg_id:
        new_chain.insert(0, Reply(id=quote_msg_id))

    return new_chain


def clean_response_text_for_history(completion_text: str) -> str:
    """清理机器人回复文本中的内部控制标签，生成写入历史的纯文本。

    Args:
        completion_text: 模型生成的原始文本。

    Returns:
        去除标签后的纯文本字符串。
    """
    text = MENTION_RE.sub(r"[At: \1]", completion_text)
    text = MENTION_CLOSE_RE.sub("", text)
    text = QUOTE_RE.sub("", text)
    text = QUOTE_CLOSE_RE.sub("", text)
    return text.strip()


def build_interaction_instructions(
    mention_parse: bool,
    include_sender_id: bool,
) -> str:
    """构造群聊交互控制指令（Mention / Quote），引导模型自然发言。

    Args:
        mention_parse: 是否开启 Mention 标签解析。
        include_sender_id: 群聊历史中是否包含发送者 ID。

    Returns:
        注入到提示词中的交互说明文本。
    """
    instructions = ""
    if mention_parse and include_sender_id:
        instructions += (
            "\n\n## Mention\n"
            'Use `<mention id="user_id"/>` ONLY when you specifically need to call out or address a particular user in a busy chatroom.\n'
            "In normal ongoing conversation, DO NOT mention the user. Chat naturally like a human group member without unnecessary @.\n"
            'Format: <mention id="123456"/> Hello!\n'
            "The user_id can be found in the chat history format [nickname/user_id/time]. Do NOT use this for yourself, and do NOT output closing tag </mention>."
        )

    instructions += (
        "\n\n## Quote\n"
        'Use `<quote id="msg_id"/>` ONLY when clearly necessary (e.g. replying to an older message from earlier, clarifying your target among interleaved conversations, or specifically countering a point).\n'
        "In normal conversational flow, DO NOT quote every single message. Prefer direct, natural replies without quote.\n"
        'If quoting, place it at the very beginning: `<quote id="12345"/> I agree with this!\n'
        "The msg_id can be found in the chat history after #msg (e.g. #msg12345). Only quote at most ONE message per reply, and do NOT output closing tag </quote>."
    )
    return instructions


def bounded_chat_history_text(messages: list[str]) -> str:
    """将群聊消息列表拼接为带明确边界标记的历史文本块。

    Args:
        messages: 群聊历史记录单行文本列表。

    Returns:
        包装了边界起始与结束标记的历史字符串。
    """
    chats_str = "\n---\n".join(messages)
    return f"=== CHAT_HISTORY_BEGIN ===\n{chats_str}\n=== CHAT_HISTORY_END ==="
