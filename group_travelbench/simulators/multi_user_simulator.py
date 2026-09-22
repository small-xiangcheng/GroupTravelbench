"""
Multi-user simulator for the multi-person travel planning scenario.
Each instance simulates one user in the group chat.
"""

from typing import Dict, Any, List, Optional

from ..core.config import OpenAIConfig
from ..core.openai_client import OpenAIClient
from ..core.messages import SystemMessage, UserMessage, AssistantMessage, Message
from ..utils.multi_util import build_user_messages
from .multi_prompt import MULTI_USER_USER_SYSTEM_PROMPT


class MultiUserSimulator:
    """
    Simulates a single user in a multi-user travel planning group chat.
    
    Each user has their own system prompt (with user_preference embedded),
    and shares the same conversation context with all other participants.

    Model choice: this is the demanding role. The simulator has to stay in
    character, react to the group, and know when to compromise, so it MUST be a
    strong thinking model — a weak / non-thinking model degrades simulation
    quality badly. Thinking must stay ON (this work uses DeepSeek-V4-Flash).
    """

    def __init__(
        self,
        config: OpenAIConfig,
        user_name: str,
        user_preference: str,
        query: str,
        time: str = "",
        user_list_description: str = "",
        compromisable: bool = False,
    ):
        self.config = config
        self.client = OpenAIClient(config)
        self.user_name = user_name
        self.user_preference = user_preference
        self.query = query
        self.time = time
        self.user_list_description = user_list_description
        # When compromisable=True:
        #   - the user still speaks at their natural preference strength (must stays firm)
        #   - but they may yield when the Agent @-asks "can you accept X / drop Y"
        #   - they must not volunteer a compromise unless asked
        # The flag is mutable: when this user has exhausted their per-user
        # compromise quota, the orchestrator flips it to False and calls
        # `set_compromisable(False)` to rebuild the system prompt accordingly.
        self.compromisable = compromisable

        self.system_prompt = self._build_system_prompt()

    def set_compromisable(self, compromisable: bool) -> None:
        """Update compromisable flag and rebuild the system prompt.

        The system prompt is captured once at __init__; if the orchestrator
        mutates `self.compromisable` directly without calling this method,
        the live system prompt will be stale.
        """
        if self.compromisable == compromisable:
            return
        self.compromisable = compromisable
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for this user."""
        if self.compromisable:
            compromise_block = (
                "你的妥协态度：可妥协。\n"
                "- 你**仍然按偏好的真实强度表达**（must 类偏好仍然要用强硬语气说，避免一上来就让步）。\n"
                "- 但当 Agent 主动 @ 你并询问『你能不能接受 X』、『能不能放弃/换成 Y』、"
                "『…能不能将就一下』之类的协商性问题时，**你倾向于同意让步**：\n"
                "  - 哪怕被问的是你之前说过的 must 偏好，也可以表态接受 Agent 的方案；\n"
                "  - 表达上要自然，例如『行吧，那就听你的』『可以，我让一下』『没事，我能接受』；\n"
                "  - 不要每次都解释一长串，简短同意即可。\n"
                "- **没有被 Agent 主动询问的情况下，严禁主动放弃自己的偏好**——"
                "不要在自由发言时突然说『我改主意了』。"
            )
        else:
            compromise_block = (
                "你的妥协态度：不可妥协。\n"
                "- 你的 must / reject 类偏好就是底线，即使被 Agent @ 询问『能不能接受』，"
                "也要明确拒绝（『不行，这个我之前说了一定要…』）。\n"
                "- 弱档（prefer / avoid）可以在被询问时温和让步，但强档不可以。"
            )

        return MULTI_USER_USER_SYSTEM_PROMPT.format(
            user_name=self.user_name,
            user_list_description=self.user_list_description,
            query=self.query,
            time=self.time,
            user_preference=self.user_preference,
            compromise_block=compromise_block,
        )

    @staticmethod
    def check_is_pass(text: str) -> bool:
        """Check if a user reply is the [pass] signal.

        `[pass]` is exclusively a user-side mechanism: the Agent has no
        pass option, so this check should only ever be applied to user
        outputs. The signal must be the *entire* (trimmed) message — a
        message containing extra text alongside `[pass]` is treated as a
        normal substantive reply, not a pass.
        """
        return text.strip() == "[pass]"

    def generate_response(
        self,
        conversation_history: List[Dict[str, Any]],
        is_being_mentioned: bool = False,
        extra_instruction: Optional[str] = None,
    ) -> str:
        """
        Generate a response during the free interaction phase.

        Message construction follows the OpenAI multi-turn convention from
        *this user's perspective*:
          - This user's own prior messages → AssistantMessage
          - All other participants' messages (Agent + other Users) merged
            into UserMessage entries with "RoleName: content" lines.
          - Agent intermediate reasoning and tool entries are skipped
            (Users never see tool details).

        This ensures the model sees its own prior utterances in the
        ``assistant`` role, reinforcing role consistency and preventing
        confusion about "who am I" in long conversations.

        Args:
            conversation_history: The global structured conversation history.
            is_being_mentioned: Whether this user was @mentioned and must reply directly.
            extra_instruction: Optional extra instruction to append to the user message.

        Returns:
            The user's response text.
        """
        system_content = self.system_prompt
        if is_being_mentioned:
            system_content += (
                "\n\n【重要提醒】你刚刚被 @ 了，你必须直接回答问题。"
                "不能输出 [pass]，也不能再 @别人。"
            )

        # Build structured message list from this user's perspective
        history_messages = build_user_messages(
            conversation_history, self_role=self.user_name
        )

        # If extra_instruction is provided, append it to the last
        # UserMessage or create a new one.
        if extra_instruction and history_messages:
            last_msg = history_messages[-1]
            if isinstance(last_msg, UserMessage):
                existing_content = last_msg.content or ""
                history_messages[-1] = UserMessage(
                    content=existing_content + f"\n\n{extra_instruction}"
                )
            else:
                history_messages.append(
                    UserMessage(content=extra_instruction)
                )
        elif extra_instruction:
            history_messages.append(
                UserMessage(content=extra_instruction)
            )

        messages: List[Message] = [
            SystemMessage(content=system_content),
        ] + history_messages

        response, _ = self.client.generate_response(messages=messages)
        return (response.content or "").strip()
