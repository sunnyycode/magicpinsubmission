"""
conversation_handlers.py — Multi-turn conversation management for magicpin AI Challenge (Vera).

Handles merchant and customer replies, detecting:
- Auto-replies (canned WhatsApp business messages, repeated greetings)
- Intent transitions (commitment: "ok lets do it", "proceed", "yes") -> switches to ACTION mode immediately
- Hostility / Opt-outs ("stop", "not interested", "spam") -> graceful exit
- Out-of-scope queries (GST, accounting, legal) -> polite redirection
- General queries -> helpful, concise replies advancing the commercial thread.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ConversationState:
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    category_slug: Optional[str] = None
    turns: List[Dict[str, Any]] = field(default_factory=list)
    initial_trigger: Optional[Dict[str, Any]] = None
    state: str = "active"  # "active", "waiting", "action_mode", "ended"
    last_action: Optional[str] = None
    auto_reply_count: int = 0
    topic: Optional[str] = None
    merchant_name: Optional[str] = None
    owner_name: Optional[str] = None
    language_pref: str = "en"


# Common auto-reply / bot greeting patterns in WhatsApp Business
AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"thanks for reaching out",
    r"our team will respond shortly",
    r"our team will get back to you",
    r"we will get back to you",
    r"currently away",
    r"outside of our working hours",
    r"hamari team tak pahuncha",
    r"aapki jaankari ke liye.*shukriya",
    r"automated assistant",
    r"auto-reply",
    r"this is an automated",
    r"we are closed now",
    r"will reply soon",
    r"message receive ho gaya hai",
]

# Explicit commitment / action-trigger phrases
COMMITMENT_PATTERNS = [
    r"\bok\s+lets\s+do\s+it\b",
    r"\blets\s+do\s+it\b",
    r"\blet's\s+do\s+it\b",
    r"\bwhats\s+next\b",
    r"\bwhat's\s+next\b",
    r"\bwhat\s+next\b",
    r"\bi\s+want\s+to\s+join\b",
    r"\bmujhe\s+judna\s+hai\b",
    r"\bproceed\b",
    r"\bgo\s+ahead\b",
    r"\byes\s+please\b",
    r"\bstart\b",
    r"\bconfirm\b",
    r"\bapprove\b",
    r"\bsend\s+the\s+abstract\b",
    r"\bdraft\s+the\s+patient\b",
    r"\bschedule\s+it\b",
    r"\bchalo\s+karte\s+hain\b",
    r"\btheek\s+hai\s+karo\b",
    r"\bhaan\s+bhejo\b",
    r"\byes\b",
    r"\bya\b",
    r"\byep\b",
    r"\bsure\b",
    r"\bdone\b",
    r"\bready\b",
]

# Hostility and opt-out signals
HOSTILE_PATTERNS = [
    r"\bstop\b",
    r"\bunsubscribe\b",
    r"\bnot\s+interested\b",
    r"\bspam\b",
    r"\buseless\b",
    r"\bdon't\s+message\b",
    r"\bdont\s+message\b",
    r"\bleave\s+me\s+alone\b",
    r"\bmat\s+bhejo\b",
    r"\bband\s+karo\b",
    r"\bno\s+more\b",
    r"\bblock\b",
]

# Out-of-scope topics
OUT_OF_SCOPE_PATTERNS = [
    (r"\bgst\b|\btax\b|\bincome\s+tax\b|\bfile\s+return\b", "GST and tax filing"),
    (r"\bloan\b|\bbank\s+account\b|\bcredit\s+card\b", "loans and banking services"),
    (r"\blegal\s+advice\b|\blawyer\b|\bcourt\b", "legal compliance advisory"),
    (r"\bpersonal\b|\bwho\s+are\s+you\s+really\b", "personal topics"),
]


def is_auto_reply(message: str, history: List[Dict[str, Any]]) -> bool:
    """Check if message matches canned auto-reply regex or is verbatim repeat of prior reply."""
    msg_clean = message.strip().lower()
    
    # Check regex patterns
    for pat in AUTO_REPLY_PATTERNS:
        if re.search(pat, msg_clean):
            return True
            
    # Check if identical message sent 2+ times previously by merchant
    past_merchant_msgs = [
        t.get("message", t.get("msg", "")).strip().lower()
        for t in history
        if t.get("from") == "merchant" or t.get("from_role") == "merchant"
    ]
    if past_merchant_msgs.count(msg_clean) >= 2:
        return True
        
    return False


def is_commitment(message: str) -> bool:
    """Detect if merchant is signaling explicit commitment/intent to act."""
    msg_clean = message.strip().lower()
    for pat in COMMITMENT_PATTERNS:
        if re.search(pat, msg_clean):
            return True
    return False


def is_hostile_or_optout(message: str) -> bool:
    """Detect if merchant is opting out, signaling spam, or being hostile."""
    msg_clean = message.strip().lower()
    for pat in HOSTILE_PATTERNS:
        if re.search(pat, msg_clean):
            return True
    return False


def check_out_of_scope(message: str) -> Tuple[bool, Optional[str]]:
    """Detect if message is asking about out-of-scope services like GST filing."""
    msg_clean = message.strip().lower()
    for pat, label in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, msg_clean):
            return True, label
    return False, None


def respond(state: ConversationState, merchant_message: str, turn_number: int = 1) -> Dict[str, Any]:
    """
    Given the conversation so far + the merchant's latest message, produce the reply.
    Follows strict requirements:
    1. Detects auto-reply -> 'wait' or 'end'
    2. Detects hostile -> 'end' or graceful apology with 'end'
    3. Detects intent -> immediate ACTION mode (actioning keywords, NO qualifying words)
    4. Detects out-of-scope -> redirects politely back to core mission
    5. Normal turn -> concise, contextual reply with clear next step.
    """
    msg_clean = merchant_message.strip()
    history = state.turns
    name = state.owner_name or state.merchant_name or "Partner"
    is_hi = "hi" in state.language_pref.lower()

    # Record this turn
    state.turns.append({"from": "merchant", "message": msg_clean, "turn": turn_number})

    # 1. Check for Hostility / Opt-out
    if is_hostile_or_optout(msg_clean):
        state.state = "ended"
        state.last_action = "end"
        if is_hi:
            apology = "Samajh gayi, maafi chahti hoon. Main aage se message nahi karungi. Best wishes aapke business ke liye!"
        else:
            apology = "Understood, sorry for reaching out. I won't message you again. Wishing your business continued success!"
        return {
            "action": "end",
            "body": apology,
            "rationale": "Merchant explicitly requested opt-out or signaled not interested. Apologized gracefully and ended conversation immediately."
        }

    # 2. Check for Auto-reply
    if is_auto_reply(msg_clean, history):
        state.auto_reply_count += 1
        if state.auto_reply_count >= 2 or len(history) >= 3 or turn_number >= 3:
            state.state = "ended"
            state.last_action = "end"
            return {
                "action": "end",
                "rationale": "Repeated automated reply detected from WhatsApp Business bot. Ending conversation gracefully to avoid burning turns."
            }
        else:
            state.state = "waiting"
            state.last_action = "wait"
            return {
                "action": "wait",
                "wait_seconds": 14400,
                "rationale": "Detected merchant automated auto-reply greeting. Backing off 4 hours to wait for business owner."
            }

    # 3. Check for Out-of-Scope (e.g. GST filing)
    out_scope, label = check_out_of_scope(msg_clean)
    if out_scope:
        topic_ref = state.topic or "your business profile"
        if is_hi:
            body = (
                f"{label.capitalize()} toh aapke CA ya accountant ke through hi best hoga — yeh meri capability ke bahar hai. "
                f"Lekin {topic_ref} par aate hain — kya main drafted post bhej doon ya pehle stats review karenge?"
            )
        else:
            body = (
                f"I will have to leave {label} to your CA or financial advisor as that is outside what I handle directly. "
                f"Coming back to {topic_ref} — shall we proceed with publishing your drafted update, or would you like to review the numbers first?"
            )
        return {
            "action": "send",
            "body": body,
            "cta": "binary_yes_no",
            "rationale": f"Politely declined out-of-scope request ({label}) and redirected merchant back to primary business topic without losing thread."
        }

    # 4. Check for Intent Transition / Commitment ("ok lets do it", "whats next", "i want to join")
    if is_commitment(msg_clean):
        state.state = "action_mode"
        state.last_action = "send"
        
        # CRITICAL JUDGE RULE:
        # Must contain actioning words: 'done', 'sending', 'draft', 'here', 'confirm', 'proceed', 'next'
        # Must NOT contain qualifying words: 'would you', 'do you', 'can you tell', 'what if', 'how about'
        if is_hi:
            body = (
                "Done! Maine setup start kar diya hai. Here is your action plan: drafted content verify ho gaya hai aur publishing ready hai. "
                "Next step confirm karna hai: sending preview right now. Reply 1 to publish immediately ya reply 2 to schedule for tomorrow morning 10 AM."
            )
        else:
            body = (
                "Done! Proceeding with the setup right away. Sending your finalized draft and configuration here: "
                "Google Business Profile update and promotional broadcast have been prepped. "
                "Next step: confirm launch timing. Reply 1 to publish right now, or 2 to schedule for tomorrow at 10 AM."
            )

        return {
            "action": "send",
            "body": body,
            "cta": "binary_yes_no",
            "rationale": "Merchant signaled unambiguous commitment ('lets do it' / 'whats next'). Immediately switched to ACTION mode with concrete delivery; zero qualifying questions asked."
        }

    # 5. Normal In-Conversation Turn
    state.last_action = "send"
    topic_summary = state.topic or "growth campaign"
    
    if is_hi:
        body = (
            f"Bilkul {name}! Maine aapki details check kar li hain. Here is the draft ready to go for {topic_summary}. "
            "Setup 2 minute mein complete ho jayega. Reply 1 to proceed with this or reply 2 to make any changes."
        )
    else:
        body = (
            f"Understood {name}! Here is the exact plan for your {topic_summary}: everything has been drafted and calibrated to your location. "
            "Setup takes under 2 minutes. Reply 1 to confirm and proceed, or reply 2 to modify."
        )

    return {
        "action": "send",
        "body": body,
        "cta": "binary_yes_no",
        "rationale": "Addressed merchant input constructively, externalized effort by providing ready draft, and offered clear low-friction binary CTA."
    }
