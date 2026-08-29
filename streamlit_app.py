import streamlit as st
from rag import __RAG_VERSION__
from rag import (
    get_chain, normalize, is_greeting, is_reset, detect_social,
    is_empty_or_punct, _SOCIAL_REPLIES, _GREETING_REPLY,
    RECIPE_CHOICE_WORDS, NUTRITION_CHOICE_WORDS, youtube_search,
)

st.set_page_config(page_title="Food Recipe RAG", page_icon="🍲", layout="centered")
st.title("🍲 Food Recipe RAG")
st.caption("Ask me about Indian recipes — ingredients, instructions, diet type, and more!")
st.caption(f"_Running: {__RAG_VERSION__}_")


# ─── Cached chain ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    return get_chain()

chain = load_chain()


# ─── Session state ────────────────────────────────────────────────────────────
DEFAULTS = {
    "messages":            [],   # each msg: {role, content, options?, video?}
    "last_recipe_options": [],
    "conv_state":          "idle",
    "pending_ingredient":  None,
    "pending_recipes":     [],
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ─── Option resolution ────────────────────────────────────────────────────────
_ORDINALS = ["first","second","third","fourth","fifth",
             "sixth","seventh","eighth","ninth","tenth"]

def _build_option_map():
    m = {}
    for i in range(10):
        m[str(i + 1)]            = i
        m[f"option {i + 1}"]    = i
        m[f"{_ORDINALS[i]} one"] = i
        m[f"the {_ORDINALS[i]}"] = i
        m[_ORDINALS[i]]          = i
        m[f"recipe {i + 1}"]    = i
        m[f"number {i + 1}"]    = i
        m[f"#{i + 1}"]          = i
        m[f"no {i + 1}"]        = i
    return m

OPTION_MAP = _build_option_map()

def resolve_option(text: str, pool: list) -> str | None:
    key = normalize(text)
    idx = OPTION_MAP.get(key)
    if idx is not None and idx < len(pool):
        return pool[idx]
    for name in pool:
        if key == name.lower().strip():
            return name
    if len(key) >= 4:
        for name in pool:
            if key in name.lower():
                return name
    return None


# ─── Helpers ──────────────────────────────────────────────────────────────────
def build_chat_history(messages: list, max_turns: int = 6) -> str:
    history = messages[:-1]
    recent  = history[-(max_turns * 2):]
    if not recent:
        return "(no previous conversation)"
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )

def show_user(text: str, display: str | None = None):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(display or text)

def render_chips(options: list, max_n: int = 5):
    """Render quick-select chips as Streamlit UI. NOT saved in message text."""
    if not options or len(options) < 2:
        return
    st.markdown("---")
    st.caption("**Quick select:**")
    cols = st.columns(min(len(options), max_n))
    for i, name in enumerate(options[:max_n]):
        with cols[i]:
            st.markdown(f"`{i+1}.` {name[:30]}{'…' if len(name) > 30 else ''}")

def render_video(video: dict | None):
    """Render YouTube video block as Streamlit UI."""
    if not video or not video.get("link"):
        return
    st.markdown("---")
    title = video["title"]
    link = video["link"]
    meta = " · ".join(x for x in [video.get("channel",""), video.get("duration","")] if x)
    st.markdown(f"📺 **[{title}]({link})**")
    if meta:
        st.caption(meta)

def reset_state():
    st.session_state.conv_state         = "idle"
    st.session_state.pending_ingredient = None
    st.session_state.pending_recipes    = []


def bot_reply_streaming(stream_payload: dict) -> dict:
    """
    Stream LLM tokens into chat. Returns the final result dict.
    UI artifacts (chips, video) rendered AFTER the message but not saved into content.

    stream_payload: dict passed to chain.stream()
    """
    final = {"answer": "", "recipe_options": [], "next_action": None,
             "pending_ingredient": None, "video": None}
    meta_video = None
    meta_options = None

    with st.chat_message("assistant"):
        placeholder = st.empty()
        buffer = []

        for event in chain.stream(stream_payload):
            etype = event["event"]
            data  = event["data"]

            if etype == "token":
                buffer.append(data)
                # Throttled rendering: update placeholder with current buffer
                placeholder.markdown("".join(buffer) + " ▌")

            elif etype == "meta":
                # Pre-stream metadata (recipe_options, video)
                if "video" in data and data["video"]:
                    meta_video = data["video"]
                if "recipe_options" in data and data["recipe_options"]:
                    meta_options = data["recipe_options"]

            elif etype == "done":
                final = data

        # Final render without cursor
        full_answer = "".join(buffer)
        placeholder.markdown(full_answer)

        # Render UI artifacts AFTER the answer (not embedded in text)
        video = final.get("video") or meta_video
        if video:
            render_video(video)

        options = final.get("recipe_options") or meta_options or []
        # Only show chips when in idle state and no state transition pending
        if (options and len(options) > 1
                and final.get("next_action") is None
                and not stream_payload.get("mode")):
            render_chips(options)

    # Save assistant message — content is ONLY the text, not UI artifacts
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "video":   final.get("video") or meta_video,
        "options": options if final.get("next_action") is None and not stream_payload.get("mode") else [],
    })

    # Update final state for caller
    final["answer"] = full_answer
    if not final.get("recipe_options"):
        final["recipe_options"] = options
    return final


def bot_reply_static(content: str):
    """Save+render a static (non-streamed) bot message — no UI artifacts."""
    st.session_state.messages.append({"role": "assistant", "content": content})
    with st.chat_message("assistant"):
        st.markdown(content)


# ─── Render conversation history ─────────────────────────────────────────────
for _m in st.session_state.messages:
    with st.chat_message(_m["role"]):
        st.markdown(_m["content"])
        # Re-render UI artifacts that were attached to this message
        if _m.get("video"):
            render_video(_m["video"])
        if _m.get("options"):
            render_chips(_m["options"])


# ─── Idle handler ─────────────────────────────────────────────────────────────
def process_idle(query: str, render_user: bool = True):
    if render_user:
        show_user(query)

    chat_history = build_chat_history(st.session_state.messages)
    final = bot_reply_streaming({
        "question":         query,
        "chat_history":     chat_history,
        "session_messages": st.session_state.messages,
    })

    next_action = final.get("next_action")
    new_options = final.get("recipe_options", [])
    pending_ing = final.get("pending_ingredient")

    if new_options:
        st.session_state.last_recipe_options = new_options

    if next_action == "awaiting_intent":
        st.session_state.conv_state         = "awaiting_intent"
        st.session_state.pending_ingredient = pending_ing or normalize(query)
    elif next_action == "awaiting_recipe_selection":
        st.session_state.conv_state      = "awaiting_recipe_selection"
        st.session_state.pending_recipes = new_options


# ─── Awaiting_intent handler ──────────────────────────────────────────────────
def handle_awaiting_intent(query: str):
    ingredient = st.session_state.pending_ingredient or "your ingredient"
    n          = normalize(query)

    if is_empty_or_punct(n):
        return

    show_user(query)

    if is_reset(n):
        reset_state()
        bot_reply_static("Reset. What would you like to cook?")
        return

    if n in RECIPE_CHOICE_WORDS:
        final = bot_reply_streaming({
            "question":         f"{ingredient} recipes",
            "chat_history":     build_chat_history(st.session_state.messages),
            "session_messages": st.session_state.messages,
            "mode":             "list_recipes",
            "ingredient":       ingredient,
        })
        new_options = final.get("recipe_options", [])
        if new_options:
            st.session_state.pending_recipes     = new_options
            st.session_state.last_recipe_options = new_options
            st.session_state.conv_state          = "awaiting_recipe_selection"
        else:
            reset_state()
        return

    if n in NUTRITION_CHOICE_WORDS:
        bot_reply_streaming({
            "question":         f"{ingredient} nutrition",
            "chat_history":     build_chat_history(st.session_state.messages),
            "session_messages": st.session_state.messages,
            "mode":             "nutrition",
            "ingredient":       ingredient,
        })
        reset_state()
        return

    if is_greeting(n):
        bot_reply_static(
            f"{_GREETING_REPLY}\n\n---\n\n"
            f"Back to **{ingredient}** — type **1** for Recipes or **2** for Nutrition."
        )
        return
    social_tag = detect_social(n)
    if social_tag:
        bot_reply_static(
            f"{_SOCIAL_REPLIES[social_tag]}\n\n---\n\n"
            f"Back to **{ingredient}** — **1** Recipes / **2** Nutrition"
        )
        return

    # Escape hatch
    reset_state()
    process_idle(query, render_user=False)


# ─── Awaiting_recipe_selection handler ────────────────────────────────────────
def handle_awaiting_recipe(query: str):
    n    = normalize(query)
    pool = st.session_state.pending_recipes

    if is_empty_or_punct(n):
        return

    if is_reset(n):
        show_user(query)
        reset_state()
        bot_reply_static("Reset. What would you like to cook?")
        return

    resolved = resolve_option(n, pool)
    if resolved:
        display = f"{query} → **{resolved}**" if normalize(resolved) != n else query
        show_user(query, display)
        bot_reply_streaming({
            "question":         resolved,
            "chat_history":     build_chat_history(st.session_state.messages),
            "session_messages": st.session_state.messages,
            "mode":             "full_recipe",
            "recipe_name":      resolved,
        })
        reset_state()
        return

    show_user(query)

    if n.isdigit():
        bot_reply_static(
            f"That's outside the list — only {len(pool)} recipes shown. "
            f"Pick 1–{len(pool)}, type a name, or say **back** to reset."
        )
        return

    if is_greeting(n):
        bot_reply_static(f"{_GREETING_REPLY}\n\n---\n\nType a number or name from the list above.")
        return
    social_tag = detect_social(n)
    if social_tag:
        bot_reply_static(f"{_SOCIAL_REPLIES[social_tag]}\n\n---\n\nPick a recipe by number or name.")
        return

    reset_state()
    process_idle(query, render_user=False)


# ─── Chat input ───────────────────────────────────────────────────────────────
_PLACEHOLDER = {
    "idle":                      "Ask for a recipe, ingredient, or cooking tips…",
    "awaiting_intent":           "Type 1 for Recipes or 2 for Nutrition (or ask anything)…",
    "awaiting_recipe_selection": "Type a number, recipe name, or 'back' to reset…",
}

if user_input := st.chat_input(_PLACEHOLDER.get(st.session_state.conv_state, "Ask me anything…")):
    state = st.session_state.conv_state

    if   state == "awaiting_intent":            handle_awaiting_intent(user_input)
    elif state == "awaiting_recipe_selection":  handle_awaiting_recipe(user_input)
    else:                                       process_idle(user_input)

    st.rerun()


# ─── Clear conversation ───────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear conversation"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
