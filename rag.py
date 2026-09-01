"""
Food Recipe RAG — rag.py  (v4)

NEW IN V4:
  • Streaming output via stream_chain() generator
  • YouTube search integration (with graceful fallback to search URL)
  • Sanitized chat history (strips "Quick select:" pattern to prevent LLM mimicry)
  • Expanded food intent (hunger, "what to eat", cravings → food_knowledge, not OOS)
  • Contextual "add X to it" → modifies prior recipe, doesn't trigger ingredient_inquiry
  • "summarize"/"recap" handled as conversation, not new recipe lookup
  • Stronger anti-hallucination for "mix X and Y" type queries
"""
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import re
from dotenv import load_dotenv
from ingest import build_chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

__RAG_VERSION__ = "v5.1 (2024-06-05)"
CHROMA_PATH         = "chroma_db"
RETRIEVAL_THRESHOLD = 0.9
REPEAT_WINDOW       = 5
MAX_QUERY_LEN       = 500

# ──────────────────────────────────────────────────────────────────────────────
# YOUTUBE INTEGRATION (graceful — never breaks the app)
# ──────────────────────────────────────────────────────────────────────────────
import urllib.parse

_yt_cache: dict[str, dict] = {}    # recipe_name → {title, link, thumbnail}

def youtube_search(query: str) -> dict | None:
    """
    Try to fetch top YouTube video for query.
    Returns {"title": ..., "link": ..., "thumbnail": ...} or None.
    Falls back to a search-results URL if API fails.
    """
    key = query.lower().strip()
    if key in _yt_cache:
        return _yt_cache[key]

    # Try youtube-search-python first
    try:
        from youtubesearchpython import VideosSearch
        v = VideosSearch(f"{query} recipe", limit=1)
        results = v.result().get("result", [])
        if results:
            r = results[0]
            payload = {
                "title":     r.get("title", query),
                "link":      r.get("link", ""),
                "thumbnail": (r.get("thumbnails") or [{}])[0].get("url", ""),
                "channel":   (r.get("channel") or {}).get("name", ""),
                "duration":  r.get("duration", ""),
            }
            _yt_cache[key] = payload
            return payload
    except Exception as e:
        print(f"[YouTube ERROR] {e}")

    # Fallback: just a search URL
    fallback = {
        "title":     f"Search YouTube for {query}",
        "link":      "https://www.youtube.com/results?search_query=" +
                     urllib.parse.quote_plus(f"{query} recipe"),
        "thumbnail": "",
        "channel":   "",
        "duration":  "",
    }
    _yt_cache[key] = fallback
    return fallback


def format_youtube_block(video: dict | None) -> str:
    """Markdown for a YouTube link block. Empty string if no video."""
    if not video or not video.get("link"):
        return ""
    title = video["title"]
    link  = video["link"]
    channel = video.get("channel", "")
    duration = video.get("duration", "")
    meta = " · ".join(x for x in [channel, duration] if x)
    parts = ["\n\n---\n\n📺 **Watch on YouTube**", f"[{title}]({link})"]
    if meta:
        parts.append(f"*{meta}*")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# STATIC KEYWORD SETS — handled without LLM
# ──────────────────────────────────────────────────────────────────────────────

_GREET_ROOTS = (
    "hi","hii","hello","hey","heya","hiya","howdy","sup","yo",
    "gm","gn","ge","namaste","namaskar","hola","greetings",
)
_GREET_EXACT = {
    "good morning","good afternoon","good evening","good day","good night",
    "goodmorning","goodafternoon","goodevening","goodday","goodnight",
    "morning","evening","afternoon",
    "hey there","hi there","hello there","hi bot","hey bot","hello bot",
}

_SOCIAL_PATTERNS = [
    (r"^how (are|r) ?(you|u|ya)\??$",               "social_howareyou"),
    (r"^how('?s| is) it going\??$",                 "social_howareyou"),
    (r"^hru\??$",                                   "social_howareyou"),
    (r"^are you (fine|ok|okay|alright|good)\??$",   "social_howareyou"),
    (r"^you (ok|okay|fine|good|alright)\??$",        "social_howareyou"),
    (r"^how (have you been|do you do)\??$",         "social_howareyou"),
    (r"^(what'?s up|whats up|wassup|sup)\??$",      "social_howareyou"),
    (r"^are (you|u) (dumb|stupid|smart|good|bad|real|human|ai|a bot|a robot)\??$", "social_identity"),
    (r"^(thanks?|thank you|ty|tysm|thx|thnx|cheers)!?$", "social_thanks"),
    (r"^(bye|goodbye|see ya|see you|cya|ttyl|gn)!?$",     "social_bye"),
    (r"^(who are you|what are you|whats your name|what is your name|who r u)\??$", "social_identity"),
    (r"^(what can you do|help|menu|options|what should i ask)\??$", "social_help"),
    (r"^(yes|no|ok|okay|sure|cool|nice|great|alright|got it|fine)\.?!?$", "social_ack"),
]

# ─────────────────────────────────────────────────────────────────────────
# LIFESTYLE / STATE DETECTION
# These queries describe the USER'S STATE and need contextual food suggestions
# (NOT a generic nutrition lecture and NOT a random recipe).
# ─────────────────────────────────────────────────────────────────────────

# Emotional states → comfort/mood food
_EMOTIONAL_STATES = {
    "sad", "lonely", "down", "depressed", "upset", "unhappy", "blue",
    "stressed", "anxious", "worried", "tense", "overwhelmed",
    "tired", "exhausted", "drained", "weary", "low energy",
    "happy", "excited", "joyful", "celebrating", "thrilled",
    "bored", "lazy", "sleepy", "drowsy",
    "sick", "ill", "unwell", "under the weather", "not feeling well",
    "cold", "feverish", "having a cold", "have a cold",
    "angry", "frustrated", "irritated",
}

# Physical/activity states → appropriate food for that state
_PHYSICAL_STATES = {
    "hungry", "starving", "famished", "peckish",
    "thirsty", "dehydrated",
    "full", "stuffed", "bloated",
    "going to sleep", "going to bed", "about to sleep",
    "just woke up", "just got up", "waking up",
    "studying", "working", "at work", "in office",
}

# Activity contexts → "before/after X" food suggestions
_ACTIVITY_PATTERNS = [
    r"\bbefore (gym|workout|exercise|running|run|jog|jogging|yoga|match|game|exam|test|study|studying|sleep|bed|practice)\b",
    r"\bafter (gym|workout|exercise|running|run|jog|jogging|yoga|match|game|exam|test|practice)\b",
    r"\bpre[ -]?workout\b",
    r"\bpost[ -]?workout\b",
    r"\bpre[ -]?game\b",
    r"\bpost[ -]?game\b",
    r"\b(going to|about to|gonna|will) (play|exercise|run|work ?out|study|sleep|jog|gym|exam|test|college|school|office|meeting)\b",
    r"\b(played|exercised|ran|worked out|studied|jogged|did gym)\b",
]

def detect_state(s: str) -> dict | None:
    """
    Returns {"state": <label>, "category": "emotional"|"physical"|"activity"} or None.
    Catches "i am X", "i'm X", "i feel X", "feeling X", "before X", "after X".
    """
    if not s:
        return None

    # Activity context first (most specific)
    for pattern in _ACTIVITY_PATTERNS:
        m = re.search(pattern, s)
        if m:
            return {"state": m.group(0), "category": "activity"}

    # "i am / i'm / i feel / feeling X" prefix
    m = re.search(r"\b(?:i am|i'?m|i feel|feeling|im)\s+(.+?)(?:\s+and\s|$|[,.!?])", s)
    captured = m.group(1).strip() if m else None
    if captured:
        # Strip "going to", "about to" etc — handled as activity if matched above
        for state in _EMOTIONAL_STATES:
            if state in captured or captured == state:
                return {"state": state, "category": "emotional"}
        for state in _PHYSICAL_STATES:
            if state in captured or captured == state:
                return {"state": state, "category": "physical"}

    # Direct mentions without "i am" — "hungry", "starving"
    words = set(s.split())
    for state in _PHYSICAL_STATES:
        if state in words or (" " in state and state in s):
            return {"state": state, "category": "physical"}
    for state in _EMOTIONAL_STATES:
        if state in words or (" " in state and state in s):
            return {"state": state, "category": "emotional"}

    return None



_FOOD_SUGGESTION_PATTERNS = [
    r"\bwhat (should|can|do|to) (i|we)? ?eat\b",
    r"\bwhat to eat\b",
    r"\bwhat (should|can) i cook\b",
    r"\bwhat (can|to) cook\b",
    r"\bany (food|recipe|dish) (suggestion|idea|recommendation)",
    r"\bsuggest (a |me )?(food|recipe|dish|meal)\b",
    r"\bcraving\b",
    r"\bsomething (to eat|tasty|spicy|sweet)\b",
    r"\b(breakfast|lunch|dinner|snack) (idea|option|suggestion)",
    r"\brecommend (a |me )?(food|recipe|dish|meal)\b",
]
def is_food_suggestion_query(s: str) -> bool:
    return any(re.search(p, s) for p in _FOOD_SUGGESTION_PATTERNS)

# ─────────────────────────────────────────────────────────────────────────
# WANT-TO-EAT / LIST detection
# Patterns like "I want to eat chicken", "show me paneer recipes", "list out X dishes"
# → trigger list_recipes mode for ingredient X.
# ─────────────────────────────────────────────────────────────────────────
_WANT_PATTERNS = [
    # "i want/wanna/need/would like/'d like to eat/have/try X"
    r"\bi (?:want|wanna|need|would like|['’]?d like|d like) (?:to )?(?:eat|have|try|cook|make|order)\s+(?:a |an |some |the )?(.+?)(?:\s+dish(?:es)?|\s+recipe(?:s)?|\s+food|$)",
    # "i'd like to try X" / "i'd like X"  (apostrophe-d attached to i)
    r"\bi['’]?d (?:like|love) (?:to )?(?:eat|have|try|cook|make|order)\s+(?:a |an |some |the )?(.+?)(?:\s+dish(?:es)?|\s+recipe(?:s)?|\s+food|$)",
    # "i want a X dish/recipe"
    r"\bi (?:want|wanna|need|would like|d like)\s+(?:a |an |some |the )?(.+?)\s+(?:dish(?:es)?|recipe(?:s)?|food)\b",
    # "show/list/give me X recipes/dishes/options"
    r"\b(?:show|list|give|tell)(?:\s+out|\s+down)?\s+(?:me\s+)?(?:some |a few |top |the )?(.+?)\s+(?:recipe(?:s)?|dish(?:es)?|option(?:s)?|idea(?:s)?|food(?:s)?)\b",
    # "list out/down some X"
    r"\blist(?:\s+out|\s+down)?\s+(?:some\s+|a few\s+|top\s+)?(.+?)(?:\s+recipe(?:s)?|\s+dish(?:es)?|\s+food|$)",
    # "what are some X recipes"
    r"\bwhat are some (.+?) (?:recipe|dish|food|option)",
    # "X recipes" / "X dishes" (bare)
    r"^(.+?)\s+(?:recipes?|dishes?)\??$",
]

# Common food/cuisine words to validate the extracted target
_KNOWN_FOOD_TOKENS = {
    "chicken","mutton","paneer","beef","fish","prawn","egg","eggs","mushroom","mushrooms",
    "vegetable","veg","vegetarian","vegan","non-veg","non vegetarian",
    "rice","dal","lentil","lentils","wheat","bread","pasta","noodle","noodles","biryani",
    "curry","gravy","soup","salad","snack","snacks","sweet","sweets","dessert","desserts",
    "drink","drinks","beverage","beverages","tea","coffee","juice","smoothie","smoothies",
    "breakfast","lunch","dinner","appetizer","appetiser","starter","starters",
    "indian","north indian","south indian","bengali","punjabi","gujarati","kerala",
    "italian","chinese","continental","mexican","thai","mediterranean",
    "spicy","sweet","sour","tangy","creamy","crispy",
    "potato","potatoes","tomato","tomatoes","onion","onions","corn","beans","peas",
    "cheese","butter","cream","yogurt","yoghurt","ghee","milk",
    "lamb","goat","pork","seafood","shrimp","crab","lobster",
    "pizza","burger","sandwich","wrap","roll","kebab","tikka",
    "spinach","broccoli","carrot","carrots","cauliflower","cabbage","beetroot",
    "saag","palak","baingan","aloo","matar","chana","rajma","chole",
    "pulao","kheer","halwa","ladoo","barfi","gulab jamun","jalebi",
    "samosa","pakora","pakoda","dosa","idli","vada","uttapam","poha","upma",
    "roti","paratha","naan","kulcha","puri","bhatura",
    "thali","khichdi","biryani","pulav","fried rice",
}

def _is_valid_food_target(s: str) -> bool:
    """A target like '6' or 'the' is not a real ingredient. Validate before using."""
    s = s.strip(" .,!?")
    if not s: return False
    if s.isdigit(): return False
    if len(s) < 3: return False
    if s in {"a","an","the","it","this","that","these","those","some","any","all"}:
        return False
    # Accept if any token matches known food vocabulary OR ingredient is reasonably long
    tokens = s.split()
    if any(t in _KNOWN_FOOD_TOKENS for t in tokens):
        return True
    # Single non-vocab word ≥3 chars — could be a less common ingredient, accept
    if len(tokens) == 1 and len(s) >= 4 and any(c in "aeiou" for c in s):
        return True
    # Multi-word: accept if all alpha and ≥6 chars total
    if len(tokens) > 1 and all(t.replace("-","").isalpha() for t in tokens):
        return True
    return False

def detect_want_to_eat(s: str) -> str | None:
    """
    Return the target food/ingredient/cuisine if user wants a list of recipes, else None.
    """
    if not s: return None
    _JUNK_WORDS = {"some","a","an","the","top","few","out","down","me","recipes","dishes",
                   "food","options","ideas","recipe","dish"}
    for pattern in _WANT_PATTERNS:
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if not m: continue
        cand = m.group(1).strip(" .,!?'\"’")
        # Strip leading article words
        for art in ["a ", "an ", "some ", "the ", "top ", "few "]:
            if cand.startswith(art):
                cand = cand[len(art):]
        cand = cand.strip()
        if not cand or cand in _JUNK_WORDS: continue
        if _is_valid_food_target(cand):
            return cand
    return None


# Back-compat alias
def is_hunger_query(s: str) -> bool:
    state = detect_state(s)
    if state and state["state"] in ("hungry","starving","famished","peckish"):
        return True
    return is_food_suggestion_query(s)

_RESET_KEYWORDS = {"back","cancel","menu","exit","stop","reset","clear","restart","quit"}

RECIPE_CHOICE_WORDS    = {"1","recipe","recipes","full recipe","option 1","first option","r"}
NUTRITION_CHOICE_WORDS = {"2","nutrition","health","calories","calorie","nutritional","option 2","second option","n"}

_INGREDIENT_PREFIXES = [
    r"i want to know (more )?about ",
    r"i wanna know (more )?about ",
    r"i'?d like to know (more )?about ",
    r"tell me (more )?about ",
    r"tell me ",
    r"what (is|are|about) ",
    r"(can you )?(give me |provide )?(info|information|details|facts) (on|about) ",
    r"explain (what )?",
    r"describe ",
    r"i want ",
    r"i need ",
    r"show me ",
    r"how about ",
    r"do you know (about )?",
]

# OOS — but FOOD-RELATED terms whitelisted before this check
_OOS_KEYWORDS = {
    "politics","election","president","prime minister","cricket","football","soccer",
    "nba","ipl","movie","film","actor","actress","song","music","weather","stock",
    "bitcoin","crypto","programming","python ","javascript","code ","coding","caching",
    " war ","geography","history of","when did","capital of","wikipedia","what time",
}
# These food contexts override OOS keywords
_FOOD_OVERRIDES = {
    "recipe","food","dish","cook","eat","meal","ingredient","cuisine","spice",
    "kitchen","chef","bake","fry","grill","boil","roast","gravy","curry",
    "drink","beverage","tea","coffee","juice","lassi","smoothie",
}

def looks_out_of_scope(s: str) -> bool:
    has_oos  = any(kw in s for kw in _OOS_KEYWORDS)
    has_food = any(kw in s for kw in _FOOD_OVERRIDES)
    return has_oos and not has_food

# Summarize / recap intent
_SUMMARY_PATTERNS = [
    r"^summari[sz]e\b",
    r"^recap\b",
    r"^what (did|have) we (discuss|talk|cover)",
    r"^tldr\b",
    r"^summary\b",
]
def is_summary_request(s: str) -> bool:
    return any(re.search(p, s) for p in _SUMMARY_PATTERNS)


# Complaint / correction detection — "but I didn't ask for that"
_COMPLAINT_PATTERNS = [
    r"\bbut (i|we) (didn'?t|did ?not|didnot|never)\b",
    r"\bi (didn'?t|did ?not|didnot) ask\b",
    r"\bwhy did you\b",
    r"\bthat'?s (not|wrong|incorrect)\b",
    r"\bthat is (not what|wrong|incorrect)\b",
    r"\bnot what (i|we) (wanted|asked|meant)\b",
    r"\byou'?re (wrong|incorrect|mistaken)\b",
    r"\b(you|u) (gave|told|said) me\b.*\b(but|when)\b",
    r"\bshould have\b",
    r"\binstead of (giving|telling)\b",
]
def is_complaint(s: str) -> bool:
    return any(re.search(p, s) for p in _COMPLAINT_PATTERNS)

# Contextual modification: "add X to it", "make it spicy", "without onion"
_MODIFICATION_PATTERNS = [
    r"^add \w+",
    r"^(without|no|skip|remove) \w+",
    r"^make it (more |less )?\w+",
    r"^with (extra |more |less )?\w+",
    r"\bin it\b",
    r"\bto (it|this|that)\b",
    r"^(can|could) i (add|substitute|replace|use)",
]
def is_modification(s: str) -> bool:
    return any(re.search(p, s) for p in _MODIFICATION_PATTERNS)

# Gibberish: no vowels or random char sequence
def is_gibberish(s: str) -> bool:
    """
    Detect single-word random strings.
    Heuristics:
      1. No vowels at all → gibberish (e.g. 'qwxzbnmpqr')
      2. Vowel ratio < 15% in a long word → gibberish (e.g. 'sjdhqikdyiqdldk')
      3. 4+ consonants in a row (no CV alternation) → gibberish
    Multi-word inputs are NEVER gibberish (real questions are multi-word).
    """
    if len(s) < 5: return False
    words = s.split()
    if len(words) > 1: return False
    word = words[0]
    alpha = [c for c in word if c.isalpha()]
    if len(alpha) < 5: return False
    vowels = sum(1 for c in alpha if c in "aeiou")
    vowel_ratio = vowels / len(alpha)
    if vowel_ratio == 0: return True
    if vowel_ratio < 0.18: return True
    # 5+ consonants in a row
    if re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', word): return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# PURE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return " ".join((text or "").strip()[:MAX_QUERY_LEN].lower().split())

def is_greeting(s: str) -> bool:
    if not s: return False
    s = s.strip(" !.?,")
    if s in _GREET_EXACT: return True
    for root in _GREET_ROOTS:
        if re.fullmatch(rf"{root}+[!?.]*", s): return True
        if s.startswith(root + " ") and len(s.split()) <= 3: return True
    return False

def detect_social(s: str) -> str | None:
    if not s: return None
    for pattern, tag in _SOCIAL_PATTERNS:
        if re.match(pattern, s): return tag
    return None

def is_reset(s: str) -> bool:
    return s.strip(" !.?,") in _RESET_KEYWORDS

def is_empty_or_punct(s: str) -> bool:
    return not s or not any(c.isalnum() for c in s)

def extract_ingredient(s: str) -> str:
    out = s
    for prefix in _INGREDIENT_PREFIXES:
        out = re.sub(r"^" + prefix, "", out, flags=re.IGNORECASE).strip()
    return out or s

def detect_repeat(question: str, session_messages: list) -> bool:
    user_turns = [normalize(m["content"]) for m in session_messages if m["role"] == "user"]
    recent = user_turns[-(REPEAT_WINDOW + 1):-1]
    return normalize(question) in recent

_LANG_PATTERNS = [
    (r"\bin hindi\b", "Hindi"), (r"\bin marathi\b", "Marathi"),
    (r"\bin tamil\b", "Tamil"), (r"\bin telugu\b", "Telugu"),
    (r"\bin bengali\b", "Bengali"), (r"\bin gujarati\b", "Gujarati"),
    (r"hindi mein\b", "Hindi"), (r"marathi mein\b", "Marathi"),
    (r"\btranslate to (\w+)\b", None),
]
def detect_language_request(s: str) -> str | None:
    s = s.lower()
    for pattern, lang in _LANG_PATTERNS:
        m = re.search(pattern, s)
        if m: return lang if lang else m.group(1).capitalize()
    return None


def sanitize_history(chat_history: str) -> str:
    """
    Strip 'Quick select:' lines from history to prevent LLM mimicking the pattern.
    Also strips other UI artifacts.
    """
    if not chat_history or chat_history == "(no previous conversation)":
        return chat_history
    out_lines = []
    for line in chat_history.split("\n"):
        # Skip "Quick select:" lines and "---" separators
        if re.search(r"quick select:", line, re.IGNORECASE):
            continue
        if line.strip() in {"---", "***"}:
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


# ──────────────────────────────────────────────────────────────────────────────
# CANNED RESPONSES
# ──────────────────────────────────────────────────────────────────────────────

_GREETING_REPLY = (
    "Hello! 👋 I'm your Food Recipe assistant.\n\n"
    "Ask me about any **ingredient**, **recipe**, or **cooking tips**.\n\n"
)

_SOCIAL_REPLIES = {
    "social_howareyou": "I'm doing great, thanks for asking! 😊 What recipe or ingredient can I help you with?",
    "social_thanks":    "You're welcome! 🍲 Anything else you'd like to cook?",
    "social_bye":       "Goodbye! Come back any time for more recipes. 👋",
    "social_identity":  "I'm Food Recipe RAG — a cooking assistant for Indian recipes, ingredients, and nutrition info. I'm an AI, but a friendly one! 🤖🍲",
    "social_help": (
        "Here's what I can help with:\n\n"
        "• **Ingredient info** — type any ingredient (e.g. *mutton*, *paneer*)\n"
        "• **Recipes** — ask for a specific dish (e.g. *chicken biryani*)\n"
        "• **Nutrition** — health & calorie info for any food\n"
        "• **Cooking tips** — techniques, substitutions\n"
        "• **Modifications** — *add cheese to it*, *make it spicy*"
    ),
    "social_ack": "Got it! What would you like to cook or learn about?",
}

_OUT_OF_SCOPE_REPLY = (
    "I can only help with **food, cooking, recipes, and nutrition**. 🍲\n\n"
    "Try asking about an ingredient or dish you'd like to make."
)

_GIBBERISH_REPLY = (
    "I didn't quite catch that — could you rephrase? 🤔\n\n"
    "Try asking about a food, ingredient, or recipe."
)

_EMPTY_REPLY = "I didn't catch that — could you type your question?"


def make_result(answer: str, recipe_options: list = None, next_action: str = None,
                pending_ingredient: str = None, video: dict | None = None,
                is_static: bool = False) -> dict:
    """
    Uniform return shape.
    is_static=True means the answer is a canned string (no LLM) and shouldn't be streamed.
    """
    return {
        "answer":             answer,
        "recipe_options":     recipe_options or [],
        "next_action":        next_action,
        "pending_ingredient": pending_ingredient,
        "video":              video,
        "is_static":          is_static,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CHAIN FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def _default_opener(state: str, category: str) -> str:
    """Fallback opener if LLM streaming fails."""
    if category == "emotional":
        return f"Sorry to hear you're feeling {state}. Food can definitely help! "
    if category == "physical":
        if state in ("hungry","starving","famished","peckish"):
            return "Let's find something good to eat. "
        return f"Got it — you're feeling {state}. "
    return "Here's what might work for you. "


def get_chain():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_file = os.path.join(
        CHROMA_PATH,
        "chroma.sqlite3"
    )

    if not os.path.exists(chroma_file):
        print("Chroma DB not found. Building...")
        build_chroma()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        max_tokens=900,         
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )

    # ── Router ──────────────────────────────────────────────────────────────
    router_prompt = ChatPromptTemplate.from_template("""
Classify this food chatbot query into ONE category.

ingredient_inquiry — just a food ingredient or "what is X" / "tell me about X"
    Examples: "mutton", "paneer", "tell me about chicken"
    NOT: "mutton curry", "how to cook chicken"

recipe_lookup — recipe, ingredients, steps, dish name, dish selection
    Examples: "biryani recipe", "how to make paneer butter masala"

food_knowledge — nutrition, calories, health, diet, substitutions, meal/snack suggestions,
    food pairings, drinks to pair with food, hunger ("I'm hungry"), cravings, "what to eat"
    Examples: "calories in chicken", "what to eat in evening", "drinks with biryani"

out_of_scope — politics, sports, coding, movies, geography, weather, NON-FOOD topics
    NOT this if the query mentions food, cooking, eating, drinks, or hunger

Query: {question}

Reply with ONE word: ingredient_inquiry, recipe_lookup, food_knowledge, or out_of_scope""")
    router_chain = router_prompt | llm | StrOutputParser()

    # ── Rewrite ──────────────────────────────────────────────────────────────
    rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the user's question as a standalone search query.

- Resolve "it", "this", "first one", "option 2", "tell me more"
- Carry forward active ingredient: history about mutton + "biryani" → "mutton biryani"
- Output ONLY the rewritten query.
- If already self-contained, return unchanged.

History:
{chat_history}

Question:
{question}

Rewritten:""")
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # ── Ingredient summary ──────────────────────────────────────────────────
    ingredient_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a food expert. Respond in ENGLISH only. "
         "Give a concise 3-5 sentence overview of the ingredient: what it is, origin, "
         "key culinary uses, brief nutritional note. Do NOT list recipes. "
         "Do NOT include any 'Quick select:' or option list at the end."),
        ("human", "Tell me about {ingredient} as a food ingredient."),
    ])
    ingredient_chain = ingredient_prompt | llm | StrOutputParser()

    # ── Answer prompt ──────────────────────────────────────────────────────
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Food Recipe RAG, a cooking assistant.

═══ ABSOLUTE RULES ═══

LANGUAGE
  • Default English. NEVER switch unless current message says "in Hindi"/"in Marathi" etc.
  • Numbers ("1", "2") are selections, NOT language requests.
  • Even if retrieved text is in another language, RESPOND IN ENGLISH.

ANTI-HALLUCINATION (CRITICAL)
  • NEVER invent prior conversation. NEVER say "as I mentioned" or "you asked for X earlier"
    unless it LITERALLY appears in the History section below.
  • NEVER fabricate recipes that combine unusual ingredients (e.g. "paneer mutton korma")
    unless the retrieved context contains that exact recipe.
  • When LISTING recipes, list ONLY recipe names that appear in the Retrieved Context.
    Do NOT add other recipes you happen to know (e.g. "Chicken Tikka Masala") unless they're
    in the context.
  • If retrieved context is empty, say: "I don't have that recipe in my database — here's
    a general version from cooking knowledge."
  • NEVER invent nutritional macros (calories, protein, fat in grams) unless from context.
  • Use proper grammar, capitalization, and punctuation in all responses.

OUTPUT FORMAT (no extras)
  • Use this exact format for recipes:
      **[Recipe Name]**
      *Cuisine | Course | Diet | Prep time*
      **Ingredients**
      - item
      **Instructions**
      1. Step
  • Do NOT append "Quick select:", "Options:", "1. X 2. Y" lists at the end of your answer.
  • Do NOT add any UI suffixes — just the recipe.
═══════════════════"""),
        ("human", """History:
{chat_history}

Retrieved Context:
{context}

Question:
{question}

Answer:"""),
    ])
    answer_chain = answer_prompt | llm | StrOutputParser()

    # ── Nutrition prompt ────────────────────────────────────────────────────
    nutrition_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a food and nutrition expert. Respond in English only. "
         "Structure: **Macronutrients**, **Micronutrients**, **Health Benefits**, **Dietary Notes**. "
         "Be factual. Never invent specific numbers — if unsure, give ranges or say 'approximately'. "
         "Do NOT append any 'Quick select:' or option list."),
        ("human", "History:\n{chat_history}\n\nQuestion:\n{question}\n\nAnswer:"),
    ])
    nutrition_chain = nutrition_prompt | llm | StrOutputParser()

    # ── Lifestyle prompt (emotional/physical/activity states) ──────────────
    lifestyle_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a friendly food assistant. The user just shared their CURRENT STATE "
         "(emotional, physical, or activity-related). Respond in English only.\n\n"
         "RULES:\n"
         "• Acknowledge their state in ONE empathetic sentence.\n"
         "• Suggest 4-6 SPECIFIC FOODS/DISHES (with one-line reasoning each).\n"
         "• End with: 'Want a full recipe for any of these? Just type the name.'\n"
         "• Do NOT give a full step-by-step recipe.\n"
         "• Do NOT lecture on nutrition science.\n"
         "• Do NOT append 'Quick select' or option lists.\n"
         "• For sad/stressed states: suggest comfort foods (warm soup, dark chocolate, ice cream, etc.)\n"
         "• For tired/sleepy: suggest energy foods OR light bedtime snacks depending on context.\n"
         "• For hungry: ask what kind they want — light snack, full meal, healthy, comfort, sweet.\n"
         "• For activity context (before gym/match): suggest appropriate timing foods."),
        ("human", "User state: {state} (category: {category})\nFull query: {question}\n\nResponse:"),
    ])
    lifestyle_chain = lifestyle_prompt | llm | StrOutputParser()

    # ── Complaint prompt (acknowledge + reset) ──────────────────────────────
    complaint_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "The user is correcting or complaining about your previous response. "
         "Respond in English only.\n\n"
         "RULES:\n"
         "• Briefly acknowledge the mistake in ONE sentence (e.g. 'You're right, sorry about that.').\n"
         "• Ask what they ACTUALLY want, with 2-3 specific options based on the original query.\n"
         "• Do NOT repeat the previous wrong answer.\n"
         "• Do NOT defend the previous answer.\n"
         "• Keep it short — 2-4 sentences total."),
        ("human", "History:\n{chat_history}\n\nUser complaint:\n{question}\n\nResponse:"),
    ])
    complaint_chain = complaint_prompt | llm | StrOutputParser()

    # ── Conversation prompt (for follow-ups, modifications, summaries) ──────
    conversation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Food Recipe RAG, helping the user with a cooking conversation. "
         "Respond in English only. Use the history to answer follow-ups. "
         "If the user wants to modify a previous recipe (add/remove an ingredient, change spice level), "
         "show the updated recipe clearly. "
         "If asking to summarize, give a brief recap of what was discussed — DO NOT invent new content. "
         "NEVER invent recipes or facts not in the history. "
         "Do NOT append 'Quick select:' or option lists."),
        ("human", "History:\n{chat_history}\n\nQuestion:\n{question}\n\nAnswer:"),
    ])
    conversation_chain = conversation_prompt | llm | StrOutputParser()

    # ── Retrieval ──────────────────────────────────────────────────────────
    def retrieve(query: str, k_fetch: int = 15) -> list:
        try:
            results = db.similarity_search_with_score(query, k=k_fetch)
        except Exception as e:
            print(f"[Retrieval ERROR] {e}")
            return []
        passed = [(d, s) for d, s in results if s < RETRIEVAL_THRESHOLD]
        seen: dict[str, tuple] = {}
        for doc, score in passed:
            name = doc.metadata.get("recipe_name", "Unknown")
            if name not in seen or score < seen[name][1]:
                seen[name] = (doc, score)
        deduped = [doc for doc, _ in seen.values()]
        print(f"[Retrieval] '{query[:50]}' raw={len(results)} dedup={len(deduped)}")
        return deduped

    def format_docs(docs: list) -> str:
        if not docs: return "(No relevant recipes found)"
        return "\n\n---\n\n".join(
            f"[Recipe {i}: {d.metadata.get('recipe_name','Unknown')}]\n{d.page_content}"
            for i, d in enumerate(docs, 1)
        )

    def safe_rewrite(question: str, chat_history: str) -> str:
        if chat_history == "(no previous conversation)":
            return question
        try:
            return rewrite_chain.invoke({
                "chat_history": chat_history, "question": question
            }).strip() or question
        except Exception as e:
            print(f"[Rewrite ERROR] {e}")
            return question

    # ──────────────────────────────────────────────────────────────────────────
    # MODE HANDLERS
    # ──────────────────────────────────────────────────────────────────────────

    def handle_list_recipes(ingredient: str, chat_history: str) -> dict:
        ingredient = extract_ingredient(normalize(ingredient))
        if not _is_valid_food_target(ingredient):
            return make_result(
                "Could you tell me what specific food or ingredient you'd like recipes for?",
                is_static=True,
            )
        docs = retrieve(f"{ingredient} recipe", k_fetch=30)[:10]
        if not docs:
            return make_result(
                f"I couldn't find specific recipes for **{ingredient.title()}**. "
                f"Try a specific dish like *{ingredient} curry*.",
                is_static=True,
            )
        names = [d.metadata.get("recipe_name", "Unknown") for d in docs]
        numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
        answer = (
            f"Here are **{len(names)} {ingredient.title()} recipes** from my database:\n\n"
            f"{numbered}\n\n"
            "Type a number (e.g. **3**) or the recipe name to see the full recipe."
        )
        return make_result(answer, recipe_options=names, next_action="awaiting_recipe_selection",
                          pending_ingredient=ingredient, is_static=True)

    def handle_full_recipe(recipe_name: str, chat_history: str,
                          fetch_video: bool = True) -> dict:
        docs = retrieve(recipe_name, k_fetch=10)
        if not docs:
            return make_result(
                f"I couldn't find **{recipe_name}** in my database.",
                is_static=True,
            )
        exact = [d for d in docs if normalize(d.metadata.get("recipe_name","")) == normalize(recipe_name)]
        chosen = (exact or docs)[:1]
        context = format_docs(chosen)
        try:
            answer = answer_chain.invoke({
                "chat_history": sanitize_history(chat_history),
                "context":      context,
                "question":     f"Give me the complete recipe for {recipe_name}",
            })
        except Exception as e:
            print(f"[Answer ERROR] {e}")
            answer = f"I had trouble generating the recipe for {recipe_name}."

        video = youtube_search(recipe_name) if fetch_video else None
        return make_result(answer, video=video)

    def handle_nutrition(ingredient: str, chat_history: str) -> dict:
        ingredient = extract_ingredient(normalize(ingredient))
        try:
            answer = nutrition_chain.invoke({
                "chat_history": sanitize_history(chat_history),
                "question":     f"Detailed nutrition and health information for {ingredient}",
            })
        except Exception as e:
            print(f"[Nutrition ERROR] {e}")
            answer = "Couldn't fetch nutrition info — try rephrasing."
        return make_result(answer)

    # ──────────────────────────────────────────────────────────────────────────
    # STREAMING SUPPORT
    # The main chain can be called in two ways:
    #   1) full_chain(x) → dict (non-streaming, backward-compatible)
    #   2) full_chain.stream(x) → generator yielding {"event": ..., "data": ...}
    # ──────────────────────────────────────────────────────────────────────────

    def _classify(x: dict) -> dict:
        """
        Returns a decision dict telling the caller WHAT to do next.
        keys: kind ∈ {static, llm_invoke, llm_stream, mode}, payload depends on kind.
        Used by both invoke and stream paths so logic is shared.
        """
        question     = (x.get("question") or "").strip()
        chat_history = x.get("chat_history", "(no previous conversation)")
        session_msgs = x.get("session_messages", [])
        mode         = x.get("mode")

        if mode == "list_recipes":
            return {"kind": "mode_static", "result": handle_list_recipes(x.get("ingredient", question), chat_history)}
        if mode == "full_recipe":
            return {"kind": "mode_llm_stream",
                    "recipe_name": x.get("recipe_name", question),
                    "chat_history": chat_history}
        if mode == "nutrition":
            return {"kind": "mode_llm_stream_nutrition",
                    "ingredient": x.get("ingredient", question),
                    "chat_history": chat_history}

        n = normalize(question)
        print(f"\n{'─'*50}\nQ: {question!r}  norm: {n!r}\n{'─'*50}")

        if is_empty_or_punct(n):
            return {"kind": "static", "result": make_result(_EMPTY_REPLY, is_static=True)}

        if is_reset(n):
            return {"kind": "static", "result": make_result(
                "Conversation reset. What would you like to cook?", is_static=True)}

        if is_greeting(n):
            return {"kind": "static", "result": make_result(_GREETING_REPLY, is_static=True)}

        social_tag = detect_social(n)
        if social_tag:
            return {"kind": "static", "result": make_result(
                _SOCIAL_REPLIES[social_tag], is_static=True)}

        if detect_repeat(question, session_msgs):
            return {"kind": "static", "result": make_result(
                "You've asked this before. What specifically would you like?\n\n"
                "1. **Recipe** — step-by-step instructions\n"
                "2. **Nutrition** — health & calorie info\n"
                "3. **Cooking tips** — techniques & substitutions\n"
                "4. **Different dish** — try another recipe",
                is_static=True,
            )}

        # COMPLAINT FIRST — user correcting a previous answer
        if is_complaint(n) and chat_history != "(no previous conversation)":
            return {"kind": "complaint_stream",
                    "question": question, "chat_history": chat_history}

        # GIBBERISH (before state — gibberish can match "i am xyz" by accident)
        if is_gibberish(n):
            return {"kind": "static", "result": make_result(_GIBBERISH_REPLY, is_static=True)}

        # WANT TO EAT / LIST X RECIPES — bypass router, go straight to list mode
        want_target = detect_want_to_eat(n)
        if want_target:
            return {"kind": "want_to_eat",
                    "ingredient": want_target,
                    "chat_history": chat_history}

        # STATE: emotional/physical/activity
        state_info = detect_state(n)
        if state_info:
            return {"kind": "lifestyle_stream",
                    "state": state_info["state"],
                    "category": state_info["category"],
                    "question": question,
                    "chat_history": chat_history}

        # FOOD SUGGESTION (no state but clearly asking for food ideas)
        if is_food_suggestion_query(n):
            return {"kind": "lifestyle_stream",
                    "state": "looking for food ideas",
                    "category": "physical",
                    "question": question,
                    "chat_history": chat_history}

        # OOS (only after we've ruled out state/food queries)
        if looks_out_of_scope(n):
            return {"kind": "static", "result": make_result(_OUT_OF_SCOPE_REPLY, is_static=True)}

        # Modification request — needs history
        if is_modification(n) and chat_history != "(no previous conversation)":
            return {"kind": "llm_stream", "intent": "conversation",
                    "question": question, "chat_history": chat_history}

        # Summary request
        if is_summary_request(n):
            return {"kind": "llm_stream", "intent": "conversation",
                    "question": question, "chat_history": chat_history}

        # LLM router for the rest
        try:
            raw_intent = router_chain.invoke({"question": n}).strip().lower()
        except Exception:
            raw_intent = "recipe_lookup"
        valid = {"ingredient_inquiry","recipe_lookup","food_knowledge","out_of_scope"}
        intent = raw_intent if raw_intent in valid else "recipe_lookup"
        print(f"Intent: {raw_intent!r} → {intent!r}")

        if intent == "out_of_scope":
            return {"kind": "static", "result": make_result(_OUT_OF_SCOPE_REPLY, is_static=True)}

        if intent == "ingredient_inquiry":
            ingredient = extract_ingredient(n)
            # Reject numeric / too-short / pronoun "ingredients" that cause LLM to ramble
            if not _is_valid_food_target(ingredient):
                return {"kind": "static", "result": make_result(
                    "Could you tell me what specific food, ingredient, or dish you'd like to know about?",
                    is_static=True,
                )}
            return {"kind": "ingredient_inquiry_stream",
                    "ingredient": ingredient, "chat_history": chat_history}

        if intent == "food_knowledge":
            return {"kind": "llm_stream", "intent": "food_knowledge",
                    "question": question, "chat_history": chat_history}

        # recipe_lookup
        return {"kind": "recipe_rag_stream",
                "question": question, "chat_history": chat_history}

    # ── Non-streaming (collect from stream) ─────────────────────────────────
    def full_chain(x: dict) -> dict:
        chunks = []
        meta = None
        for event in stream_chain(x):
            if event["event"] == "token":
                chunks.append(event["data"])
            elif event["event"] == "done":
                meta = event["data"]
        text = "".join(chunks).strip()
        if meta is None:
            return make_result(text)
        meta["answer"] = text or meta.get("answer", "")
        return meta

    # ── Streaming generator ─────────────────────────────────────────────────
    def stream_chain(x: dict):
        """
        Yields dicts: {"event": "token"|"meta"|"done", "data": ...}
          - "token": partial text chunk
          - "meta":  metadata available before streaming (recipe_options, video)
          - "done":  final result dict
        """
        decision = _classify(x)
        kind = decision["kind"]

        # Static / mode_static: no streaming, emit as one chunk
        if kind in ("static", "mode_static"):
            result = decision["result"]
            yield {"event": "meta", "data": {
                "recipe_options": result["recipe_options"],
                "next_action":    result["next_action"],
                "pending_ingredient": result["pending_ingredient"],
                "video":          result["video"],
            }}
            yield {"event": "token", "data": result["answer"]}
            yield {"event": "done",  "data": result}
            return

        # Mode: full_recipe (state machine)
        if kind == "mode_llm_stream":
            recipe_name  = decision["recipe_name"]
            chat_history = decision["chat_history"]
            docs = retrieve(recipe_name, k_fetch=10)
            if not docs:
                msg = f"I couldn't find **{recipe_name}** in my database."
                yield {"event": "token", "data": msg}
                yield {"event": "done",  "data": make_result(msg, is_static=True)}
                return
            exact = [d for d in docs if normalize(d.metadata.get("recipe_name","")) == normalize(recipe_name)]
            chosen = (exact or docs)[:1]
            context = format_docs(chosen)
            video = youtube_search(recipe_name)
            yield {"event": "meta", "data": {"video": video}}
            collected = []
            try:
                for chunk in answer_chain.stream({
                    "chat_history": sanitize_history(chat_history),
                    "context":      context,
                    "question":     f"Give me the complete recipe for {recipe_name}",
                }):
                    collected.append(chunk)
                    yield {"event": "token", "data": chunk}
            except Exception as e:
                err = f"\n\n[error: {e}]"
                collected.append(err)
                yield {"event": "token", "data": err}
            yield {"event": "done", "data": make_result(
                "".join(collected), video=video)}
            return

        # Mode: nutrition (state machine)
        if kind == "mode_llm_stream_nutrition":
            ingredient   = extract_ingredient(normalize(decision["ingredient"]))
            chat_history = decision["chat_history"]
            collected = []
            try:
                for chunk in nutrition_chain.stream({
                    "chat_history": sanitize_history(chat_history),
                    "question":     f"Detailed nutrition and health information for {ingredient}",
                }):
                    collected.append(chunk)
                    yield {"event": "token", "data": chunk}
            except Exception as e:
                err = f"[error: {e}]"
                collected.append(err)
                yield {"event": "token", "data": err}
            yield {"event": "done", "data": make_result("".join(collected))}
            return

        # Ingredient inquiry (idle flow)
        if kind == "ingredient_inquiry_stream":
            ingredient   = decision["ingredient"]
            collected = []
            try:
                for chunk in ingredient_chain.stream({"ingredient": ingredient}):
                    collected.append(chunk)
                    yield {"event": "token", "data": chunk}
            except Exception as e:
                err = f"**{ingredient.title()}** is a food ingredient."
                collected.append(err)
                yield {"event": "token", "data": err}
            suffix = (
                "\n\n---\n\n**What would you like to know?**\n\n"
                "1️⃣  **Recipes** — Browse top recipes\n"
                "2️⃣  **Nutrition** — Health & calorie information"
            )
            collected.append(suffix)
            yield {"event": "token", "data": suffix}
            yield {"event": "done", "data": make_result(
                "".join(collected),
                next_action="awaiting_intent",
                pending_ingredient=ingredient,
            )}
            return

        # food_knowledge / conversation streams
        if kind == "llm_stream":
            intent       = decision["intent"]
            question     = decision["question"]
            chat_history = decision["chat_history"]
            standalone   = safe_rewrite(question, chat_history) if intent != "conversation" else question
            chain_use    = conversation_chain if intent == "conversation" else nutrition_chain
            collected = []
            try:
                for chunk in chain_use.stream({
                    "chat_history": sanitize_history(chat_history),
                    "question":     standalone,
                }):
                    collected.append(chunk)
                    yield {"event": "token", "data": chunk}
            except Exception as e:
                err = f"[error: {e}]"
                collected.append(err)
                yield {"event": "token", "data": err}
            yield {"event": "done", "data": make_result("".join(collected))}
            return

        # recipe_lookup (RAG)
        if kind == "recipe_rag_stream":
            question     = decision["question"]
            chat_history = decision["chat_history"]
            standalone   = safe_rewrite(question, chat_history)
            docs         = retrieve(standalone)[:5]
            context      = format_docs(docs)
            recipe_names = [d.metadata.get("recipe_name", "Unknown") for d in docs]
            # Only attach video when we have a CLEAR top match (much closer than rest)
            video = None
            if docs and len(recipe_names) >= 1:
                video = youtube_search(recipe_names[0])

            yield {"event": "meta", "data": {
                "recipe_options": recipe_names, "video": video,
            }}

            requested_lang = detect_language_request(question)
            effective_q = f"{question} (Respond in {requested_lang})" if requested_lang else question
            if not docs:
                effective_q += " [No DB match — use general cooking knowledge and say so.]"

            collected = []
            try:
                for chunk in answer_chain.stream({
                    "chat_history": sanitize_history(chat_history),
                    "context":      context,
                    "question":     effective_q,
                }):
                    collected.append(chunk)
                    yield {"event": "token", "data": chunk}
            except Exception as e:
                err = f"[error: {e}]"
                collected.append(err)
                yield {"event": "token", "data": err}

            yield {"event": "done", "data": make_result(
                "".join(collected), recipe_options=recipe_names, video=video,
            )}
            return

        # Want-to-eat: ingredient list directly from DB
        if kind == "want_to_eat":
            ingredient   = decision["ingredient"]
            chat_history = decision["chat_history"]
            result = handle_list_recipes(ingredient, chat_history)
            answer = result["answer"]
            yield {"event": "meta", "data": {
                "recipe_options": result["recipe_options"],
                "next_action":    result["next_action"],
                "pending_ingredient": result["pending_ingredient"],
            }}
            yield {"event": "token", "data": answer}
            yield {"event": "done",  "data": result}
            return

        # Lifestyle stream (emotional/physical/activity state)
        # Strategy: retrieve real DB recipes that match the state, then have LLM frame them.
        if kind == "lifestyle_stream":
            state    = decision["state"]
            category = decision["category"]
            question = decision["question"]
            chat_history = decision.get("chat_history", "(no previous conversation)")

            # Map state to a retrieval query for fetching real recipes from DB
            state_to_query = {
                "hungry":      "quick easy snack meal",
                "starving":    "filling hearty meal",
                "famished":    "filling hearty meal",
                "tired":       "energy boosting snack",
                "sleepy":      "light dinner before bed",
                "sad":         "comfort food warm soup",
                "stressed":    "calming light comfort food",
                "happy":       "celebration dessert sweet",
                "celebrating": "celebration festive dish",
                "sick":        "soup khichdi light easy digest",
                "having a cold": "soup hot spicy ginger",
                "thirsty":     "drinks beverage juice",
                "looking for food ideas": "popular indian recipe",
            }
            if category == "activity":
                if "before" in state or "pre" in state:
                    retrieval_q = "light pre workout snack energy"
                elif "after" in state or "post" in state:
                    retrieval_q = "recovery protein meal"
                elif "sleep" in state or "bed" in state:
                    retrieval_q = "light dinner before bed warm milk"
                else:
                    retrieval_q = "indian recipe meal"
            else:
                retrieval_q = state_to_query.get(state, "popular indian recipe")

            # Retrieve real recipes the user can actually pick
            docs = retrieve(retrieval_q, k_fetch=20)[:6]
            recipe_names = [d.metadata.get("recipe_name","Unknown") for d in docs]

            # Build a deterministic list response if we have results
            if recipe_names:
                # Use LLM ONLY for the opening empathetic line; the list is hard-coded
                opening_collected = []
                try:
                    opener_prompt_q = (
                        f"User state: {state} ({category}). "
                        "Write ONLY a single warm, empathetic opening sentence (max 20 words) "
                        "acknowledging their state. Do NOT list any food. Do NOT add explanation."
                    )
                    for chunk in llm.stream(opener_prompt_q):
                        text = chunk.content if hasattr(chunk, "content") else str(chunk)
                        opening_collected.append(text)
                        yield {"event": "token", "data": text}
                except Exception as e:
                    print(f"[Lifestyle opener ERROR] {e}")
                    opening_collected.append(_default_opener(state, category))
                    yield {"event": "token", "data": opening_collected[-1]}

                # Now append the deterministic list (NOT from LLM)
                list_text = "\n\n**Here are some recipes from my database that might suit:**\n\n"
                for i, name in enumerate(recipe_names, 1):
                    list_text += f"{i}. {name}\n"
                list_text += "\n_Type a number or recipe name to see the full recipe._"

                yield {"event": "token", "data": list_text}
                yield {"event": "meta", "data": {
                    "recipe_options": recipe_names,
                    "next_action":    "awaiting_recipe_selection",
                    "pending_ingredient": state,
                }}
                yield {"event": "done", "data": make_result(
                    "".join(opening_collected) + list_text,
                    recipe_options=recipe_names,
                    next_action="awaiting_recipe_selection",
                    pending_ingredient=state,
                )}
                return

            # No DB results — fall back to LLM-only suggestions
            collected = []
            try:
                for chunk in lifestyle_chain.stream({
                    "state":    state,
                    "category": category,
                    "question": question,
                }):
                    collected.append(chunk)
                    yield {"event": "token", "data": chunk}
            except Exception as e:
                err = f"[error: {e}]"
                collected.append(err)
                yield {"event": "token", "data": err}
            yield {"event": "done", "data": make_result("".join(collected))}
            return

        # Complaint stream
        if kind == "complaint_stream":
            question     = decision["question"]
            chat_history = decision["chat_history"]
            collected = []
            try:
                for chunk in complaint_chain.stream({
                    "chat_history": sanitize_history(chat_history),
                    "question":     question,
                }):
                    collected.append(chunk)
                    yield {"event": "token", "data": chunk}
            except Exception as e:
                err = f"[error: {e}]"
                collected.append(err)
                yield {"event": "token", "data": err}
            yield {"event": "done", "data": make_result("".join(collected))}
            return

        # Fallback
        yield {"event": "token", "data": "I had trouble processing that."}
        yield {"event": "done", "data": make_result("Error", is_static=True)}

    full_chain.stream = stream_chain
    return full_chain
