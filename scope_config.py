"""
Scope configuration for the Stacking Benjamins RAG API.

Each scope corresponds to a paid guide on stackingbenjamins.com.
The book is included in every scope (universal content).

To add a new scope or update URLs/sources, edit this file only.
No app.py changes needed.
"""

SCOPE_CONFIG = {
    "tax-guide": {
        "display_name": "Tax Time Guide",
        "url": "https://www.stackingbenjamins.com/taxguide/",
        "allowed_sources": [
            "v14_SB_Tax Time Guide_May_2026.md",
            "stacked_book.md",
        ],
    },
    "college-guide": {
        "display_name": "Planning for College Guide",
        "url": "https://www.stackingbenjamins.com/collegeguide/",
        "allowed_sources": [
            "v9_SB_Planning for College_Guide_May_2026.md",
            "stacked_book.md",
        ],
    },
    "workplace-benefits": {
        "display_name": "Choosing Workplace Benefits Guide",
        "url": "https://www.stackingbenjamins.com/benefits/",
        "allowed_sources": [
            "v18_SB_Choosing Workplace Benefits Guide_May_2026.md",
            "stacked_book.md",
        ],
    },
}

# Reverse map: which scope owns each source file (used for upsell routing).
# The book is NOT in this map - it's universally included and never upsold.
SOURCE_TO_SCOPE = {
    "v14_SB_Tax Time Guide_May_2026.md": "tax-guide",
    "v9_SB_Planning for College_Guide_May_2026.md": "college-guide",
    "v18_SB_Choosing Workplace Benefits Guide_May_2026.md": "workplace-benefits",
}

# Threshold below which we refuse to answer.
# Validated empirically: in-scope queries score 0.74-0.94, out-of-scope 0.41-0.58.
# 0.70 gives a clean ~0.12 margin on both sides.
REFUSE_THRESHOLD = 0.70

# Number of chunks sent to Haiku for synthesis.
CHUNKS_FOR_SYNTHESIS = 3

# Refusal copy shown when a query genuinely can't be answered.
REFUSAL_MESSAGE = (
    "I couldn't find that trail on this map yet. "
    "I've added your question to our guide update list so we can chart it for future Stackers."
)


def get_upsell_message(target_scope: str) -> str:
    """Generate the cross-scope upsell message for a given target guide."""
    config = SCOPE_CONFIG[target_scope]
    return (
        f"Great question Stacker! We cover those topics in our "
        f"{config['display_name']} - you can find it at {config['url']}"
    )


# Topic overrides: specific phrases that should UPSELL to a target scope
# from listed source scopes, even if normal routing would answer.
#
# Each entry maps: trigger phrase -> {target_scope, upsell_from_scopes}
# If the user's scope is in upsell_from_scopes, we UPSELL to target_scope.
# If the user's scope is NOT in upsell_from_scopes (e.g., they're already
# on target_scope, OR on a scope that's allowed to answer), normal routing
# applies.
#
# Keep this list small. Only add entries with a specific UX problem.
TOPIC_OVERRIDES = [
    {
        "phrases": ["529"],
        "target_scope": "college-guide",
        "upsell_from_scopes": ["workplace-benefits"],
    },
]


def find_topic_override(query: str, current_scope: str):
    """
    Check if the query contains a phrase that's been hardcoded to UPSELL
    to a specific scope from the user's current scope. Returns the target
    scope name if matched, None otherwise.
    """
    query_lower = query.lower()
    for override in TOPIC_OVERRIDES:
        if current_scope not in override["upsell_from_scopes"]:
            continue  # Current scope is allowed to answer; no override
        for phrase in override["phrases"]:
            if phrase.lower() in query_lower:
                return override["target_scope"]
    return None