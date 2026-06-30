"""
Scope configuration for the Stacking Benjamins RAG API.

Each scope corresponds to a paid guide on stackingbenjamins.com.
The book is included in every scope (universal content).

Source filenames are matched by case-insensitive SUBSTRING patterns rather
than exact filenames. This means new versions of the guides can be uploaded
with any filename, as long as they contain the recognizable substring
(e.g., "tax time" for the tax guide). No code changes needed when guides
are updated, as long as the naming convention is preserved.

To add a new scope or update URLs, edit SCOPE_CONFIG below.
To adjust which filenames map to which scope, edit the source_patterns.
"""

# The substring that identifies the universal book (included in every scope).
BOOK_PATTERN = "stacked_book"

SCOPE_CONFIG = {
    "tax-guide": {
        "display_name": "Tax Time Guide",
        "url": "https://www.stackingbenjamins.com/taxguide/",
        # Filenames containing this substring (case-insensitive) belong to this scope.
        # The book pattern is automatically included as well - no need to list it here.
        "source_patterns": ["tax time"],
    },
    "college-guide": {
        "display_name": "Planning for College Guide",
        "url": "https://www.stackingbenjamins.com/collegeguide/",
        "source_patterns": ["planning for college"],
    },
    "workplace-benefits": {
        "display_name": "Choosing Workplace Benefits Guide",
        "url": "https://www.stackingbenjamins.com/benefits/",
        "source_patterns": ["workplace benefits"],
    },
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


def source_matches_scope(source_filename: str, scope: str) -> bool:
    """
    True if the given source filename belongs to the given scope.

    Match logic (case-insensitive substring):
      - Files containing BOOK_PATTERN belong to EVERY scope (universal).
      - Files containing any pattern in scope.source_patterns belong to that scope.

    Examples:
      source_matches_scope("v14_SB_Tax_Time_Guide_May_2026.md", "tax-guide") -> True
      source_matches_scope("v15_Tax_Time_Guide_2027.md", "tax-guide") -> True
      source_matches_scope("stacked_book.md", "tax-guide") -> True (universal)
      source_matches_scope("v9_Planning_for_College.md", "tax-guide") -> False
    """
    source_lower = source_filename.lower()
    if BOOK_PATTERN.lower() in source_lower:
        return True
    if scope not in SCOPE_CONFIG:
        return False
    for pattern in SCOPE_CONFIG[scope]["source_patterns"]:
        if pattern.lower() in source_lower:
            return True
    return False


def get_scope_for_source(source_filename: str):
    """
    Returns the scope name that "owns" a given source filename, for the
    purpose of upsell routing.

    The book is universal and not owned by any single scope, so it returns
    None for book files. This means out-of-scope book chunks never trigger
    an upsell - they're just inert.

    Examples:
      get_scope_for_source("v14_SB_Tax_Time_Guide_May_2026.md") -> "tax-guide"
      get_scope_for_source("stacked_book.md") -> None (no upsell for the book)
    """
    source_lower = source_filename.lower()
    if BOOK_PATTERN.lower() in source_lower:
        return None  # Book is universal, doesn't trigger upsells
    for scope, config in SCOPE_CONFIG.items():
        for pattern in config["source_patterns"]:
            if pattern.lower() in source_lower:
                return scope
    return None


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