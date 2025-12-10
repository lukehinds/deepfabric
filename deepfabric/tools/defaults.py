"""Default tool definitions for common use cases."""

from ..schemas import ToolDefinition, ToolParameter, ToolRegistry

# =============================================================================
# VFS (Virtual Filesystem) Tools - Spin component: "vfs"
# These tools have real execution via Spin
# =============================================================================

READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read content from a file",
    parameters=[
        ToolParameter(
            name="file_path",
            type="str",
            description="Path to the file to read",
            required=True,
        ),
    ],
    returns="File content as a string",
    category="filesystem",
    component="vfs",
)

WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description="Write content to a file",
    parameters=[
        ToolParameter(
            name="file_path",
            type="str",
            description="Path to the file to write",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="str",
            description="Content to write to the file",
            required=True,
        ),
    ],
    returns="Confirmation message with bytes written",
    category="filesystem",
    component="vfs",
)

LIST_FILES_TOOL = ToolDefinition(
    name="list_files",
    description="List all files in the current session",
    parameters=[],
    returns="JSON array of file paths",
    category="filesystem",
    component="vfs",
)

DELETE_FILE_TOOL = ToolDefinition(
    name="delete_file",
    description="Delete a file",
    parameters=[
        ToolParameter(
            name="file_path",
            type="str",
            description="Path to the file to delete",
            required=True,
        ),
    ],
    returns="Confirmation that file was deleted",
    category="filesystem",
    component="vfs",
)

# =============================================================================
# GitHub Tools - Spin component: "github"
# These tools have real execution via Spin with GitHub API access
# =============================================================================

GH_GET_FILE_CONTENTS = ToolDefinition(
    name="gh_get_file_contents",
    description="Get contents of a file or directory from a GitHub repository",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner (username or organization)",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="path",
            type="str",
            description="Path to file or directory (empty for root)",
            required=False,
            default="",
        ),
        ToolParameter(
            name="ref",
            type="str",
            description="Git reference (branch, tag, or commit SHA)",
            required=False,
        ),
    ],
    returns="File content or directory listing as JSON",
    category="github",
    component="github",
)

GH_SEARCH_CODE = ToolDefinition(
    name="gh_search_code",
    description="Search for code across GitHub repositories",
    parameters=[
        ToolParameter(
            name="query",
            type="str",
            description="Search query (supports GitHub code search syntax)",
            required=True,
        ),
        ToolParameter(
            name="perPage",
            type="int",
            description="Results per page (max 100)",
            required=False,
            default="10",
        ),
        ToolParameter(
            name="page",
            type="int",
            description="Page number for pagination",
            required=False,
            default="1",
        ),
    ],
    returns="JSON with total count and code search results",
    category="github",
    component="github",
)

GH_SEARCH_REPOSITORIES = ToolDefinition(
    name="gh_search_repositories",
    description="Search for GitHub repositories",
    parameters=[
        ToolParameter(
            name="query",
            type="str",
            description="Search query (supports GitHub repo search syntax)",
            required=True,
        ),
        ToolParameter(
            name="perPage",
            type="int",
            description="Results per page (max 100)",
            required=False,
            default="10",
        ),
        ToolParameter(
            name="page",
            type="int",
            description="Page number for pagination",
            required=False,
            default="1",
        ),
        ToolParameter(
            name="sort",
            type="str",
            description="Sort by: stars, forks, updated, help-wanted-issues",
            required=False,
        ),
        ToolParameter(
            name="order",
            type="str",
            description="Sort order: asc or desc",
            required=False,
            default="desc",
        ),
    ],
    returns="JSON with total count and repository results",
    category="github",
    component="github",
)

GH_LIST_ISSUES = ToolDefinition(
    name="gh_list_issues",
    description="List issues in a GitHub repository",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="state",
            type="str",
            description="Issue state: open, closed, or all",
            required=False,
            default="open",
        ),
        ToolParameter(
            name="perPage",
            type="int",
            description="Results per page (max 100)",
            required=False,
            default="10",
        ),
    ],
    returns="JSON array of issues with number, title, state, author, labels",
    category="github",
    component="github",
)

GH_GET_ISSUE = ToolDefinition(
    name="gh_get_issue",
    description="Get details of a specific GitHub issue",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="issue_number",
            type="int",
            description="Issue number",
            required=True,
        ),
    ],
    returns="JSON with issue details including body, labels, assignees, comments",
    category="github",
    component="github",
)

GH_LIST_PULL_REQUESTS = ToolDefinition(
    name="gh_list_pull_requests",
    description="List pull requests in a GitHub repository",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="state",
            type="str",
            description="PR state: open, closed, or all",
            required=False,
            default="open",
        ),
        ToolParameter(
            name="perPage",
            type="int",
            description="Results per page (max 100)",
            required=False,
            default="10",
        ),
    ],
    returns="JSON array of PRs with number, title, state, head/base branches",
    category="github",
    component="github",
)

GH_GET_PULL_REQUEST = ToolDefinition(
    name="gh_get_pull_request",
    description="Get details of a specific pull request",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="pullNumber",
            type="int",
            description="Pull request number",
            required=True,
        ),
    ],
    returns="JSON with PR details including body, mergeable status, diff stats",
    category="github",
    component="github",
)

GH_LIST_COMMITS = ToolDefinition(
    name="gh_list_commits",
    description="List commits in a GitHub repository",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="sha",
            type="str",
            description="Branch name or commit SHA to start from",
            required=False,
        ),
        ToolParameter(
            name="perPage",
            type="int",
            description="Results per page (max 100)",
            required=False,
            default="10",
        ),
    ],
    returns="JSON array of commits with SHA, message, author, date",
    category="github",
    component="github",
)

GH_GET_COMMIT = ToolDefinition(
    name="gh_get_commit",
    description="Get details of a specific commit",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="sha",
            type="str",
            description="Commit SHA (full or abbreviated)",
            required=True,
        ),
    ],
    returns="JSON with commit details including stats and changed files",
    category="github",
    component="github",
)

GH_LIST_BRANCHES = ToolDefinition(
    name="gh_list_branches",
    description="List branches in a GitHub repository",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="perPage",
            type="int",
            description="Results per page (max 100)",
            required=False,
            default="30",
        ),
    ],
    returns="JSON array of branches with name, SHA, protection status",
    category="github",
    component="github",
)

GH_ADD_ISSUE_COMMENT = ToolDefinition(
    name="gh_add_issue_comment",
    description="Add a comment to a GitHub issue (requires authentication)",
    parameters=[
        ToolParameter(
            name="owner",
            type="str",
            description="Repository owner",
            required=True,
        ),
        ToolParameter(
            name="repo",
            type="str",
            description="Repository name",
            required=True,
        ),
        ToolParameter(
            name="issue_number",
            type="int",
            description="Issue number",
            required=True,
        ),
        ToolParameter(
            name="body",
            type="str",
            description="Comment text (supports markdown)",
            required=True,
        ),
    ],
    returns="JSON with comment ID, URL, and creation timestamp",
    category="github",
    component="github",
)

# =============================================================================
# Mock Tools - Require user-provided Spin components
# These tools have no default component - users must provide their own
# =============================================================================

# Weather and Time tools
WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get current weather conditions for a location",
    parameters=[
        ToolParameter(
            name="location",
            type="str",
            description="City name or location (e.g., 'Paris', 'New York')",
            required=True,
        ),
        ToolParameter(
            name="time",
            type="str",
            description="Time period for weather data",
            required=False,
            default="now",
        ),
    ],
    returns="Weather data including temperature, conditions, and precipitation chance",
    category="information",
)

TIME_TOOL = ToolDefinition(
    name="get_time",
    description="Get current time for a timezone",
    parameters=[
        ToolParameter(
            name="timezone",
            type="str",
            description="Timezone name (e.g., 'UTC', 'America/New_York')",
            required=False,
            default="UTC",
        ),
    ],
    returns="Current time and date in the specified timezone",
    category="information",
)

# Search and Information tools
SEARCH_TOOL = ToolDefinition(
    name="search_web",
    description="Search the web for information",
    parameters=[
        ToolParameter(
            name="query",
            type="str",
            description="Search query terms",
            required=True,
        ),
        ToolParameter(
            name="limit",
            type="int",
            description="Maximum number of results to return",
            required=False,
            default="5",
        ),
    ],
    returns="List of web search results with titles and snippets",
    category="information",
)

NEWS_TOOL = ToolDefinition(
    name="get_news",
    description="Get recent news articles on a topic",
    parameters=[
        ToolParameter(
            name="topic",
            type="str",
            description="News topic or keyword",
            required=True,
        ),
        ToolParameter(
            name="limit",
            type="int",
            description="Number of articles to retrieve",
            required=False,
            default="3",
        ),
    ],
    returns="Recent news articles with headlines and summaries",
    category="information",
)

# Calculation and Analysis tools
CALCULATOR_TOOL = ToolDefinition(
    name="calculate",
    description="Evaluate mathematical expressions",
    parameters=[
        ToolParameter(
            name="expression",
            type="str",
            description="Mathematical expression to evaluate (e.g., '2 + 2', 'sin(pi/2)')",
            required=True,
        ),
    ],
    returns="Numerical result of the calculation",
    category="computation",
)

STOCK_TOOL = ToolDefinition(
    name="get_stock_price",
    description="Get current stock price for a symbol",
    parameters=[
        ToolParameter(
            name="symbol",
            type="str",
            description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL')",
            required=True,
        ),
    ],
    returns="Current stock price and basic market data",
    category="information",
)

# Communication tools
EMAIL_TOOL = ToolDefinition(
    name="send_email",
    description="Send an email to a recipient",
    parameters=[
        ToolParameter(
            name="to",
            type="str",
            description="Recipient email address",
            required=True,
        ),
        ToolParameter(
            name="subject",
            type="str",
            description="Email subject line",
            required=True,
        ),
        ToolParameter(
            name="body",
            type="str",
            description="Email message content",
            required=True,
        ),
    ],
    returns="Confirmation that email was sent successfully",
    category="communication",
)

TRANSLATE_TOOL = ToolDefinition(
    name="translate",
    description="Translate text to a target language",
    parameters=[
        ToolParameter(
            name="text",
            type="str",
            description="Text to translate",
            required=True,
        ),
        ToolParameter(
            name="target_lang",
            type="str",
            description="Target language code (e.g., 'es', 'fr', 'de')",
            required=True,
        ),
    ],
    returns="Translated text in the target language",
    category="communication",
)

# Navigation and Travel tools
DIRECTIONS_TOOL = ToolDefinition(
    name="get_directions",
    description="Get navigation directions between two locations",
    parameters=[
        ToolParameter(
            name="origin",
            type="str",
            description="Starting location",
            required=True,
        ),
        ToolParameter(
            name="destination",
            type="str",
            description="Destination location",
            required=True,
        ),
    ],
    returns="Turn-by-turn directions and estimated travel time",
    category="navigation",
)

TRAFFIC_TOOL = ToolDefinition(
    name="get_traffic",
    description="Get current traffic conditions between locations",
    parameters=[
        ToolParameter(
            name="origin",
            type="str",
            description="Starting location",
            required=True,
        ),
        ToolParameter(
            name="destination",
            type="str",
            description="Destination location",
            required=True,
        ),
    ],
    returns="Current traffic conditions and estimated travel time",
    category="navigation",
)

# Productivity tools
CALENDAR_TOOL = ToolDefinition(
    name="get_calendar_events",
    description="Get calendar events for a specific date",
    parameters=[
        ToolParameter(
            name="date",
            type="str",
            description="Date in YYYY-MM-DD format",
            required=True,
        ),
    ],
    returns="List of scheduled events for the specified date",
    category="productivity",
)

REMINDER_TOOL = ToolDefinition(
    name="set_reminder",
    description="Set a reminder for a specific time",
    parameters=[
        ToolParameter(
            name="message",
            type="str",
            description="Reminder message",
            required=True,
        ),
        ToolParameter(
            name="time",
            type="str",
            description="When to remind (e.g., '2024-01-15 14:30', 'tomorrow at 9am')",
            required=True,
        ),
    ],
    returns="Confirmation that reminder was set successfully",
    category="productivity",
)

# Create the default tool registry
DEFAULT_TOOL_REGISTRY = ToolRegistry(
    tools=[
        # VFS tools (have Spin components)
        READ_FILE_TOOL,
        WRITE_FILE_TOOL,
        LIST_FILES_TOOL,
        DELETE_FILE_TOOL,
        # GitHub tools (have Spin components)
        GH_GET_FILE_CONTENTS,
        GH_SEARCH_CODE,
        GH_SEARCH_REPOSITORIES,
        GH_LIST_ISSUES,
        GH_GET_ISSUE,
        GH_LIST_PULL_REQUESTS,
        GH_GET_PULL_REQUEST,
        GH_LIST_COMMITS,
        GH_GET_COMMIT,
        GH_LIST_BRANCHES,
        GH_ADD_ISSUE_COMMENT,
        # Mock tools (require user-provided components)
        WEATHER_TOOL,
        TIME_TOOL,
        SEARCH_TOOL,
        NEWS_TOOL,
        CALCULATOR_TOOL,
        STOCK_TOOL,
        EMAIL_TOOL,
        TRANSLATE_TOOL,
        DIRECTIONS_TOOL,
        TRAFFIC_TOOL,
        CALENDAR_TOOL,
        REMINDER_TOOL,
    ]
)

# VFS-only registry for use with the built-in Spin component
VFS_TOOL_REGISTRY = ToolRegistry(
    tools=[
        READ_FILE_TOOL,
        WRITE_FILE_TOOL,
        LIST_FILES_TOOL,
        DELETE_FILE_TOOL,
    ]
)

# GitHub-only registry for use with the GitHub Spin component
GITHUB_TOOL_REGISTRY = ToolRegistry(
    tools=[
        GH_GET_FILE_CONTENTS,
        GH_SEARCH_CODE,
        GH_SEARCH_REPOSITORIES,
        GH_LIST_ISSUES,
        GH_GET_ISSUE,
        GH_LIST_PULL_REQUESTS,
        GH_GET_PULL_REQUEST,
        GH_LIST_COMMITS,
        GH_GET_COMMIT,
        GH_LIST_BRANCHES,
        GH_ADD_ISSUE_COMMENT,
    ]
)


def get_default_tools(category: str | None = None) -> list[ToolDefinition]:
    """Get default tools, optionally filtered by category.

    Args:
        category: Optional category to filter by

    Returns:
        List of tool definitions
    """
    if category is None:
        return DEFAULT_TOOL_REGISTRY.tools
    return DEFAULT_TOOL_REGISTRY.get_tools_by_category(category)


def get_tool_categories() -> list[str]:
    """Get list of all tool categories."""
    categories = set()
    for tool in DEFAULT_TOOL_REGISTRY.tools:
        categories.add(tool.category)
    return sorted(categories)
