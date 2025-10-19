"""Default tool definitions for common use cases."""

from ..schemas import ToolDefinition, ToolParameter, ToolRegistry

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
