"""GitHub Tools Spin Component for DeepFabric.

This component provides real GitHub API access with safety controls:
- Repo allowlisting: Only allowed repos can be accessed
- Write protection: Write operations disabled by default
- Helpful error messages when accessing non-allowed repos

Configuration via Spin variables:
- SPIN_VARIABLE_GITHUB_TOKEN: GitHub personal access token
- SPIN_VARIABLE_ALLOWED_REPOS: Comma-separated list of allowed repos (e.g., "org/repo1,org/repo2")
- SPIN_VARIABLE_ALLOW_WRITES: Set to "true" to enable write operations (default: false)
"""

import base64
import json

from urllib.parse import quote

from spin_sdk import http, variables
from spin_sdk.http import send

# =============================================================================
# Configuration
# =============================================================================

# HTTP status range used to determine successful responses:
HTTP_STATUS_SUCCESS_MIN = 200
HTTP_STATUS_SUCCESS_MAX = 300


def get_github_token() -> str:
    """Get GitHub token from Spin variables."""
    try:
        return variables.get("github_token")
    except Exception:
        return ""


def get_allowed_repos() -> set[str]:
    """Get set of allowed repositories from Spin variables."""
    try:
        repos_str = variables.get("allowed_repos")
        if not repos_str or repos_str == "{{ allowed_repos }}":
            return set()  # Empty means allow all (for backwards compat)
        return {r.strip().lower() for r in repos_str.split(",") if r.strip()}
    except Exception:
        return set()


def is_writes_allowed() -> bool:
    """Check if write operations are allowed."""
    try:
        allow_writes = variables.get("allow_writes")
        return allow_writes.lower() == "true"
    except Exception:
        return False


def validate_repo_access(owner: str, repo: str) -> tuple[bool, str]:
    """Validate that a repository is in the allowlist.

    Returns:
        (allowed, error_message) - If not allowed, returns helpful error message
    """
    allowed_repos = get_allowed_repos()

    # If no allowlist configured, allow all (backwards compatibility)
    if not allowed_repos:
        return True, ""

    repo_full = f"{owner}/{repo}".lower()

    if repo_full in allowed_repos:
        return True, ""

    # Build helpful error message
    allowed_list = ", ".join(sorted(allowed_repos))
    return False, f"Repository '{owner}/{repo}' is not in the allowed list. Allowed repositories: {allowed_list}"


def validate_write_operation(tool_name: str) -> tuple[bool, str]:
    """Validate that write operations are allowed.

    Returns:
        (allowed, error_message)
    """
    write_tools = {"gh_add_issue_comment", "gh_create_issue", "gh_create_pr"}

    if tool_name not in write_tools:
        return True, ""

    if is_writes_allowed():
        return True, ""

    return False, f"Write operation '{tool_name}' is disabled. Set SPIN_VARIABLE_ALLOW_WRITES=true to enable."


# =============================================================================
# GitHub API Client
# =============================================================================


def github_api_request(endpoint: str, method: str = "GET", body: dict | None = None) -> dict:
    """Make a request to the GitHub API using Spin's native HTTP client.

    Args:
        endpoint: API endpoint (e.g., "/repos/owner/repo/contents/path")
        method: HTTP method
        body: Request body for POST/PATCH requests

    Returns:
        API response as dict

    Raises:
        Exception: On API errors
    """
    token = get_github_token()
    if not token or token == "{{ github_token }}":  # noqa: S105
        raise ValueError("GitHub token not configured. Set SPIN_VARIABLE_GITHUB_TOKEN.")

    url = f"https://api.github.com{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "DeepFabric-Spin/1.0",
    }

    request_body = b""
    if body:
        request_body = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    try:
        # Perform the HTTP request using Spin's send helper by creating an http.Request
        request = http.Request(method=method, uri=url, headers=headers, body=request_body)
        response = send(request)
    except Exception as e:
        # Convert network/send errors into a ValueError
        raise ValueError(f"Network error: {e}") from e

    response_body = response.body.decode("utf-8") if response.body else ""

    # Successful responses
    if HTTP_STATUS_SUCCESS_MIN <= response.status < HTTP_STATUS_SUCCESS_MAX:
        if response_body:
            try:
                return json.loads(response_body)
            except json.JSONDecodeError:
                # Return raw body if it's not valid JSON
                return {"raw": response_body}
        return {}

    # Handle error responses
    try:
        error_json = json.loads(response_body) if response_body else {}
        message = error_json.get("message", response_body) if isinstance(error_json, dict) else response_body
    except json.JSONDecodeError:
        message = response_body or f"HTTP {response.status}"

    raise ValueError(f"GitHub API error ({response.status}): {message}")


# =============================================================================
# Tool Handlers
# =============================================================================


def handle_get_file_contents(args: dict) -> dict:  # noqa: PLR0911
    """Get contents of a file or directory from a repository."""
    owner = args.get("owner")
    repo = args.get("repo")
    path = args.get("path", "")
    ref = args.get("ref", "")

    if not owner or not repo:
        return {"success": False, "result": "Missing required: owner, repo", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    try:
        endpoint = f"/repos/{owner}/{repo}/contents/{path}"
        if ref:
            endpoint += f"?ref={ref}"

        data = github_api_request(endpoint)

        # Handle file vs directory response
        if isinstance(data, list):
            # Directory listing
            files = [{"name": f["name"], "type": f["type"], "path": f["path"]} for f in data]
            return {"success": True, "result": json.dumps(files)}
        # File content (base64 encoded)
        if data.get("encoding") == "base64" and data.get("content"):
            content = base64.b64decode(data["content"]).decode("utf-8")
            return {"success": True, "result": content, "sha": data.get("sha")}
        return {"success": True, "result": json.dumps(data)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_search_code(args: dict) -> dict:
    """Search for code across GitHub repositories."""
    query = args.get("query")
    if not query:
        return {"success": False, "result": "Missing required: query", "error_type": "InvalidArguments"}

    per_page = min(args.get("perPage", 10), 100)
    page = args.get("page", 1)

    # Note: Code search doesn't have repo restriction in the same way
    # Results will come from any public repos matching the query

    try:
        endpoint = f"/search/code?q={quote(query)}&per_page={per_page}&page={page}"
        data = github_api_request(endpoint)

        result = {
            "total_count": data.get("total_count", 0),
            "items": [
                {
                    "name": item["name"],
                    "path": item["path"],
                    "repository": item["repository"]["full_name"],
                    "url": item["html_url"],
                }
                for item in data.get("items", [])
            ],
        }
        return {"success": True, "result": json.dumps(result)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_search_repositories(args: dict) -> dict:
    """Search for GitHub repositories."""
    query = args.get("query")
    if not query:
        return {"success": False, "result": "Missing required: query", "error_type": "InvalidArguments"}

    per_page = min(args.get("perPage", 10), 100)
    page = args.get("page", 1)
    sort = args.get("sort", "")
    order = args.get("order", "desc")

    try:
        endpoint = f"/search/repositories?q={quote(query)}&per_page={per_page}&page={page}"
        if sort:
            endpoint += f"&sort={sort}&order={order}"

        data = github_api_request(endpoint)

        result = {
            "total_count": data.get("total_count", 0),
            "items": [
                {
                    "full_name": item["full_name"],
                    "description": item.get("description", ""),
                    "stars": item.get("stargazers_count", 0),
                    "language": item.get("language", ""),
                    "url": item["html_url"],
                }
                for item in data.get("items", [])
            ],
        }
        return {"success": True, "result": json.dumps(result)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_list_issues(args: dict) -> dict:
    """List issues in a repository."""
    owner = args.get("owner")
    repo = args.get("repo")

    if not owner or not repo:
        return {"success": False, "result": "Missing required: owner, repo", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    state = args.get("state", "open")
    per_page = min(args.get("perPage", 10), 100)

    try:
        endpoint = f"/repos/{owner}/{repo}/issues?state={state}&per_page={per_page}"
        data = github_api_request(endpoint)

        issues = [
            {
                "number": issue["number"],
                "title": issue["title"],
                "state": issue["state"],
                "author": issue["user"]["login"],
                "labels": [label["name"] for label in issue.get("labels", [])],
                "created_at": issue["created_at"],
            }
            for issue in data
            if "pull_request" not in issue  # Filter out PRs
        ]
        return {"success": True, "result": json.dumps(issues)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_get_issue(args: dict) -> dict:
    """Get details of a specific issue."""
    owner = args.get("owner")
    repo = args.get("repo")
    issue_number = args.get("issue_number")

    if not owner or not repo or issue_number is None:
        return {"success": False, "result": "Missing required: owner, repo, issue_number", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    try:
        endpoint = f"/repos/{owner}/{repo}/issues/{issue_number}"
        data = github_api_request(endpoint)

        issue = {
            "number": data["number"],
            "title": data["title"],
            "body": data.get("body", ""),
            "state": data["state"],
            "author": data["user"]["login"],
            "labels": [label["name"] for label in data.get("labels", [])],
            "assignees": [assignee["login"] for assignee in data.get("assignees", [])],
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
            "comments": data.get("comments", 0),
        }
        return {"success": True, "result": json.dumps(issue)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_list_pull_requests(args: dict) -> dict:
    """List pull requests in a repository."""
    owner = args.get("owner")
    repo = args.get("repo")

    if not owner or not repo:
        return {"success": False, "result": "Missing required: owner, repo", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    state = args.get("state", "open")
    per_page = min(args.get("perPage", 10), 100)

    try:
        endpoint = f"/repos/{owner}/{repo}/pulls?state={state}&per_page={per_page}"
        data = github_api_request(endpoint)

        prs = [
            {
                "number": pr["number"],
                "title": pr["title"],
                "state": pr["state"],
                "author": pr["user"]["login"],
                "head": pr["head"]["ref"],
                "base": pr["base"]["ref"],
                "draft": pr.get("draft", False),
                "created_at": pr["created_at"],
            }
            for pr in data
        ]
        return {"success": True, "result": json.dumps(prs)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_get_pull_request(args: dict) -> dict:
    """Get details of a specific pull request."""
    owner = args.get("owner")
    repo = args.get("repo")
    pull_number = args.get("pullNumber")

    if not owner or not repo or pull_number is None:
        return {"success": False, "result": "Missing required: owner, repo, pullNumber", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    try:
        endpoint = f"/repos/{owner}/{repo}/pulls/{pull_number}"
        data = github_api_request(endpoint)

        pr = {
            "number": data["number"],
            "title": data["title"],
            "body": data.get("body", ""),
            "state": data["state"],
            "author": data["user"]["login"],
            "head": data["head"]["ref"],
            "base": data["base"]["ref"],
            "draft": data.get("draft", False),
            "mergeable": data.get("mergeable"),
            "additions": data.get("additions", 0),
            "deletions": data.get("deletions", 0),
            "changed_files": data.get("changed_files", 0),
            "created_at": data["created_at"],
            "updated_at": data["updated_at"],
        }
        return {"success": True, "result": json.dumps(pr)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_list_commits(args: dict) -> dict:
    """List commits in a repository."""
    owner = args.get("owner")
    repo = args.get("repo")

    if not owner or not repo:
        return {"success": False, "result": "Missing required: owner, repo", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    sha = args.get("sha", "")
    per_page = min(args.get("perPage", 10), 100)

    try:
        endpoint = f"/repos/{owner}/{repo}/commits?per_page={per_page}"
        if sha:
            endpoint += f"&sha={sha}"

        data = github_api_request(endpoint)

        commits = [
            {
                "sha": commit["sha"][:7],
                "message": commit["commit"]["message"].split("\n")[0],  # First line only
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
            }
            for commit in data
        ]
        return {"success": True, "result": json.dumps(commits)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_get_commit(args: dict) -> dict:
    """Get details of a specific commit."""
    owner = args.get("owner")
    repo = args.get("repo")
    sha = args.get("sha")

    if not owner or not repo or not sha:
        return {"success": False, "result": "Missing required: owner, repo, sha", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    try:
        endpoint = f"/repos/{owner}/{repo}/commits/{sha}"
        data = github_api_request(endpoint)

        commit = {
            "sha": data["sha"],
            "message": data["commit"]["message"],
            "author": data["commit"]["author"]["name"],
            "date": data["commit"]["author"]["date"],
            "stats": data.get("stats", {}),
            "files": [
                {
                    "filename": f["filename"],
                    "status": f["status"],
                    "changes": f.get("changes", 0),
                }
                for f in data.get("files", [])[:20]  # Limit files returned
            ],
        }
        return {"success": True, "result": json.dumps(commit)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_list_branches(args: dict) -> dict:
    """List branches in a repository."""
    owner = args.get("owner")
    repo = args.get("repo")

    if not owner or not repo:
        return {"success": False, "result": "Missing required: owner, repo", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    per_page = min(args.get("perPage", 30), 100)

    try:
        endpoint = f"/repos/{owner}/{repo}/branches?per_page={per_page}"
        data = github_api_request(endpoint)

        branches = [
            {
                "name": branch["name"],
                "sha": branch["commit"]["sha"][:7],
                "protected": branch.get("protected", False),
            }
            for branch in data
        ]
        return {"success": True, "result": json.dumps(branches)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


def handle_add_issue_comment(args: dict) -> dict:
    """Add a comment to an issue (requires write permission)."""
    owner = args.get("owner")
    repo = args.get("repo")
    issue_number = args.get("issue_number")
    body = args.get("body")

    if not owner or not repo or issue_number is None or not body:
        return {"success": False, "result": "Missing required: owner, repo, issue_number, body", "error_type": "InvalidArguments"}

    # Validate repo access
    allowed, error = validate_repo_access(owner, repo)
    if not allowed:
        return {"success": False, "result": error, "error_type": "RepoNotAllowed"}

    # Validate write permission
    allowed, error = validate_write_operation("gh_add_issue_comment")
    if not allowed:
        return {"success": False, "result": error, "error_type": "WriteNotAllowed"}

    try:
        endpoint = f"/repos/{owner}/{repo}/issues/{issue_number}/comments"
        data = github_api_request(endpoint, method="POST", body={"body": body})

        result = {
            "id": data["id"],
            "url": data["html_url"],
            "created_at": data["created_at"],
        }
        return {"success": True, "result": json.dumps(result)}

    except ValueError as e:
        return {"success": False, "result": str(e), "error_type": "APIError"}
    except Exception as e:
        return {"success": False, "result": f"Unexpected error: {e}", "error_type": "Error"}


# =============================================================================
# Tool Router
# =============================================================================


TOOL_HANDLERS = {
    "gh_get_file_contents": handle_get_file_contents,
    "gh_search_code": handle_search_code,
    "gh_search_repositories": handle_search_repositories,
    "gh_list_issues": handle_list_issues,
    "gh_get_issue": handle_get_issue,
    "gh_list_pull_requests": handle_list_pull_requests,
    "gh_get_pull_request": handle_get_pull_request,
    "gh_list_commits": handle_list_commits,
    "gh_get_commit": handle_get_commit,
    "gh_list_branches": handle_list_branches,
    "gh_add_issue_comment": handle_add_issue_comment,
}


class IncomingHandler(http.IncomingHandler):
    def handle_request(self, request: http.Request) -> http.Response:
        """Handle incoming HTTP requests."""
        # Strip the /github prefix if present (Spin routing includes it)
        path = request.uri
        if path.startswith("/github"):
            path = path[7:]  # Remove "/github" prefix
        if not path:
            path = "/"
        method = request.method

        # Health check
        if path == "/health" and method == "GET":
            # Check if token is configured
            token = get_github_token()
            token_ok = bool(token and token != "{{ github_token }}")  # noqa: S105
            allowed_repos = get_allowed_repos()

            status = {
                "status": "healthy" if token_ok else "degraded",
                "component": "github",
                "mode": "real_api",
                "token_configured": token_ok,
                "allowed_repos": list(allowed_repos) if allowed_repos else "all",
                "writes_enabled": is_writes_allowed(),
            }
            return http.Response(
                200,
                {"content-type": "application/json"},
                bytes(json.dumps(status), "utf-8"),
            )

        # Component info
        if path == "/components" and method == "GET":
            return http.Response(
                200,
                {"content-type": "application/json"},
                bytes(json.dumps({
                    "components": ["github"],
                    "tools": list(TOOL_HANDLERS.keys()),
                    "mode": "real_api",
                }), "utf-8"),
            )

        # Execute tool
        if path == "/execute" and method == "POST":
            try:
                body = json.loads(request.body) # type: ignore
                tool = body.get("tool", "")
                args = body.get("args", {})

                if tool in TOOL_HANDLERS:
                    result = TOOL_HANDLERS[tool](args)
                else:
                    result = {
                        "success": False,
                        "result": f"Unknown tool: {tool}",
                        "error_type": "UnknownTool",
                    }

                return http.Response(
                    200,
                    {"content-type": "application/json"},
                    bytes(json.dumps(result), "utf-8"),
                )

            except json.JSONDecodeError as e:
                return http.Response(
                    400,
                    {"content-type": "application/json"},
                    bytes(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}), "utf-8"),
                )
            except Exception as e:
                return http.Response(
                    500,
                    {"content-type": "application/json"},
                    bytes(json.dumps({"success": False, "error": str(e)}), "utf-8"),
                )

        # 404 for unknown routes
        return http.Response(
            404,
            {"content-type": "application/json"},
            bytes(json.dumps({"error": "Not found"}), "utf-8"),
        )
