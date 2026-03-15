from Evaluator.schema_validator import validate_assistant_response


def test_validate_assistant_response_filters_wrapper_schema_warning():
    response = """
<tool_call>
{
  "name": "useTools",
  "arguments": {
    "context": {
      "workspaceId": "ws_1732300800000_atlasroll",
      "sessionId": "session_1732300800000_eval01234",
      "memory": "Search for the crash notes.",
      "goal": "Find the crash notes."
    },
    "calls": [
      {
        "agent": "searchManager",
        "tool": "searchContent",
        "params": {
          "query": "workspace crash",
          "limit": 10
        }
      }
    ]
  }
}
</tool_call>
""".strip()

    result = validate_assistant_response(response)

    assert [tool.name for tool in result.tool_calls] == ["searchManager_searchContent"]
    assert all("No schema found for this tool" not in issue.message for issue in result.issues)


def test_validate_assistant_response_preserves_wrapper_context_on_expanded_calls():
    response = """
<tool_call>
{
  "name": "useTools",
  "arguments": {
    "context": {
      "workspaceId": "ws_1732300800000_atlasroll",
      "sessionId": "session_1732300800000_eval01234",
      "memory": "Search for the crash notes.",
      "goal": "Find the crash notes."
    },
    "calls": [
      {
        "agent": "searchManager",
        "tool": "searchContent",
        "params": {
          "query": "workspace crash",
          "limit": 10
        }
      }
    ]
  }
}
</tool_call>
""".strip()

    result = validate_assistant_response(
        response,
        eval_context={
            "session_id": "session_1732300800000_eval01234",
            "workspace_id": "ws_1732300800000_atlasroll",
            "workspace_ids": [],
            "agent_ids": [],
        },
    )

    assert result.context_validation is not None
    assert result.context_validation["all_match"] is True
    assert result.tool_calls[0].arguments["context"]["goal"] == "Find the crash notes."
