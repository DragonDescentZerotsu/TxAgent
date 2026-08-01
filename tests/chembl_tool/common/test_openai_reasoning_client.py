from types import SimpleNamespace

from tools.chembl_tool.common.openai_reasoning_client import OpenAICompatibleClient


def _response(
    content,
    *,
    tool_calls=None,
    response_id="response-1",
    reasoning_content="",
    reasoning=None,
):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        reasoning=reasoning,
        tool_calls=tool_calls or [],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(model_dump=lambda mode: {"total_tokens": 3}),
        model="test-model",
        id=response_id,
    )


def _client(*, enable_group_tools=True, max_tool_rounds=1):
    client = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    client.model = "test-model"
    client.max_tokens = 100
    client.enable_group_tools = enable_group_tools
    client.max_tool_rounds = max_tool_rounds
    client.reasoning_effort = ""
    client.enable_thinking = False
    return client


def test_chat_json_with_tools_executes_allowlisted_tool_then_returns_json():
    client = _client()
    tool_call = SimpleNamespace(
        id="call-1",
        type="function",
        function=SimpleNamespace(name="molecule_properties", arguments='{"query_smiles":"CCO"}'),
    )
    responses = iter([_response("", tool_calls=[tool_call]), _response('{"confidence":"high"}')])
    client._create_completion = lambda *args, **kwargs: next(responses)
    client.tool_service = SimpleNamespace(
        invoke_function_call=lambda call, allowed_tool_names: {
            "tool_name": call.function.name,
            "status": "ok",
            "content": "properties",
        }
    )

    result = client.chat_json_with_tools(
        [{"role": "user", "content": "analyze"}],
        tools=[{"type": "function"}],
        allowed_tool_names={"molecule_properties"},
    )

    assert result["content"] == {"confidence": "high"}
    assert result["tool_results"][0]["status"] == "ok"
    assert result["usage"] == {"total_tokens": 6}


def test_optional_group_tools_falls_back_to_plain_json_when_disabled():
    client = _client(enable_group_tools=False)
    client._create_completion = lambda *args, **kwargs: _response('{"transferability":"low"}')

    result = client.chat_json_with_optional_tools(
        [{"role": "user", "content": "analyze"}],
        tools=[{"type": "function"}],
        allowed_tool_names={"properties_compare"},
    )

    assert result["content"] == {"transferability": "low"}
    assert result["tool_calls"] == []


def test_chat_json_accepts_vllm_reasoning_field():
    client = _client()
    client._create_completion = lambda *args, **kwargs: _response(
        '{"confidence":"high"}',
        reasoning="provider reasoning",
    )

    result = client.chat_json([{"role": "user", "content": "analyze"}])

    assert result["reasoning_content"] == "provider reasoning"
    assert result["messages"][-1]["reasoning"] == "provider reasoning"
