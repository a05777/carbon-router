这是一个我闲的没事用AI写的玩意，不喜勿喷，谢谢你

# 《古法AI中转站》

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Anti-Ban](https://img.shields.io/badge/Security-99%25%20Ban--Proof-brightgreen)

我没钱买API用来VibeCoding了，但是我又不会写代码，而且AI Agent真的是用过就回不去，有问题它自己就能解决，所以有了这玩意，原理：
在AI Agent中把模型地址设置成127.0.0.1:8000，这个网关给AI Agent发来的东西注入系统提示词，显示在Web上，你复制，AI响应，你粘贴，Ai Agent干活，没毛病，哈哈（

以下是AI写的README原文：

# Web AI Clipboard Bridge

Web AI Clipboard Bridge 是一个本地、纯手工的协议转换器：OpenCode、Continue、Cline 或 Open WebUI 以 OpenAI-compatible API 发起请求，Bridge 把消息整理成可复制的 Markdown Prompt；你在 Gemini、ChatGPT 或 Claude 网页中完成对话，再把回答粘贴回本地控制台。Bridge 不会打开、读取或控制任何外部网页。

## 需求与架构

Bridge 刻意限制为单并发请求。一个 `POST /v1/chat/completions` 或 `POST /v1/responses` 请求进入后，`BridgeState` 持有它并进入 `WAITING_FOR_INPUT`；`/ui` 轮询到当前 Prompt。用户提交回答后，状态短暂进入 `PROCESSING`，挂起的 API 请求得到对应的 OpenAI JSON，之后回到 `IDLE`。第二个同时到达的请求立即收到 429。

`stream: true` 使用伪 Streaming：Bridge 仍然等待完整粘贴结果，再发送 SSE。Chat Completions 使用 `chat.completion.chunk` 和 `[DONE]`；Responses API 使用官方命名事件序列 `response.created`、`response.output_text.delta`、`response.completed`，不发送 `[DONE]`。

每个 Prompt 顶部都会注入 Bridge system prompt。传统 Web AI 不能直接访问文件、Shell 或网络，但可按 `Bridge Tool Protocol` 请求宿主 Coding Agent 执行工具。Gateway 会把协议转换为 Chat Completions `tool_calls` 或 Responses API `function_call`；工具结果会在 OpenCode 的下一轮请求中重新进入 Prompt。可用 `.env` 中的 `BRIDGE_SYSTEM_PROMPT` 覆盖基础提示词，工具协议本身始终保留。

当前实现的 OpenAI-compatible surface：

| Method | Path | 用途 |
| --- | --- | --- |
| `GET` | `/v1/models` | 列出模型 |
| `GET` | `/v1/models/{model}` | 获取模型 |
| `POST` | `/v1/chat/completions` | Chat Completions，支持 JSON/SSE |
| `POST` | `/v1/responses` | Responses API，支持 JSON/官方命名 SSE 事件 |

## 目录

```text
web_ai_clipboard_bridge/
  config.py       # .env 读取与 loopback 配置校验
  formatter.py    # messages -> Markdown Prompt
  models.py       # OpenAI 与 UI 请求模型
  responses.py    # Responses API 输入转换、响应对象与 SSE 事件
  tool_protocol.py # Web AI JSON 协议解析与工具调用校验
  state.py        # 单并发状态机、Future、超时和历史
  server.py       # FastAPI 路由、鉴权、JSON/SSE 响应
  main.py         # uvicorn 导入入口
  __main__.py     # python -m 启动入口
  static/index.html
tests/
```

## 启动

需要 Python 3.10+。

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，至少把 GATEWAY_API_KEY 改成随机长密钥
python -m web_ai_clipboard_bridge
```

打开 <http://127.0.0.1:8000/ui>。默认只接受 loopback `HOST`，并且启动时拒绝空的或占位符 API key。

## API 使用

查看模型：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer your-real-secret"
```

发起一个非流式请求。命令会保持连接，直到你在 UI 中复制 Prompt、交给网页 AI，并提交完整回答：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer your-real-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "web-ai-clipboard",
    "messages": [
      {"role": "system", "content": "You are a careful coding assistant."},
      {"role": "user", "content": "Explain this error and propose a fix."}
    ]
  }'
```

流式客户端可将 `stream` 设为 `true`：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer your-real-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"web-ai-clipboard","stream":true,"messages":[{"role":"user","content":"Say hello."}]}'
```

Responses API（新版 OpenCode/OpenAI provider 常用）：

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Authorization: Bearer your-real-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "web-ai-clipboard",
    "instructions": "You are a careful coding agent.",
    "input": [{
      "type": "message",
      "role": "user",
      "content": [{"type": "input_text", "text": "Inspect this bug."}]
    }]
  }'
```

错误也使用 OpenAI 风格的 `{"error": {"message", "type", "param", "code"}}` 结构：缺 key 是 401、忙碌是 429、等待超时是 504。

## Web AI 工具调用格式

当 Prompt 包含工具定义时，Web AI 会同时看到完整的 JSON Schema 和以下输出协议。需要调用工具时，将网页 AI 的整个 JSON 回答原样粘贴回 UI：

```json
{
  "bridge_version": "1",
  "type": "tool_calls",
  "tool_calls": [
    {
      "id": "call_read_file_1",
      "name": "read_file",
      "arguments": {"path": "README.md"}
    }
  ]
}
```

完成任务时返回：

```json
{
  "bridge_version": "1",
  "type": "final",
  "content": "最终回答，可包含 Markdown、代码块或 unified diff"
}
```

`name` 必须匹配请求中真实存在的工具，`arguments` 必须是符合工具 schema 的 JSON object。Gateway 只解析带 `bridge_version: "1"` 的完整对象；普通文本、普通 JSON 和无效/未知工具调用仍按 assistant 文本返回，不会被误执行。

## Provider 配置示例

以下示例中的 key 必须与 `.env` 的 `GATEWAY_API_KEY` 相同。

### OpenCode

在 `~/.config/opencode/opencode.json`（macOS/Linux；Windows 通常为 `%APPDATA%\\opencode\\opencode.json`）加入 provider。下面使用 `@ai-sdk/openai`，新版默认走 `/v1/responses`：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "clipboard": {
      "npm": "@ai-sdk/openai",
      "name": "Web AI Clipboard Bridge",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "your-real-secret"
      },
      "models": {
        "web-ai-clipboard": {"name": "Web AI Clipboard"}
      }
    }
  },
  "model": "clipboard/web-ai-clipboard"
}
```

也可以将 `npm` 改为 `@ai-sdk/openai-compatible`，该 provider 走 `/v1/chat/completions`。两条协议路径都由 Bridge 支持。`baseURL` 必须恰好为 `http://127.0.0.1:8000/v1`；如果日志显示 `/v1/v1/...`，说明客户端配置重复添加了 `/v1`。

### Continue

在 `~/.continue/config.yaml`（Windows：`%USERPROFILE%\\.continue\\config.yaml`）加入：

```yaml
name: Web AI Clipboard Bridge
version: 0.0.1
schema: v1
models:
  - name: Web AI Clipboard
    provider: openai
    model: web-ai-clipboard
    apiBase: http://127.0.0.1:8000/v1
    apiKey: your-real-secret
    roles:
      - chat
      - edit
```

### Cline

Cline 的 provider 配置通常保存在 VS Code 的 extension global state，而不是稳定的项目配置文件。打开 Cline **Settings → API Provider → OpenAI Compatible**，设置：

```text
Base URL: http://127.0.0.1:8000/v1
API Key:  your-real-secret
Model ID: web-ai-clipboard
```

### Open WebUI

在 **Admin Settings → Connections → OpenAI API** 新增连接：

```text
API Base URL: http://host.docker.internal:8000/v1  # Docker 部署时
API Key:      your-real-secret
```

如果 Open WebUI 在宿主机运行，Base URL 使用 `http://127.0.0.1:8000/v1`。容器场景需确保容器能访问宿主机，并且 Bridge 仍只绑定 loopback；可在 Docker 中使用 host networking 或一个受控的本地转发层。单并发限制意味着多个用户同时发送会得到 429。

## 测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

不启动服务也可以检查格式化器：

```bash
python -m compileall web_ai_clipboard_bridge
```

## 安全边界与限制

- Gateway 默认绑定 `127.0.0.1`，并拒绝配置为公网地址；`/v1/*` 每次都要求 `Authorization: Bearer ...`。
- `/api/submit` 不接受普通跨站表单，前端必须发送 `X-Bridge-UI: clipboard-bridge`；它只用于本机 UI，不替代 API key。
- Prompt 中的图片只会以引用 URL 呈现，Bridge 不下载图片。
- Gateway 不自行执行工具；它只把经过版本标记、工具名校验的 Bridge Tool Protocol 转成 OpenAI tool call，由 OpenCode、Continue 或 Cline 执行。未知工具和不符合协议的内容不会作为 tool call 转发。
- 超时由 `TIMEOUT_SECONDS` 控制，默认 1800 秒。进程重启会丢失当前挂起请求和内存中的历史。
