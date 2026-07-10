# DeepResearch

DeepResearch is an AI-assisted OSINT research package. It generates search queries,
collects public web evidence through SerpApi, reads source content, and returns a
structured research result.

## Features

- Pluggable AI providers: OpenAI, Anthropic Claude, Gemini, Mistral, DeepSeek,
  xAI, OpenRouter, Perplexity, and local Ollama.
- SerpApi-backed search with Google, Bing, Yahoo, DuckDuckGo, and Yandex engines.
- Structured request and result models with Pydantic.
- Local cache for search, page reads, and AI responses.
- Router and service APIs for embedding in other Python applications.

## Installation

```bash
pip install .
```

For development with uv:

```bash
uv sync
uv run python -m unittest discover -s tests
```

## Configuration

Required:

```bash
SERPAPI_KEY=your_serpapi_key
LLM_PROVIDER=openai
MODEL_NAME=openai/gpt-4.1-mini
OPENAI_API_KEY=your_openai_api_key
```

PowerShell example:

```powershell
$env:SERPAPI_KEY = "your_serpapi_key"
$env:LLM_PROVIDER = "openai"
$env:MODEL_NAME = "openai/gpt-4.1-mini"
$env:OPENAI_API_KEY = "your_openai_api_key"
```

Supported provider variables:

| Provider | `LLM_PROVIDER` | API key variable | Example `MODEL_NAME` |
| --- | --- | --- | --- |
| OpenAI / GPT | `openai` | `OPENAI_API_KEY` | `openai/gpt-4.1-mini` |
| Claude | `anthropic` | `ANTHROPIC_API_KEY` | `anthropic/claude-3-5-sonnet-latest` |
| Gemini | `gemini` | `GEMINI_API_KEY` | `gemini/gemini-2.5-flash` |
| Mistral | `mistral` | `MISTRAL_API_KEY` | `mistral/mistral-small-latest` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| xAI | `xai` | `XAI_API_KEY` | `xai/grok-3-mini` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `openrouter/openai/gpt-4.1-mini` |
| Perplexity | `perplexity` | `PERPLEXITY_API_KEY` | `perplexity/sonar` |
| Ollama | `ollama` | none required | `ollama/llama3.1` |

Optional variables:

```bash
MAX_ROUNDS=2
MAX_SOURCES=20
MAX_SOURCES_PER_QUERY=5
MAX_QUERIES=12
QUERIES_PER_SUBQUESTION=2
MAX_CHARS_PER_SOURCE=6000
MAX_FINAL_CONTEXT_CHARS=16000
LLM_TIMEOUT=20
USE_CACHE=true
DEEPRESEARCH_HOME=/path/for/cache
```

Provider base URLs can also be overridden with variables such as
`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GEMINI_BASE_URL`, and
`OLLAMA_BASE_URL`.

## Usage

```python
from deepresearch import ResearchRequest, run_deep_research

request = ResearchRequest(
    topic="Ada Lovelace public biographies",
    engine="google",
    max_queries=4,
    max_sources=10,
)

result = run_deep_research(request)
print(result.model_dump_json(indent=2))
```

Router API:

```python
from deepresearch import route_from_dict

result = route_from_dict(
    {
        "topic": "Ada Lovelace public biographies",
        "engine": "google",
        "max_queries": 4,
        "max_sources": 10,
    }
)
```

## Example Result

```json
{
  "topic": "Ada Lovelace public biographies",
  "target_profile": {
    "raw_topic": "Ada Lovelace public biographies",
    "search_name": "Ada Lovelace",
    "identity_hints": []
  },
  "created_at": "2026-07-10T18:30:00+00:00",
  "duration_seconds": 8.421,
  "model": "openai/gpt-4.1-mini",
  "engine": "google",
  "settings": {
    "max_rounds": 2,
    "max_sources": 10,
    "max_queries": 4,
    "fetch_pages": true
  },
  "plan": {
    "topic": "Ada Lovelace",
    "subquestions": [
      "Which public profiles appear related to Ada Lovelace?",
      "Which public documents contain relevant mentions of Ada Lovelace?"
    ],
    "rationale": "Analysis plan for reviewing collected public OSINT search results."
  },
  "query_plans": [
    {
      "round": 1,
      "queries": [
        "\"Ada Lovelace\"",
        "Ada Lovelace biography"
      ]
    }
  ],
  "rounds": [
    {
      "round": 1,
      "queries": 2,
      "new_sources": 5,
      "sources_collected": 5,
      "gaps_remaining": 1
    }
  ],
  "gap_report": {
    "gaps": []
  },
  "sources": [
    {
      "title": "Ada Lovelace - Biography",
      "url": "https://example.org/ada-lovelace",
      "domain": "example.org",
      "snippet": "A public biography of Ada Lovelace...",
      "query": "\"Ada Lovelace\"",
      "subquestion": "Which public profiles appear related to Ada Lovelace?",
      "fetched": true,
      "source_type": "web"
    }
  ]
}
```

## Development

Run tests:

```bash
python -m unittest discover -s tests
```

Compile check:

```bash
python -m compileall deepresearch tests
```

## License

[MIT](LICENSE)
