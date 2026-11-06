# scrapedo-haystack

[![PyPI](https://img.shields.io/pypi/v/scrapedo-haystack.svg)](https://pypi.org/project/scrapedo-haystack/)

A [Haystack](https://haystack.deepset.ai/) component that fetches web pages through the [Scrape.do](https://scrape.do) API and returns them as Haystack `Document`s.

Scrape.do handles proxies, retries, CAPTCHAs and anti-bot protection, so the component works on pages that block normal HTTP requests. Pages can be returned as Markdown (ready for an LLM) or raw HTML.

## Installation

```bash
pip install scrapedo-haystack
```

Get an API token from [Scrape.do](https://scrape.do) and set it as an environment variable:

```bash
export SCRAPEDO_API_KEY="your-token"
```

## Usage

### Fetch pages

```python
from scrapedo_haystack import ScrapeDoFetcher

fetcher = ScrapeDoFetcher()
result = fetcher.run(urls=["https://example.com"])

doc = result["documents"][0]
print(doc.meta["url"])
print(doc.content)  # page content as Markdown
```

### Options

```python
fetcher = ScrapeDoFetcher(
    output="markdown",   # "markdown" or "raw" (original HTML)
    render=True,         # run JavaScript in a headless browser
    super_proxy=True,    # use residential and mobile IPs
    geo_code="us",       # route the request through a country
    extra_params={"waitUntil": "networkidle0"},  # any other Scrape.do parameter
    timeout=60,
    max_workers=4,       # URLs fetched in parallel
    raise_on_failure=False,  # skip failed URLs instead of raising
)
```

All Scrape.do API parameters are listed in the [Scrape.do documentation](https://scrape.do/documentation/).

### Use in a RAG pipeline

Fetch pages, split them, and answer a question from their content:

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ChatMessage

from scrapedo_haystack import ScrapeDoFetcher

template = [
    ChatMessage.from_user(
        """Answer the question using only these pages.

{% for doc in documents %}
Source: {{ doc.meta.url }}
{{ doc.content }}
{% endfor %}

Question: {{ question }}"""
    )
]

pipe = Pipeline()
pipe.add_component("fetcher", ScrapeDoFetcher(render=True))
pipe.add_component("splitter", DocumentSplitter(split_by="word", split_length=300))
pipe.add_component("prompt", ChatPromptBuilder(template=template, required_variables=["question"]))
pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))

pipe.connect("fetcher.documents", "splitter.documents")
pipe.connect("splitter.documents", "prompt.documents")
pipe.connect("prompt.prompt", "llm.messages")

result = pipe.run({
    "fetcher": {"urls": ["https://haystack.deepset.ai/overview/intro"]},
    "prompt": {"question": "What is Haystack?"},
})
print(result["llm"]["replies"][0].text)
```

This example also needs `OPENAI_API_KEY` set.

### Load a saved pipeline

Haystack 3 only loads components from allowed modules when reading a pipeline from YAML. Add this package to the allowlist:

```python
pipe = Pipeline.loads(yaml_text, allowed_modules=["scrapedo_haystack.fetcher"])
```

The API token is never written to the YAML; it is read from `SCRAPEDO_API_KEY` when the pipeline loads.

## Output

Each fetched URL becomes one `Document`:

| Field | Value |
|---|---|
| `content` | Page as Markdown, or raw HTML when `output="raw"` |
| `meta["url"]` | The requested URL |
| `meta["status_code"]` | HTTP status returned by Scrape.do |
| `meta["content_type"]` | `text/markdown`, or the page's content type for raw output |

## Development

```bash
pip install -e ".[test]"
pytest
```

## License

Apache-2.0
