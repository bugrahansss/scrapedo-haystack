from unittest.mock import MagicMock, patch

import pytest
import requests
from haystack import Document, Pipeline
from haystack.utils import Secret

from scrapedo_haystack import ScrapeDoFetcher


def _response(text="# Hello", status=200, content_type="text/plain"):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    resp.headers = {"Content-Type": content_type}
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"{status}")
    return resp


@pytest.fixture
def fetcher():
    return ScrapeDoFetcher(api_key=Secret.from_token("test-token"))


@patch("scrapedo_haystack.fetcher.requests.get")
def test_returns_documents(mock_get, fetcher):
    mock_get.return_value = _response("# Example page")
    docs = fetcher.run(urls=["https://example.com"])["documents"]

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].content == "# Example page"
    assert docs[0].meta["url"] == "https://example.com"
    assert docs[0].meta["content_type"] == "text/markdown"


@patch("scrapedo_haystack.fetcher.requests.get")
def test_sends_scrapedo_params(mock_get):
    mock_get.return_value = _response()
    fetcher = ScrapeDoFetcher(
        api_key=Secret.from_token("test-token"),
        render=True,
        super_proxy=True,
        geo_code="us",
        extra_params={"waitUntil": "networkidle0"},
    )
    fetcher.run(urls=["https://example.com"])

    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.scrape.do/"
    assert kwargs["params"] == {
        "token": "test-token",
        "url": "https://example.com",
        "output": "markdown",
        "render": "true",
        "super": "true",
        "geoCode": "us",
        "waitUntil": "networkidle0",
    }


@patch("scrapedo_haystack.fetcher.requests.get")
def test_raw_output_has_no_output_param(mock_get):
    mock_get.return_value = _response("<html></html>", content_type="text/html")
    fetcher = ScrapeDoFetcher(api_key=Secret.from_token("t"), output="raw")
    docs = fetcher.run(urls=["https://example.com"])["documents"]

    assert "output" not in mock_get.call_args.kwargs["params"]
    assert docs[0].meta["content_type"] == "text/html"


@patch("scrapedo_haystack.fetcher.requests.get")
def test_failed_url_is_skipped(mock_get, fetcher):
    mock_get.side_effect = [_response("ok"), _response(status=500)]
    docs = fetcher.run(urls=["https://a.com", "https://b.com"])["documents"]
    assert [d.meta["url"] for d in docs] == ["https://a.com"]


@patch("scrapedo_haystack.fetcher.requests.get")
def test_raise_on_failure(mock_get):
    mock_get.return_value = _response(status=401)
    fetcher = ScrapeDoFetcher(api_key=Secret.from_token("t"), raise_on_failure=True)
    with pytest.raises(requests.HTTPError):
        fetcher.run(urls=["https://example.com"])


def test_empty_urls(fetcher):
    assert fetcher.run(urls=[]) == {"documents": []}


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("SCRAPEDO_API_KEY", raising=False)
    fetcher = ScrapeDoFetcher(api_key=Secret.from_env_var("SCRAPEDO_API_KEY", strict=False), raise_on_failure=True)
    with pytest.raises(ValueError, match="API key is missing"):
        fetcher.run(urls=["https://example.com"])


def test_invalid_output():
    with pytest.raises(ValueError):
        ScrapeDoFetcher(api_key=Secret.from_token("t"), output="pdf")


def test_serialization_roundtrip(monkeypatch):
    monkeypatch.setenv("SCRAPEDO_API_KEY", "env-token")
    fetcher = ScrapeDoFetcher(render=True, geo_code="de")
    data = fetcher.to_dict()

    assert "env-token" not in str(data)
    restored = ScrapeDoFetcher.from_dict(data)
    assert restored.render is True
    assert restored.geo_code == "de"
    assert restored.api_key.resolve_value() == "env-token"


@patch("scrapedo_haystack.fetcher.requests.get")
def test_works_in_pipeline_and_yaml(mock_get, monkeypatch):
    monkeypatch.setenv("SCRAPEDO_API_KEY", "env-token")
    mock_get.return_value = _response("# Page")

    from haystack.components.preprocessors import DocumentSplitter

    pipe = Pipeline()
    pipe.add_component("fetcher", ScrapeDoFetcher())
    pipe.add_component("splitter", DocumentSplitter(split_by="word", split_length=50))
    pipe.connect("fetcher.documents", "splitter.documents")

    result = pipe.run({"fetcher": {"urls": ["https://example.com"]}})
    assert result["splitter"]["documents"][0].content == "# Page"

    restored = Pipeline.loads(pipe.dumps(), allowed_modules=["scrapedo_haystack.fetcher"])
    assert "fetcher" in restored.graph.nodes
