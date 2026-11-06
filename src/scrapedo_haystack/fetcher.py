"""Haystack component that fetches web pages through the Scrape.do API."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import requests
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

SCRAPEDO_API_URL = "https://api.scrape.do/"

OutputFormat = Literal["markdown", "raw"]


@component
class ScrapeDoFetcher:
    """
    Fetches web pages through the Scrape.do API and returns them as Haystack `Document`s.

    Scrape.do handles proxies, retries and anti-bot protection. Set `render=True` for pages
    that need JavaScript, `super_proxy=True` for residential and mobile IPs, and `geo_code`
    to route the request through a specific country.

    Usage example:
    ```python
    from scrapedo_haystack import ScrapeDoFetcher

    fetcher = ScrapeDoFetcher()  # reads the SCRAPEDO_API_KEY environment variable
    result = fetcher.run(urls=["https://example.com"])
    print(result["documents"][0].content)
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SCRAPEDO_API_KEY"),
        output: OutputFormat = "markdown",
        render: bool = False,
        super_proxy: bool = False,
        geo_code: str | None = None,
        extra_params: dict[str, Any] | None = None,
        timeout: int = 60,
        max_workers: int = 4,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Create a ScrapeDoFetcher.

        :param api_key: Scrape.do API token. Defaults to the `SCRAPEDO_API_KEY` environment variable.
        :param output: `"markdown"` returns the page as Markdown, `"raw"` returns the original HTML.
        :param render: Render the page in a headless browser so JavaScript runs.
        :param super_proxy: Use residential and mobile proxies.
        :param geo_code: Two-letter country code to route the request through, for example `"us"`.
        :param extra_params: Any other Scrape.do API parameters, passed as-is (for example `{"waitUntil": "networkidle0"}`).
        :param timeout: Request timeout in seconds.
        :param max_workers: Number of URLs fetched in parallel.
        :param raise_on_failure: Raise an error when a URL fails. If `False`, failed URLs are logged and skipped.
        """
        if output not in ("markdown", "raw"):
            raise ValueError(f"output must be 'markdown' or 'raw', got {output!r}")
        self.api_key = api_key
        self.output = output
        self.render = render
        self.super_proxy = super_proxy
        self.geo_code = geo_code
        self.extra_params = extra_params or {}
        self.timeout = timeout
        self.max_workers = max_workers
        self.raise_on_failure = raise_on_failure

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            output=self.output,
            render=self.render,
            super_proxy=self.super_proxy,
            geo_code=self.geo_code,
            extra_params=self.extra_params,
            timeout=self.timeout,
            max_workers=self.max_workers,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScrapeDoFetcher":
        """Deserialize the component from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _build_params(self, url: str) -> dict[str, Any]:
        token = self.api_key.resolve_value()
        if not token:
            raise ValueError("Scrape.do API key is missing. Set the SCRAPEDO_API_KEY environment variable.")
        params: dict[str, Any] = {"token": token, "url": url}
        if self.output == "markdown":
            params["output"] = "markdown"
        if self.render:
            params["render"] = "true"
        if self.super_proxy:
            params["super"] = "true"
        if self.geo_code:
            params["geoCode"] = self.geo_code
        params.update(self.extra_params)
        return params

    def _fetch(self, url: str) -> Document | None:
        try:
            response = requests.get(SCRAPEDO_API_URL, params=self._build_params(url), timeout=self.timeout)
            response.raise_for_status()
        except Exception as error:
            if self.raise_on_failure:
                raise
            # never log the request URL: it contains the API token
            logger.warning("Scrape.do could not fetch %s: %s", url, type(error).__name__)
            return None

        meta = {
            "url": url,
            "status_code": response.status_code,
            "content_type": "text/markdown" if self.output == "markdown" else response.headers.get("Content-Type"),
        }
        return Document(content=response.text, meta=meta)

    @component.output_types(documents=list[Document])
    def run(self, urls: list[str]) -> dict[str, list[Document]]:
        """
        Fetch the given URLs through Scrape.do.

        :param urls: The web pages to fetch.
        :returns: A dictionary with `documents`: one Document per successfully fetched URL,
            with the page content and `url`, `status_code` and `content_type` in `meta`.
        """
        if not urls:
            return {"documents": []}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(urls))) as pool:
            results = list(pool.map(self._fetch, urls))
        return {"documents": [doc for doc in results if doc is not None]}
