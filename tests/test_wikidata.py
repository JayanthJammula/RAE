"""Tests for wikidata retry logic with mocked HTTP."""

import json
import pytest
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError


class TestSafeJsonGet:

    @patch("wiki_api.wikidata.ur.urlopen")
    @patch("wiki_api.wikidata.ur.Request")
    def test_success_on_first_try(self, mock_request, mock_urlopen):
        from wiki_api.wikidata import _safe_json_get

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"search": []}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _safe_json_get("http://example.com/api")
        assert result == {"search": []}
        assert mock_urlopen.call_count == 1

    @patch("wiki_api.wikidata.time.sleep")
    @patch("wiki_api.wikidata.ur.urlopen")
    @patch("wiki_api.wikidata.ur.Request")
    def test_retries_on_http_error(self, mock_request, mock_urlopen, mock_sleep):
        from wiki_api.wikidata import _safe_json_get

        # Fail twice, succeed third time
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            HTTPError("http://x", 500, "error", {}, None),
            URLError("timeout"),
            mock_resp,
        ]

        result = _safe_json_get("http://example.com/api", max_retries=3)
        assert result == {"ok": True}
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("wiki_api.wikidata.time.sleep")
    @patch("wiki_api.wikidata.ur.urlopen")
    @patch("wiki_api.wikidata.ur.Request")
    def test_returns_none_after_max_retries(self, mock_request, mock_urlopen, mock_sleep):
        from wiki_api.wikidata import _safe_json_get

        mock_urlopen.side_effect = URLError("connection refused")

        result = _safe_json_get("http://example.com/api", max_retries=3)
        assert result is None
        assert mock_urlopen.call_count == 3

    @patch("wiki_api.wikidata.ur.urlopen")
    @patch("wiki_api.wikidata.ur.Request")
    def test_returns_none_on_invalid_json(self, mock_request, mock_urlopen):
        from wiki_api.wikidata import _safe_json_get

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json at all"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _safe_json_get("http://example.com/api")
        assert result is None
