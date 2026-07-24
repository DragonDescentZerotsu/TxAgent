from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.chembl_tool.paper_experiments.inject_portable_report_language_toggle import (
    BUTTON_ID,
    SCRIPT_ID,
    STYLE_ID,
    inject_language_toggle,
    load_language_config,
)


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "default_language": "en",
                "languages": {
                    "en": {
                        "title": "English title",
                        "description": "English description",
                        "button_label": "中文",
                        "aria_label": "Show Chinese version",
                        "html_lang": "en",
                    },
                    "zh": {
                        "title": "中文标题",
                        "description": "中文描述",
                        "button_label": "English",
                        "aria_label": "显示英文版本",
                        "html_lang": "zh-CN",
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_inject_language_toggle_uses_configured_prefixes(tmp_path: Path) -> None:
    config_path = tmp_path / "language_config.json"
    html_path = tmp_path / "report.html"
    _write_config(config_path)
    html_path.write_text(
        (
            "<!doctype html><html><head><title>English title</title></head><body>"
            '<section data-artifact-block-id="en_summary"><h1>English title</h1></section>'
            '<section data-artifact-block-id="zh_summary"><h1>中文标题</h1></section>'
            "</body></html>"
        ),
        encoding="utf-8",
    )

    inject_language_toggle(html_path, load_language_config(config_path))
    html = html_path.read_text(encoding="utf-8")

    assert f'id="{STYLE_ID}"' in html
    assert f'id="{BUTTON_ID}"' in html
    assert f'id="{SCRIPT_ID}"' in html
    assert 'html[data-report-language="en"] [data-artifact-block-id^="zh_"]' in html
    assert 'html[data-report-language="zh"] [data-artifact-block-id^="en_"]' in html
    assert '"defaultLanguage": "en"' in html
    assert "中文标题" in html


def test_inject_language_toggle_rejects_duplicate_injection(tmp_path: Path) -> None:
    config_path = tmp_path / "language_config.json"
    html_path = tmp_path / "report.html"
    _write_config(config_path)
    html_path.write_text(
        (
            "<html><head></head><body>"
            '<div data-artifact-block-id="en_a"></div>'
            '<div data-artifact-block-id="zh_a"></div>'
            "</body></html>"
        ),
        encoding="utf-8",
    )
    config = load_language_config(config_path)

    inject_language_toggle(html_path, config)
    with pytest.raises(ValueError, match="already been injected"):
        inject_language_toggle(html_path, config)


def test_load_language_config_rejects_invalid_language_key(tmp_path: Path) -> None:
    config_path = tmp_path / "language_config.json"
    _write_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["languages"]["zh<script>"] = payload["languages"].pop("zh")
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid language key"):
        load_language_config(config_path)
