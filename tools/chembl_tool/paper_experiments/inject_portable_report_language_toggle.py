#!/usr/bin/env python3
"""Add a reusable language switcher to a portable Data Analytics report.

The input report must use language-prefixed artifact block IDs such as
``en_summary`` and ``zh_summary``. The language-specific title, description,
button label, and accessibility label live in a small JSON config so this
injector remains report-agnostic.
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


STYLE_ID = "portable-report-language-style"
BUTTON_ID = "portable-report-language-toggle"
SCRIPT_ID = "portable-report-language-script"
LANGUAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class LanguageText:
    title: str
    description: str
    button_label: str
    aria_label: str
    html_lang: str


@dataclass(frozen=True)
class LanguageConfig:
    default_language: str
    languages: Mapping[str, LanguageText]


def _required_string(payload: Mapping[str, Any], field: str, *, context: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def load_language_config(path: Path) -> LanguageConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("language config must be a JSON object")

    raw_languages = payload.get("languages")
    if not isinstance(raw_languages, dict) or len(raw_languages) < 2:
        raise ValueError("language config must define at least two languages")

    languages: dict[str, LanguageText] = {}
    for language, raw_text in raw_languages.items():
        if not isinstance(language, str) or not LANGUAGE_RE.fullmatch(language):
            raise ValueError(f"invalid language key: {language!r}")
        if not isinstance(raw_text, dict):
            raise ValueError(f"languages.{language} must be a JSON object")
        languages[language] = LanguageText(
            title=_required_string(raw_text, "title", context=f"languages.{language}"),
            description=_required_string(
                raw_text,
                "description",
                context=f"languages.{language}",
            ),
            button_label=_required_string(
                raw_text,
                "button_label",
                context=f"languages.{language}",
            ),
            aria_label=_required_string(
                raw_text,
                "aria_label",
                context=f"languages.{language}",
            ),
            html_lang=_required_string(
                raw_text,
                "html_lang",
                context=f"languages.{language}",
            ),
        )

    default_language = payload.get("default_language")
    if default_language not in languages:
        raise ValueError("default_language must match a key in languages")
    return LanguageConfig(
        default_language=str(default_language),
        languages=languages,
    )


def _json_for_html_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _build_style(language_order: Sequence[str]) -> str:
    visibility_rules = []
    for active_language in language_order:
        hidden_selectors = [
            (
                f'html[data-report-language="{active_language}"] '
                f'[data-artifact-block-id^="{other_language}_"]'
            )
            for other_language in language_order
            if other_language != active_language
        ]
        visibility_rules.append(",\n".join(hidden_selectors) + "{display:none!important}")

    return f"""
<style id="{STYLE_ID}">
#{BUTTON_ID}{{
  position:fixed;top:14px;right:18px;z-index:2147483647;
  appearance:none;border:1px solid rgba(100,116,139,.35);border-radius:999px;
  padding:8px 14px;background:#fff;color:#172033;font:600 13px/1.2
  -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  box-shadow:0 6px 24px rgba(15,23,42,.14);cursor:pointer;
  transition:transform .15s ease,box-shadow .15s ease;
}}
#{BUTTON_ID}:hover{{transform:translateY(-1px);box-shadow:0 8px 28px rgba(15,23,42,.2)}}
#{BUTTON_ID}:focus-visible{{outline:3px solid rgba(37,99,235,.28);outline-offset:2px}}
{chr(10).join(visibility_rules)}
@media (prefers-color-scheme:dark){{
  #{BUTTON_ID}{{background:#182033;color:#f8fafc;border-color:rgba(148,163,184,.4)}}
}}
@media print{{#{BUTTON_ID}{{display:none!important}}}}
</style>
"""


def _build_button(config: LanguageConfig) -> str:
    initial = config.languages[config.default_language]
    return (
        f'<button id="{BUTTON_ID}" type="button" '
        f'aria-label="{html_module.escape(initial.aria_label, quote=True)}" '
        f'aria-pressed="false">'
        f"{html_module.escape(initial.button_label)}</button>"
    )


def _build_script(config: LanguageConfig) -> str:
    browser_config = {
        "defaultLanguage": config.default_language,
        "order": list(config.languages),
        "languages": {
            key: {
                "title": value.title,
                "description": value.description,
                "buttonLabel": value.button_label,
                "ariaLabel": value.aria_label,
                "htmlLang": value.html_lang,
            }
            for key, value in config.languages.items()
        },
    }
    serialized_config = _json_for_html_script(browser_config)
    return f"""
<script id="{SCRIPT_ID}">
(() => {{
  const config = {serialized_config};
  const root = document.documentElement;
  const button = document.getElementById("{BUTTON_ID}");
  let language = config.defaultLanguage;
  let syncing = false;

  function syncChrome() {{
    if (syncing) return;
    syncing = true;
    const current = config.languages[language];
    document.title = current.title;
    document.querySelectorAll("h1").forEach((node) => {{
      const text = (node.textContent || "").trim();
      const isManaged = config.order.some((key) => text === config.languages[key].title);
      if (isManaged && text !== current.title) node.textContent = current.title;
    }});
    document.querySelectorAll(".portable-description").forEach((node) => {{
      if (node.textContent !== current.description) node.textContent = current.description;
    }});
    syncing = false;
  }}

  function setLanguage(nextLanguage) {{
    if (!config.languages[nextLanguage]) nextLanguage = config.defaultLanguage;
    language = nextLanguage;
    const current = config.languages[language];
    root.dataset.reportLanguage = language;
    root.lang = current.htmlLang;
    button.textContent = current.buttonLabel;
    button.setAttribute("aria-label", current.ariaLabel);
    button.setAttribute("aria-pressed", String(language !== config.defaultLanguage));
    syncChrome();
  }}

  button.addEventListener("click", () => {{
    const index = config.order.indexOf(language);
    setLanguage(config.order[(index + 1) % config.order.length]);
  }});
  new MutationObserver(syncChrome).observe(document.body, {{childList:true,subtree:true}});
  setLanguage(config.defaultLanguage);
}})();
</script>
"""


def inject_language_toggle(html_path: Path, config: LanguageConfig) -> None:
    html = html_path.read_text(encoding="utf-8")
    if any(marker in html for marker in (STYLE_ID, BUTTON_ID, SCRIPT_ID)):
        raise ValueError("language toggle has already been injected")
    if "</head>" not in html or "<body" not in html or "</body>" not in html:
        raise ValueError("input is not a complete HTML document")

    missing_languages = [
        language
        for language in config.languages
        if f'data-artifact-block-id="{language}_' not in html
    ]
    if missing_languages:
        joined = ", ".join(missing_languages)
        raise ValueError(f"report has no artifact blocks for language(s): {joined}")

    html = html.replace("</head>", _build_style(list(config.languages)) + "\n</head>", 1)
    body_match = re.search(r"<body(?:\s[^>]*)?>", html, flags=re.IGNORECASE)
    if body_match is None:
        raise ValueError("input HTML has no body start tag")
    body_end = body_match.end()
    html = html[:body_end] + "\n" + _build_button(config) + html[body_end:]
    html = html.replace("</body>", _build_script(config) + "\n</body>", 1)
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    inject_language_toggle(args.html, load_language_config(args.config))


if __name__ == "__main__":
    main()
