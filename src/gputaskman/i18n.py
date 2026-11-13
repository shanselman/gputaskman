"""Russian presentation, with English retained for the Python API and saved data.

Collection, evidence identifiers, paths and JSON keys are never translated. The
CLI selects Russian by default; library callers can use ``language('ru')``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
import inspect
import os
import re
import string
import sys

from .ru import MESSAGES

_language = ContextVar("gputaskman_language", default="en")


def current_language() -> str:
    return _language.get()


@contextmanager
def language(code: str):
    if code not in ("ru", "en"):
        raise ValueError(f"Unsupported language: {code}")
    token = _language.set(code)
    try:
        yield
    finally:
        _language.reset(token)


def _patterns():
    # Some diagnostics are produced by collectors and are also serialized. Match
    # their complete English templates only at the presentation boundary. Captures
    # retain names, paths, numbers and formatting verbatim.
    formatter = string.Formatter()
    for source, target in sorted(MESSAGES.items(), key=lambda item: -len(item[0])):
        if "{" not in source:
            continue
        parts, fields = [], []
        for literal, field, _spec, _conversion in formatter.parse(source):
            parts.append(re.escape(literal))
            if field is not None:
                parts.append("([\\s\\S]*?)")
                fields.append(field)
        yield re.compile("".join(parts)), fields, target


@lru_cache(maxsize=1)
def _compiled():
    return tuple(_patterns())


@lru_cache(maxsize=2048)
def _russian(message: str) -> str:
    direct = MESSAGES.get(message)
    if direct is not None:
        return direct
    for pattern, fields, target in _compiled():
        match = pattern.fullmatch(message)
        if match:
            values = dict(zip(fields, match.groups()))
            # Values have already been formatted by the producer.
            return "".join(literal + (values[field] if field is not None else "")
                           for literal, field, _spec, _conv in string.Formatter().parse(target))
    return message


def tr(message: str) -> str:
    return _russian(message) if current_language() == "ru" else message


def diagnostic(message: str) -> str:
    """Composite availability reasons can contain several independent messages."""
    translated = tr(message)
    if translated != message or current_language() == "en":
        return translated
    return " ".join(tr(part) for part in re.split(r"(?<=\.) (?=[A-Z])", message))


def kind_label(value: str) -> str:
    if value.startswith("likely ") and value.endswith(" (name)"):
        return tr("likely {0} (name)").format(tr(value[7:-7]))
    return tr(value)


def translate_bindings(bindings) -> None:
    for key, entries in bindings.key_to_bindings.items():
        bindings.key_to_bindings[key] = [replace(entry, description=tr(entry.description)) for entry in entries]


def inside(row) -> str:
    """Translate the explanation, preserving model names and paths verbatim."""
    if current_language() == "en":
        return row.inside
    if row.handy:
        if row.handy.last_model:
            return tr("Handy last log ({0}): {1}").format(
                tr(row.handy.last_event), row.handy.last_model.rsplit("/", 1)[-1])
        if row.handy.selected_model:
            return tr("Handy selected (not load proof): {0}").format(row.handy.selected_model.rsplit("/", 1)[-1])
    if row.llm and row.llm.server and row.llm.server.kind == "vllm" and row.llm.server.models:
        source = tr("parent API") if row.llm.server_source == "parent" else "API"
        return tr("reported ({0}): {1}").format(source, ", ".join(m["name"] for m in row.llm.server.models))
    if row.llm and row.llm.models:
        return ", ".join("+" + tr(model.kind) if model.kind in ("vision projector", "LoRA adapter")
                         else f"{row.llm.label(i)} {model.params_label} {model.quant}"
                         for i, model in enumerate(row.llm.models))
    if not row.llm and (row.detail.models or row.detail.candidates):
        models = sorted(row.detail.models, key=lambda m: not m.has_model_evidence)
        if models:
            return ", ".join(("" if m.has_model_evidence else tr("path-arg (unproven): "))
                             + (m.store_name or os.path.basename(m.path))
                             + (f" ({kind_label(m.kind_label)})" if m.kind_hint else "") for m in models[:2])
        from .llm import gb
        return tr("installed (unproven): ") + ", ".join(
            f"{m.store_name or os.path.basename(m.path)} {gb(m.size)}"
            + (f" ({kind_label(m.kind_label)})" if m.kind_hint else "") for m in row.detail.candidates[:2])
    return tr(row.inside)


def in_app_language(function):
    """Textual workers and thread-safe callbacks do not inherit ContextVars."""
    @wraps(function)
    def run(self, *args, **kwargs):
        with language(getattr(self, "language", current_language())):
            return function(self, *args, **kwargs)
    if inspect.isgeneratorfunction(function):
        @wraps(function)
        def generate(self, *args, **kwargs):
            iterator = function(self, *args, **kwargs)
            while True:
                with language(self.language):
                    try:
                        item = next(iterator)
                    except StopIteration:
                        return
                yield item
        return generate
    return run


def localized_app(cls):
    for name, value in list(vars(cls).items()):
        if inspect.isfunction(value):
            setattr(cls, name, in_app_language(value))
    return cls


def cli_language(function):
    """Scope the locale to a CLI invocation (including direct diff/record calls)."""
    @wraps(function)
    def run(*args, **kwargs):
        argv = args[0] if args else sys.argv[1:]
        code = os.environ.get("GPUTASKMAN_LANG", "ru")
        for i, arg in enumerate(argv):
            if arg.startswith("--lang="):
                code = arg.partition("=")[2]
            elif arg == "--lang" and i + 1 < len(argv):
                code = argv[i + 1]
        # argparse itself reports an invalid --lang value.
        code = code if code in ("ru", "en") else "ru"
        previous = argparse._
        # argparse uses %-style gettext messages, including trailing newlines.
        gettext = lambda message: tr(message.rstrip("\n")) + ("\n" if message.endswith("\n") else "")
        with language(code):
            argparse._ = gettext
            try:
                return function(*args, **kwargs)
            finally:
                argparse._ = previous
    return run
