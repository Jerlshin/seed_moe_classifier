#!/usr/bin/env python3
"""
CODEBASE HEALTH ANALYZER v2
===========================

Professional repository analyzer for mixed Python / JS / frontend /
data / runtime / testing repositories.

Designed for large research/software-engineering repositories.

Primary goals
-------------
1. Measure REAL source-code size.
2. Prevent JSON/data/runtime artifacts from inflating LOC.
3. Perform AST-based analysis of Python.
4. Analyze Python architecture and imports.
5. Detect circular dependencies.
6. Measure module/function/class sizes.
7. Identify complexity hotspots.
8. Analyze tests separately from test fixtures.
9. Analyze repository composition.
10. Produce a detailed Markdown engineering report.

Usage
-----

From repository root:

    python3 codebase_analyzer_v2.py

Optional:

    python3 codebase_analyzer_v2.py --root .
    python3 codebase_analyzer_v2.py --output REPORT.md
    python3 codebase_analyzer_v2.py --root . --output reports/CODEBASE.md

No third-party Python packages are required.

Python 3.10+ recommended.
Python 3.11+ preferred.

Important
---------
This tool is an engineering telemetry tool.

It does NOT claim that:
    - more LOC = worse software
    - more comments = better software
    - more tests = automatically better software
    - lower complexity = automatically better architecture

The report provides evidence for engineering review.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import math
import os
import re
import sys
import tokenize
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ============================================================================
# VERSION
# ============================================================================

ANALYZER_VERSION = "2.0.0"


# ============================================================================
# DIRECTORY CLASSIFICATION
# ============================================================================

# Never descend into these directories.
ALWAYS_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".bzr",

    "__pycache__",

    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".hypothesis",

    ".venv",
    "venv",
    "env",
    ".env",
    "virtualenv",

    "node_modules",
    "bower_components",

    ".idea",
    ".vscode",

    ".cowork_tmp",
}


# These directories are not source, but are still analyzed for repository
# composition. Their files are classified as data/generated/artifacts rather
# than source.
DATA_DIR_NAMES = {
    "data",
    "datasets",
    "dataset",
    "raw",
    "processed",
    "cache",
    "caches",
}

RUNTIME_DIR_NAMES = {
    "runtime",
    "runs",
    "run",
    "logs",
    "reports",
    "manifests",
    "checkpoints",
    "snapshots",
}

ARTIFACT_DIR_NAMES = {
    "artifacts",
    "outputs",
    "output",
    "generated",
    "build",
    "dist",
    "out",
    "target",
}

TEST_DIR_NAMES = {
    "test",
    "tests",
    "__tests__",
    "testing",
}

DOC_DIR_NAMES = {
    "docs",
    "documentation",
}

SCRIPT_DIR_NAMES = {
    "scripts",
    "tools",
    "tooling",
}

UI_DIR_NAMES = {
    "ui",
    "frontend",
    "front-end",
    "client",
    "web",
    "website",
}

CONFIG_DIR_NAMES = {
    "config",
    "configs",
    "configuration",
    "infra",
    "infrastructure",
    "deployment",
    "deploy",
}


# ============================================================================
# FILE EXTENSIONS
# ============================================================================

PYTHON_EXTENSIONS = {
    ".py",
    ".pyw",
}

JAVASCRIPT_EXTENSIONS = {
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
}

TYPESCRIPT_EXTENSIONS = {
    ".ts",
    ".tsx",
}

FRONTEND_EXTENSIONS = {
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
}

MARKUP_EXTENSIONS = {
    ".xml",
    ".xsd",
    ".xsl",
}

DOCUMENTATION_EXTENSIONS = {
    ".md",
    ".mdx",
    ".rst",
    ".txt",
}

CONFIG_EXTENSIONS = {
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
}

DATA_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".ndjson",
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".arrow",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".tif",
    ".tiff",

    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".aac",

    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",

    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",

    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",

    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",

    ".pyc",
    ".pyo",

    ".class",
    ".jar",
    ".war",

    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",

    ".sqlite",
    ".db",
}

SPECIAL_FILES = {
    "Dockerfile": "Dockerfile",
    "Makefile": "Makefile",
    "CMakeLists.txt": "CMake",
    "Jenkinsfile": "Jenkinsfile",
    "Procfile": "Procfile",

    "requirements.txt": "Python requirements",
    "pyproject.toml": "Python project configuration",
    "Pipfile": "Python dependency configuration",
    "Pipfile.lock": "Python lockfile",
    "poetry.lock": "Poetry lockfile",

    "package.json": "Node package manifest",
    "package-lock.json": "Node lockfile",
    "yarn.lock": "Yarn lockfile",
    "pnpm-lock.yaml": "PNPM lockfile",

    "Cargo.toml": "Rust project configuration",
    "Cargo.lock": "Rust lockfile",

    "go.mod": "Go module",
    "go.sum": "Go lockfile",

    ".gitignore": "Git configuration",
    ".dockerignore": "Docker configuration",
    ".editorconfig": "Editor configuration",
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LOC:
    physical: int = 0
    code: int = 0
    blank: int = 0
    comments: int = 0

    def add(self, other: "LOC") -> None:
        self.physical += other.physical
        self.code += other.code
        self.blank += other.blank
        self.comments += other.comments


@dataclass
class FileRecord:
    path: str
    size_bytes: int

    language: str
    classification: str

    loc: LOC = field(default_factory=LOC)

    is_source: bool = False
    is_test_source: bool = False
    is_test_fixture: bool = False
    is_data: bool = False
    is_runtime: bool = False
    is_artifact: bool = False
    is_documentation: bool = False
    is_configuration: bool = False
    is_binary: bool = False

    python_analysis: Optional["PythonFileAnalysis"] = None


@dataclass
class FunctionInfo:
    qualname: str
    name: str
    path: str
    line_start: int
    line_end: int
    lines: int

    complexity: int = 1

    is_async: bool = False
    is_method: bool = False
    is_nested: bool = False

    arguments: int = 0
    annotated_arguments: int = 0
    has_return_annotation: bool = False

    decorators: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    qualname: str
    name: str
    path: str

    line_start: int
    line_end: int
    lines: int

    methods: int = 0
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)


@dataclass
class PythonFileAnalysis:
    path: str

    module_name: str = ""

    imports: set[str] = field(default_factory=set)
    from_imports: set[str] = field(default_factory=set)

    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)

    decorators: Counter = field(default_factory=Counter)
    exceptions: Counter = field(default_factory=Counter)

    todo_count: int = 0
    fixme_count: int = 0
    hack_count: int = 0

    bare_except_count: int = 0
    broad_except_count: int = 0

    comprehensions: int = 0
    async_functions: int = 0

    annotated_functions: int = 0
    total_functions: int = 0

    module_docstring: bool = False

    parse_error: Optional[str] = None


@dataclass
class RepositoryMetrics:
    files: int = 0
    bytes: int = 0

    physical: int = 0
    code: int = 0
    blank: int = 0
    comments: int = 0

    def add_file(self, record: FileRecord) -> None:
        self.files += 1
        self.bytes += record.size_bytes

        self.physical += record.loc.physical
        self.code += record.loc.code
        self.blank += record.loc.blank
        self.comments += record.loc.comments


@dataclass
class RepositoryAnalysis:
    root: Path

    all_files: list[FileRecord] = field(default_factory=list)

    source: RepositoryMetrics = field(default_factory=RepositoryMetrics)
    test_source: RepositoryMetrics = field(default_factory=RepositoryMetrics)
    test_fixtures: RepositoryMetrics = field(default_factory=RepositoryMetrics)

    data: RepositoryMetrics = field(default_factory=RepositoryMetrics)
    runtime: RepositoryMetrics = field(default_factory=RepositoryMetrics)
    artifacts: RepositoryMetrics = field(default_factory=RepositoryMetrics)

    documentation: RepositoryMetrics = field(default_factory=RepositoryMetrics)
    configuration: RepositoryMetrics = field(default_factory=RepositoryMetrics)

    binaries: RepositoryMetrics = field(default_factory=RepositoryMetrics)
    unknown: RepositoryMetrics = field(default_factory=RepositoryMetrics)

    languages: defaultdict = field(
        default_factory=lambda: defaultdict(RepositoryMetrics)
    )

    classifications: defaultdict = field(
        default_factory=lambda: defaultdict(RepositoryMetrics)
    )

    directories: defaultdict = field(
        default_factory=lambda: defaultdict(RepositoryMetrics)
    )

    python_files: list[FileRecord] = field(default_factory=list)

    python_functions: list[FunctionInfo] = field(default_factory=list)
    python_classes: list[ClassInfo] = field(default_factory=list)

    python_imports: defaultdict = field(
        default_factory=lambda: defaultdict(set)
    )

    python_analysis: list[PythonFileAnalysis] = field(
        default_factory=list
    )

    excluded_dirs: Counter = field(default_factory=Counter)

    errors: list[str] = field(default_factory=list)

    framework_signals: Counter = field(default_factory=Counter)


# ============================================================================
# BASIC HELPERS
# ============================================================================

def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"

    units = ["KB", "MB", "GB", "TB"]

    value = float(n)

    for unit in units:
        value /= 1024.0

        if value < 1024:
            return f"{value:.2f} {unit}"

    return f"{value:.2f} PB"


def percentage(part: float, total: float) -> str:
    if total == 0:
        return "0.0%"

    return f"{100.0 * part / total:.1f}%"


def safe_ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def relpath(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def path_parts(path: Path) -> set[str]:
    return {p.lower() for p in path.parts[:-1]}


def contains_any_part(path: Path, names: set[str]) -> bool:
    return bool(path_parts(path) & names)


def read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return None


def looks_binary(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True

    try:
        with path.open("rb") as f:
            sample = f.read(8192)

        if b"\x00" in sample:
            return True

        if not sample:
            return False

        suspicious = sum(
            1
            for b in sample
            if b < 8 or 14 <= b < 32
        )

        return suspicious / len(sample) > 0.20

    except Exception:
        return True


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_language(path: Path) -> str:
    name = path.name

    if name in SPECIAL_FILES:
        return SPECIAL_FILES[name]

    ext = path.suffix.lower()

    if ext in PYTHON_EXTENSIONS:
        return "Python"

    if ext in JAVASCRIPT_EXTENSIONS:
        return "JavaScript"

    if ext in TYPESCRIPT_EXTENSIONS:
        return "TypeScript"

    if ext in FRONTEND_EXTENSIONS:
        return ext.lstrip(".").upper()

    if ext in MARKUP_EXTENSIONS:
        return "XML"

    if ext in DOCUMENTATION_EXTENSIONS:
        return "Markdown" if ext in {".md", ".mdx"} else "Documentation"

    if ext in CONFIG_EXTENSIONS:
        return ext.lstrip(".").upper()

    if ext in DATA_EXTENSIONS:
        return "Data"

    return "Unknown"


# ============================================================================
# REPOSITORY CLASSIFICATION
# ============================================================================

def classify_file(
    root: Path,
    path: Path,
    language: str,
) -> str:

    rel = path.relative_to(root)
    parts = path_parts(rel)
    filename = path.name.lower()

    # ------------------------------------------------------------
    # Tests first
    # ------------------------------------------------------------

    in_tests = bool(parts & TEST_DIR_NAMES)

    looks_like_test = (
        filename.startswith("test_")
        or filename.endswith("_test.py")
        or ".test." in filename
        or ".spec." in filename
    )

    if in_tests:
        if language == "Python":
            return "Test Source"

        if language in {
            "JavaScript",
            "TypeScript",
        }:
            return "Test Source"

        if language == "Data":
            return "Test Fixture"

        return "Test Support"

    if looks_like_test:
        if language in {"Python", "JavaScript", "TypeScript"}:
            return "Test Source"

    # ------------------------------------------------------------
    # Runtime/generated/artifacts
    # ------------------------------------------------------------

    if parts & RUNTIME_DIR_NAMES:
        return "Runtime Artifact"

    if parts & ARTIFACT_DIR_NAMES:
        return "Generated Artifact"

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    if parts & DATA_DIR_NAMES:
        return "Data"

    if language == "Data":
        # JSON outside obvious source/config directories is normally
        # data, not source code.
        return "Data"

    # ------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------

    if parts & DOC_DIR_NAMES:
        return "Documentation"

    if language in {"Markdown", "Documentation"}:
        return "Documentation"

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------

    if parts & UI_DIR_NAMES:
        if language in {
            "JavaScript",
            "TypeScript",
            "HTML",
            "CSS",
            "SCSS",
            "SASS",
            "LESS",
            "VUE",
            "SVELTE",
        }:
            return "Frontend Source"

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    if parts & CONFIG_DIR_NAMES:
        return "Configuration"

    if language in {
        "YAML",
        "YML",
        "TOML",
        "INI",
        "CFG",
        "CONF",
        "Dockerfile",
        "Makefile",
        "CMake",
        "Python project configuration",
        "Python requirements",
        "Node package manifest",
        "Node lockfile",
        "PNPM lockfile",
        "Yarn lockfile",
        "Python lockfile",
        "Git configuration",
        "Docker configuration",
        "Editor configuration",
    }:
        return "Configuration"

    # package.json is configuration even outside config dirs.
    if filename in {
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
    }:
        return "Configuration"

    # ------------------------------------------------------------
    # Scripts
    # ------------------------------------------------------------

    if parts & SCRIPT_DIR_NAMES:
        if language in {
            "Python",
            "JavaScript",
            "TypeScript",
        }:
            return "Tooling / Script"

    # ------------------------------------------------------------
    # Actual source
    # ------------------------------------------------------------

    if language in {
        "Python",
        "JavaScript",
        "TypeScript",
        "HTML",
        "CSS",
        "SCSS",
        "SASS",
        "LESS",
        "VUE",
        "SVELTE",
    }:
        if contains_any_part(rel, UI_DIR_NAMES):
            return "Frontend Source"

        return "Source"

    return "Other"


# ============================================================================
# LOC ANALYSIS
# ============================================================================

def calculate_text_loc(
    text: str,
    language: str,
) -> LOC:

    lines = text.splitlines()

    result = LOC(
        physical=len(lines)
    )

    block_comment = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            result.blank += 1
            continue

        # Python
        if language == "Python":
            if stripped.startswith("#"):
                result.comments += 1
                continue

        # Shell
        if language == "Shell":
            if stripped.startswith("#"):
                result.comments += 1
                continue

        # JS / TS / CSS / C-style
        if language in {
            "JavaScript",
            "TypeScript",
            "CSS",
            "SCSS",
            "SASS",
            "LESS",
        }:
            if block_comment:
                result.comments += 1

                if "*/" in stripped:
                    block_comment = False

                continue

            if stripped.startswith("//"):
                result.comments += 1
                continue

            if stripped.startswith("/*"):
                result.comments += 1

                if "*/" not in stripped[2:]:
                    block_comment = True

                continue

        # HTML / XML
        if language in {
            "HTML",
            "XML",
        }:
            if stripped.startswith("<!--"):
                result.comments += 1
                continue

        result.code += 1

    return result


# ============================================================================
# PYTHON AST UTILITIES
# ============================================================================

def node_end_line(node: ast.AST, fallback: int) -> int:
    return getattr(node, "end_lineno", fallback)


def node_start_line(node: ast.AST) -> int:
    return getattr(node, "lineno", 1)


def node_lines(node: ast.AST) -> int:
    return max(
        1,
        node_end_line(node, node_start_line(node))
        - node_start_line(node)
        + 1,
    )


def decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return decorator_name(node.value) + "." + node.attr

    if isinstance(node, ast.Call):
        return decorator_name(node.func)

    return ast.dump(node, annotate_fields=False)


def expression_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        base = expression_name(node.value)

        if base:
            return f"{base}.{node.attr}"

        return node.attr

    if isinstance(node, ast.Call):
        return expression_name(node.func)

    return ""


def import_target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        base = import_target_name(node.value)

        if base:
            return f"{base}.{node.attr}"

        return node.attr

    return ""


# ============================================================================
# CYCLOMATIC COMPLEXITY
# ============================================================================

class ComplexityVisitor(ast.NodeVisitor):
    """
    Approximate McCabe cyclomatic complexity.

    Starts at 1 and increments for branching constructs.
    """

    def __init__(self) -> None:
        self.complexity = 1

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # A and B and C has approximately two decision points.
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        # Match cases represent branching.
        self.complexity += len(node.cases)
        self.generic_visit(node)


def function_complexity(node: ast.AST) -> int:
    visitor = ComplexityVisitor()

    for child in ast.iter_child_nodes(node):
        visitor.visit(child)

    return visitor.complexity


# ============================================================================
# PYTHON AST ANALYZER
# ============================================================================

class PythonAnalyzer(ast.NodeVisitor):

    def __init__(
        self,
        path: str,
        text: str,
    ) -> None:

        self.result = PythonFileAnalysis(path=path)

        self.text = text

        self.scope: list[str] = []

        self.class_scope: list[str] = []

        self.function_depth = 0

    # ------------------------------------------------------------------
    # Module
    # ------------------------------------------------------------------

    def visit_Module(self, node: ast.Module) -> None:

        if node.body:
            first = node.body[0]

            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                self.result.module_docstring = True

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:

        for alias in node.names:
            self.result.imports.add(alias.name)

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:

        module = node.module or ""

        if node.level:
            module = "." * node.level + module

        self.result.from_imports.add(module)

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Functions
    # ------------------------------------------------------------------

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:

        name = node.name

        if self.scope:
            qualname = ".".join(self.scope + [name])
        else:
            qualname = name

        args = list(node.args.posonlyargs)
        args += list(node.args.args)
        args += list(node.args.kwonlyargs)

        annotated = sum(
            1
            for arg in args
            if arg.annotation is not None
        )

        decorators = [
            decorator_name(d)
            for d in node.decorator_list
        ]

        info = FunctionInfo(
            qualname=qualname,
            name=name,
            path=self.result.path,
            line_start=node_start_line(node),
            line_end=node_end_line(node, node_start_line(node)),
            lines=node_lines(node),
            complexity=function_complexity(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=bool(self.class_scope),
            is_nested=self.function_depth > 0,
            arguments=len(args),
            annotated_arguments=annotated,
            has_return_annotation=node.returns is not None,
            decorators=decorators,
        )

        self.result.functions.append(info)

        self.result.total_functions += 1

        if (
            node.returns is not None
            or annotated > 0
        ):
            self.result.annotated_functions += 1

        if isinstance(node, ast.AsyncFunctionDef):
            self.result.async_functions += 1

        for decorator in decorators:
            self.result.decorators[decorator] += 1

        # Enter function scope.
        self.scope.append(name)
        self.function_depth += 1

        # Do not count nested function as part of parent's complexity
        # here because the function_complexity calculation already
        # traverses the subtree. This is intentionally conservative.
        for child in node.body:
            self.visit(child)

        self.function_depth -= 1
        self.scope.pop()

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        self._analyze_function(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self._analyze_function(node)

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:

        name = node.name

        if self.scope:
            qualname = ".".join(self.scope + [name])
        else:
            qualname = name

        decorators = [
            decorator_name(d)
            for d in node.decorator_list
        ]

        methods = sum(
            isinstance(
                child,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            for child in node.body
        )

        bases = [
            expression_name(base)
            for base in node.bases
        ]

        info = ClassInfo(
            qualname=qualname,
            name=name,
            path=self.result.path,
            line_start=node_start_line(node),
            line_end=node_end_line(node, node_start_line(node)),
            lines=node_lines(node),
            methods=methods,
            bases=[b for b in bases if b],
            decorators=decorators,
        )

        self.result.classes.append(info)

        for decorator in decorators:
            self.result.decorators[decorator] += 1

        self.scope.append(name)
        self.class_scope.append(name)

        for child in node.body:
            self.visit(child)

        self.class_scope.pop()
        self.scope.pop()

    # ------------------------------------------------------------------
    # Exceptions
    # ------------------------------------------------------------------

    def visit_ExceptHandler(
        self,
        node: ast.ExceptHandler,
    ) -> None:

        if node.type is None:
            self.result.bare_except_count += 1

        elif isinstance(node.type, ast.Name):
            if node.type.id in {
                "Exception",
                "BaseException",
            }:
                self.result.broad_except_count += 1

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Comprehensions
    # ------------------------------------------------------------------

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.result.comprehensions += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.result.comprehensions += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.result.comprehensions += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.result.comprehensions += 1
        self.generic_visit(node)


def analyze_python_file(
    path: Path,
    root: Path,
    text: str,
) -> PythonFileAnalysis:

    relative = relpath(root, path)

    try:
        tree = ast.parse(
            text,
            filename=str(path),
            type_comments=True,
        )

    except SyntaxError as exc:
        return PythonFileAnalysis(
            path=relative,
            parse_error=str(exc),
        )

    analyzer = PythonAnalyzer(
        path=relative,
        text=text,
    )

    analyzer.visit(tree)

    # ------------------------------------------------------------
    # TODO / FIXME / HACK
    # ------------------------------------------------------------

    upper = text.upper()

    analyzer.result.todo_count = len(
        re.findall(r"\bTODO\b", upper)
    )

    analyzer.result.fixme_count = len(
        re.findall(r"\bFIXME\b", upper)
    )

    analyzer.result.hack_count = len(
        re.findall(r"\bHACK\b", upper)
    )

    return analyzer.result


# ============================================================================
# MODULE NAME RESOLUTION
# ============================================================================

def python_module_name(
    root: Path,
    path: Path,
) -> str:

    relative = path.relative_to(root)

    if relative.suffix != ".py":
        return ""

    parts = list(relative.with_suffix("").parts)

    if parts and parts[-1] == "__init__":
        parts = parts[:-1]

    return ".".join(parts)


def resolve_import(
    current_module: str,
    imported: str,
    root_modules: set[str],
) -> Optional[str]:

    if not imported:
        return None

    # Absolute import.
    if not imported.startswith("."):
        candidates = []

        current = imported

        while current:
            candidates.append(current)

            if "." not in current:
                break

            current = current.rsplit(".", 1)[0]

        for candidate in candidates:
            if candidate in root_modules:
                return candidate

        # Try top-level package.
        top = imported.split(".")[0]

        if top in root_modules:
            return top

        return None

    # Relative import.
    dots = len(imported) - len(imported.lstrip("."))

    remainder = imported[dots:]

    current_parts = current_module.split(".")

    if current_parts:
        current_parts = current_parts[:-1]

    if dots > 1:
        current_parts = current_parts[: max(0, len(current_parts) - (dots - 1))]

    base = ".".join(current_parts)

    candidate = (
        f"{base}.{remainder}"
        if base and remainder
        else remainder or base
    )

    if candidate in root_modules:
        return candidate

    # Walk parents.
    parts = candidate.split(".")

    while parts:
        candidate = ".".join(parts)

        if candidate in root_modules:
            return candidate

        parts.pop()

    return None


# ============================================================================
# CYCLE DETECTION
# ============================================================================

def strongly_connected_components(
    graph: dict[str, set[str]],
) -> list[list[str]]:

    index = 0
    stack: list[str] = []

    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}

    on_stack: set[str] = set()

    result: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index

        indices[v] = index
        lowlinks[v] = index

        index += 1

        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, set()):

            if w not in indices:
                strongconnect(w)

                lowlinks[v] = min(
                    lowlinks[v],
                    lowlinks[w],
                )

            elif w in on_stack:
                lowlinks[v] = min(
                    lowlinks[v],
                    indices[w],
                )

        if lowlinks[v] == indices[v]:

            component = []

            while True:
                w = stack.pop()

                on_stack.remove(w)

                component.append(w)

                if w == v:
                    break

            result.append(component)

    for node in graph:
        if node not in indices:
            strongconnect(node)

    return result


# ============================================================================
# FRAMEWORK DETECTION
# ============================================================================

def detect_frameworks(
    root: Path,
    analysis: RepositoryAnalysis,
) -> None:

    files = {
        p.name
        for p in root.rglob("*")
        if p.is_file()
    }

    checks = {
        "Python / pyproject.toml": "pyproject.toml" in files,
        "Python requirements": "requirements.txt" in files,
        "Node.js / package.json": "package.json" in files,
        "Django": "manage.py" in files,
        "Next.js": any(
            name in files
            for name in {
                "next.config.js",
                "next.config.mjs",
                "next.config.ts",
            }
        ),
        "Vite": any(
            name in files
            for name in {
                "vite.config.js",
                "vite.config.ts",
            }
        ),
        "Angular": "angular.json" in files,
        "Docker": "Dockerfile" in files,
        "Docker Compose": any(
            name in files
            for name in {
                "docker-compose.yml",
                "docker-compose.yaml",
                "compose.yml",
                "compose.yaml",
            }
        ),
        "GitHub Actions": (
            (root / ".github" / "workflows").exists()
        ),
    }

    for name, present in checks.items():
        if present:
            analysis.framework_signals[name] += 1


# ============================================================================
# REPOSITORY SCANNER
# ============================================================================

def scan_repository(
    root: Path,
) -> RepositoryAnalysis:

    analysis = RepositoryAnalysis(
        root=root,
    )

    detect_frameworks(root, analysis)

    for current, dirs, filenames in os.walk(root):

        current_path = Path(current)

        retained_dirs = []

        for dirname in dirs:

            if dirname in ALWAYS_EXCLUDED_DIRS:
                analysis.excluded_dirs[dirname] += 1
                continue

            retained_dirs.append(dirname)

        dirs[:] = retained_dirs

        for filename in filenames:

            path = current_path / filename

            try:
                stat = path.stat()
            except OSError as exc:
                analysis.errors.append(
                    f"{path}: {exc}"
                )
                continue

            size = stat.st_size

            language = detect_language(path)

            # Binary files are classified but not parsed.
            if looks_binary(path):
                record = FileRecord(
                    path=relpath(root, path),
                    size_bytes=size,
                    language=language,
                    classification="Binary",
                    is_binary=True,
                )

                analysis.all_files.append(record)
                analysis.binaries.add_file(record)

                continue

            text = read_text(path)

            if text is None:
                record = FileRecord(
                    path=relpath(root, path),
                    size_bytes=size,
                    language=language,
                    classification="Unreadable",
                )

                analysis.all_files.append(record)
                analysis.unknown.add_file(record)

                analysis.errors.append(
                    f"Could not read {path}"
                )

                continue

            classification = classify_file(
                root,
                path,
                language,
            )

            loc = calculate_text_loc(
                text,
                language,
            )

            record = FileRecord(
                path=relpath(root, path),
                size_bytes=size,
                language=language,
                classification=classification,
                loc=loc,
            )

            # --------------------------------------------------------
            # Classification flags
            # --------------------------------------------------------

            record.is_source = classification in {
                "Source",
                "Frontend Source",
                "Tooling / Script",
            }

            record.is_test_source = classification == "Test Source"

            record.is_test_fixture = classification == "Test Fixture"

            record.is_data = classification == "Data"

            record.is_runtime = classification == "Runtime Artifact"

            record.is_artifact = classification == "Generated Artifact"

            record.is_documentation = classification == "Documentation"

            record.is_configuration = classification == "Configuration"

            # --------------------------------------------------------
            # Register
            # --------------------------------------------------------

            analysis.all_files.append(record)

            analysis.languages[language].add_file(record)

            analysis.classifications[classification].add_file(record)

            parent = str(
                path.relative_to(root).parent
            )

            if parent == ".":
                parent = "."

            analysis.directories[parent].add_file(record)

            # --------------------------------------------------------
            # Main buckets
            # --------------------------------------------------------

            if record.is_source:
                analysis.source.add_file(record)

            elif record.is_test_source:
                analysis.test_source.add_file(record)

            elif record.is_test_fixture:
                analysis.test_fixtures.add_file(record)

            elif record.is_data:
                analysis.data.add_file(record)

            elif record.is_runtime:
                analysis.runtime.add_file(record)

            elif record.is_artifact:
                analysis.artifacts.add_file(record)

            elif record.is_documentation:
                analysis.documentation.add_file(record)

            elif record.is_configuration:
                analysis.configuration.add_file(record)

            else:
                analysis.unknown.add_file(record)

            # --------------------------------------------------------
            # Python AST
            # --------------------------------------------------------

            if language == "Python":

                python_result = analyze_python_file(
                    path,
                    root,
                    text,
                )

                record.python_analysis = python_result

                analysis.python_files.append(record)

                analysis.python_analysis.append(
                    python_result
                )

                analysis.python_functions.extend(
                    python_result.functions
                )

                analysis.python_classes.extend(
                    python_result.classes
                )

    return analysis


# ============================================================================
# ARCHITECTURE ANALYSIS
# ============================================================================

@dataclass
class ArchitectureAnalysis:

    module_graph: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    cycles: list[list[str]] = field(default_factory=list)

    fan_in: Counter = field(default_factory=Counter)
    fan_out: Counter = field(default_factory=Counter)

    root_packages: Counter = field(default_factory=Counter)


def analyze_architecture(
    analysis: RepositoryAnalysis,
) -> ArchitectureAnalysis:

    result = ArchitectureAnalysis()

    module_names = set()

    module_to_file: dict[str, FileRecord] = {}

    for record in analysis.python_files:

        module = python_module_name(
            analysis.root,
            analysis.root / record.path,
        )

        if not module:
            continue

        module_names.add(module)

        module_to_file[module] = record

        result.module_graph.setdefault(
            module,
            set(),
        )

    for record in analysis.python_files:

        pa = record.python_analysis

        if pa is None:
            continue

        current_module = python_module_name(
            analysis.root,
            analysis.root / record.path,
        )

        if not current_module:
            continue

        imports = set(pa.imports)

        for imported in pa.from_imports:
            if imported:
                imports.add(imported)

        for imported in imports:

            resolved = resolve_import(
                current_module,
                imported,
                module_names,
            )

            if resolved and resolved != current_module:

                result.module_graph[current_module].add(
                    resolved
                )

    # Fan-out
    for module, dependencies in result.module_graph.items():
        result.fan_out[module] = len(dependencies)

        for dependency in dependencies:
            result.fan_in[dependency] += 1

    # Cycles
    result.cycles = [
        component
        for component in strongly_connected_components(
            result.module_graph
        )
        if len(component) > 1
    ]

    # Top-level package concentration.
    for module in result.module_graph:
        root_package = module.split(".")[0]

        result.root_packages[root_package] += 1

    return result


# ============================================================================
# HOTSPOT ANALYSIS
# ============================================================================

@dataclass
class HotspotAnalysis:

    large_files: list[FileRecord] = field(default_factory=list)

    huge_files: list[FileRecord] = field(default_factory=list)

    large_functions: list[FunctionInfo] = field(
        default_factory=list
    )

    huge_functions: list[FunctionInfo] = field(
        default_factory=list
    )

    high_complexity_functions: list[FunctionInfo] = field(
        default_factory=list
    )

    large_classes: list[ClassInfo] = field(
        default_factory=list
    )

    huge_classes: list[ClassInfo] = field(
        default_factory=list
    )


def analyze_hotspots(
    analysis: RepositoryAnalysis,
) -> HotspotAnalysis:

    result = HotspotAnalysis()

    python_source = [
        r
        for r in analysis.python_files
        if r.is_source
    ]

    result.large_files = sorted(
        python_source,
        key=lambda r: r.loc.code,
        reverse=True,
    )

    result.huge_files = [
        r
        for r in result.large_files
        if r.loc.code >= 2000
    ]

    result.large_functions = sorted(
        analysis.python_functions,
        key=lambda f: f.lines,
        reverse=True,
    )

    result.large_functions = [
        f
        for f in result.large_functions
        if f.lines >= 50
    ]

    result.huge_functions = [
        f
        for f in result.large_functions
        if f.lines >= 150
    ]

    result.high_complexity_functions = sorted(
        [
            f
            for f in analysis.python_functions
            if f.complexity >= 10
        ],
        key=lambda f: f.complexity,
        reverse=True,
    )

    result.large_classes = sorted(
        [
            c
            for c in analysis.python_classes
            if c.lines >= 200
        ],
        key=lambda c: c.lines,
        reverse=True,
    )

    result.huge_classes = [
        c
        for c in result.large_classes
        if c.lines >= 500
    ]

    return result


# ============================================================================
# QUALITY SIGNALS
# ============================================================================

@dataclass
class QualitySignals:

    todo: int = 0
    fixme: int = 0
    hack: int = 0

    bare_except: int = 0
    broad_except: int = 0

    parse_errors: int = 0

    functions: int = 0
    annotated_functions: int = 0

    classes: int = 0

    module_docstrings: int = 0


def calculate_quality_signals(
    analysis: RepositoryAnalysis,
) -> QualitySignals:

    result = QualitySignals()

    for pa in analysis.python_analysis:

        result.todo += pa.todo_count
        result.fixme += pa.fixme_count
        result.hack += pa.hack_count

        result.bare_except += pa.bare_except_count
        result.broad_except += pa.broad_except_count

        result.functions += pa.total_functions
        result.annotated_functions += pa.annotated_functions

        result.classes += len(pa.classes)

        if pa.module_docstring:
            result.module_docstrings += 1

        if pa.parse_error:
            result.parse_errors += 1

    return result


# ============================================================================
# REPORT HELPERS
# ============================================================================

def metrics_table_row(
    name: str,
    metrics: RepositoryMetrics,
    total: RepositoryMetrics,
) -> str:

    return (
        f"| {md(name)} | "
        f"{metrics.files:,} | "
        f"{metrics.bytes and human_bytes(metrics.bytes) or '0 B'} | "
        f"{metrics.physical:,} | "
        f"{metrics.code:,} | "
        f"{percentage(metrics.code, total.code)} |"
    )


def source_metrics_total(
    analysis: RepositoryAnalysis,
) -> RepositoryMetrics:

    result = RepositoryMetrics()

    result.files = (
        analysis.source.files
        + analysis.test_source.files
    )

    result.bytes = (
        analysis.source.bytes
        + analysis.test_source.bytes
    )

    result.physical = (
        analysis.source.physical
        + analysis.test_source.physical
    )

    result.code = (
        analysis.source.code
        + analysis.test_source.code
    )

    result.blank = (
        analysis.source.blank
        + analysis.test_source.blank
    )

    result.comments = (
        analysis.source.comments
        + analysis.test_source.comments
    )

    return result


# ============================================================================
# MARKDOWN REPORT
# ============================================================================

def generate_report(
    analysis: RepositoryAnalysis,
    architecture: ArchitectureAnalysis,
    hotspots: HotspotAnalysis,
    signals: QualitySignals,
) -> str:

    now = dt.datetime.now().astimezone()

    # Actual maintainable production source.
    production = analysis.source

    test = analysis.test_source

    # All source excluding fixtures/data/runtime.
    source_plus_tests = source_metrics_total(
        analysis
    )

    report: list[str] = []

    # ==================================================================
    # Header
    # ==================================================================

    report.append("# Codebase Health & Architecture Report")
    report.append("")
    report.append(
        "> AST-driven repository analysis with explicit separation "
        "of source code, tests, fixtures, data, runtime artifacts, "
        "generated artifacts, documentation, and configuration."
    )
    report.append("")

    report.append(
        f"**Repository:** `{analysis.root}`"
    )
    report.append(
        f"**Generated:** `{now.isoformat(timespec='seconds')}`"
    )
    report.append(
        f"**Analyzer version:** `{ANALYZER_VERSION}`"
    )
    report.append("")

    # ==================================================================
    # Executive Summary
    # ==================================================================

    report.append("## 1. Executive Summary")
    report.append("")

    report.append("| Metric | Value |")
    report.append("|---|---:|")

    report.append(
        f"| Repository files scanned | "
        f"{len(analysis.all_files):,} |"
    )

    report.append(
        f"| **Production source files** | "
        f"**{production.files:,}** |"
    )

    report.append(
        f"| **Production code LOC** | "
        f"**{production.code:,}** |"
    )

    report.append(
        f"| Test source files | "
        f"{test.files:,} |"
    )

    report.append(
        f"| Test source LOC | "
        f"{test.code:,} |"
    )

    report.append(
        f"| Test fixtures | "
        f"{analysis.test_fixtures.files:,} files / "
        f"{human_bytes(analysis.test_fixtures.bytes)} |"
    )

    report.append(
        f"| Runtime artifacts | "
        f"{analysis.runtime.files:,} files / "
        f"{human_bytes(analysis.runtime.bytes)} |"
    )

    report.append(
        f"| Generated artifacts | "
        f"{analysis.artifacts.files:,} files / "
        f"{human_bytes(analysis.artifacts.bytes)} |"
    )

    report.append(
        f"| Data | "
        f"{analysis.data.files:,} files / "
        f"{human_bytes(analysis.data.bytes)} |"
    )

    report.append(
        f"| Documentation | "
        f"{analysis.documentation.files:,} files / "
        f"{analysis.documentation.code:,} LOC |"
    )

    report.append(
        f"| Configuration | "
        f"{analysis.configuration.files:,} files |"
    )

    report.append("")

    # ==================================================================
    # Repository composition
    # ==================================================================

    report.append("## 2. Repository Composition")
    report.append("")

    report.append(
        "| Classification | Files | Size | Physical LOC | "
        "Code LOC | Code Share |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    total_composition = RepositoryMetrics()

    for record in analysis.all_files:
        total_composition.add_file(record)

    rows = [
        ("Production Source", analysis.source),
        ("Test Source", analysis.test_source),
        ("Test Fixtures", analysis.test_fixtures),
        ("Data", analysis.data),
        ("Runtime Artifacts", analysis.runtime),
        ("Generated Artifacts", analysis.artifacts),
        ("Documentation", analysis.documentation),
        ("Configuration", analysis.configuration),
        ("Binary", analysis.binaries),
        ("Unknown / Other", analysis.unknown),
    ]

    for name, metrics in rows:
        report.append(
            metrics_table_row(
                name,
                metrics,
                total_composition,
            )
        )

    report.append("")

    report.append(
        "**Important:** repository size is deliberately separated "
        "from maintainable source size. Large JSON manifests, datasets, "
        "fixtures, reports, and runtime artifacts are not counted as "
        "production code."
    )
    report.append("")

    # ==================================================================
    # True source statistics
    # ==================================================================

    report.append("## 3. True Source-Code Statistics")
    report.append("")

    report.append("| Metric | Production | Tests |")
    report.append("|---|---:|---:|")

    report.append(
        f"| Files | {production.files:,} | {test.files:,} |"
    )

    report.append(
        f"| Physical LOC | {production.physical:,} | "
        f"{test.physical:,} |"
    )

    report.append(
        f"| Code LOC | {production.code:,} | "
        f"{test.code:,} |"
    )

    report.append(
        f"| Comment LOC | {production.comments:,} | "
        f"{test.comments:,} |"
    )

    report.append(
        f"| Blank LOC | {production.blank:,} | "
        f"{test.blank:,} |"
    )

    test_ratio = safe_ratio(
        test.code,
        production.code,
    )

    report.append(
        f"| Test / production LOC | "
        f"{test_ratio:.3f} ({test_ratio * 100:.2f}%) | |"
    )

    report.append("")

    # ==================================================================
    # Language
    # ==================================================================

    report.append("## 4. Language Distribution")
    report.append("")

    report.append(
        "| Language | Files | Physical LOC | Code LOC | "
        "Share of Repository Source |"
    )

    report.append(
        "|---|---:|---:|---:|---:|"
    )

    language_rows = sorted(
        analysis.languages.items(),
        key=lambda item: item[1].code,
        reverse=True,
    )

    for language, metrics in language_rows:

        # Only show languages that have meaningful content.
        if metrics.files == 0:
            continue

        report.append(
            f"| {md(language)} | "
            f"{metrics.files:,} | "
            f"{metrics.physical:,} | "
            f"{metrics.code:,} | "
            f"{percentage(metrics.code, source_plus_tests.code)} |"
        )

    report.append("")

    # ==================================================================
    # Python AST
    # ==================================================================

    report.append("## 5. Python AST Analysis")
    report.append("")

    report.append("| Metric | Value |")
    report.append("|---|---:|")

    report.append(
        f"| Python files | {len(analysis.python_files):,} |"
    )

    report.append(
        f"| Python functions | {signals.functions:,} |"
    )

    report.append(
        f"| Python classes | {signals.classes:,} |"
    )

    report.append(
        f"| Async functions | "
        f"{sum(pa.async_functions for pa in analysis.python_analysis):,} |"
    )

    report.append(
        f"| Annotated functions | "
        f"{signals.annotated_functions:,} |"
    )

    annotation_ratio = safe_ratio(
        signals.annotated_functions,
        signals.functions,
    )

    report.append(
        f"| Function annotation ratio | "
        f"{annotation_ratio * 100:.1f}% |"
    )

    report.append(
        f"| Module docstrings | "
        f"{signals.module_docstrings:,} |"
    )

    report.append(
        f"| TODO markers | {signals.todo:,} |"
    )

    report.append(
        f"| FIXME markers | {signals.fixme:,} |"
    )

    report.append(
        f"| HACK markers | {signals.hack:,} |"
    )

    report.append(
        f"| Bare except handlers | "
        f"{signals.bare_except:,} |"
    )

    report.append(
        f"| Broad Exception/BaseException handlers | "
        f"{signals.broad_except:,} |"
    )

    report.append(
        f"| AST parse errors | "
        f"{signals.parse_errors:,} |"
    )

    report.append("")

    # ==================================================================
    # Complexity
    # ==================================================================

    report.append("## 6. Complexity & Size Hotspots")
    report.append("")

    report.append(
        "| Metric | Count |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        f"| Python files ≥ 1,000 LOC | "
        f"{sum(r.loc.code >= 1000 for r in analysis.python_files):,} |"
    )

    report.append(
        f"| Python files ≥ 2,000 LOC | "
        f"{sum(r.loc.code >= 2000 for r in analysis.python_files):,} |"
    )

    report.append(
        f"| Functions ≥ 50 LOC | "
        f"{len(hotspots.large_functions):,} |"
    )

    report.append(
        f"| Functions ≥ 150 LOC | "
        f"{len(hotspots.huge_functions):,} |"
    )

    report.append(
        f"| Functions complexity ≥ 10 | "
        f"{len(hotspots.high_complexity_functions):,} |"
    )

    report.append(
        f"| Classes ≥ 200 LOC | "
        f"{len(hotspots.large_classes):,} |"
    )

    report.append(
        f"| Classes ≥ 500 LOC | "
        f"{len(hotspots.huge_classes):,} |"
    )

    report.append("")

    # ------------------------------------------------------------------
    # Largest Python files
    # ------------------------------------------------------------------

    report.append("### 6.1 Largest Python Files")
    report.append("")

    report.append(
        "| Rank | File | Code LOC | Physical LOC |"
    )

    report.append(
        "|---:|---|---:|---:|"
    )

    for i, record in enumerate(
        hotspots.large_files[:30],
        start=1,
    ):

        report.append(
            f"| {i} | `{md(record.path)}` | "
            f"{record.loc.code:,} | "
            f"{record.loc.physical:,} |"
        )

    report.append("")

    # ------------------------------------------------------------------
    # Largest functions
    # ------------------------------------------------------------------

    report.append("### 6.2 Largest Python Functions")
    report.append("")

    report.append(
        "| Rank | Function | File | Lines | Complexity |"
    )

    report.append(
        "|---:|---|---|---:|---:|"
    )

    for i, function in enumerate(
        hotspots.large_functions[:50],
        start=1,
    ):

        report.append(
            f"| {i} | `{md(function.qualname)}` | "
            f"`{md(function.path)}` | "
            f"{function.lines:,} | "
            f"{function.complexity} |"
        )

    report.append("")

    # ------------------------------------------------------------------
    # Complexity hotspots
    # ------------------------------------------------------------------

    report.append("### 6.3 Highest Cyclomatic Complexity")
    report.append("")

    report.append(
        "| Rank | Function | File | Complexity | Lines |"
    )

    report.append(
        "|---:|---|---|---:|---:|"
    )

    for i, function in enumerate(
        hotspots.high_complexity_functions[:50],
        start=1,
    ):

        report.append(
            f"| {i} | `{md(function.qualname)}` | "
            f"`{md(function.path)}` | "
            f"{function.complexity} | "
            f"{function.lines:,} |"
        )

    report.append("")

    # ------------------------------------------------------------------
    # Large classes
    # ------------------------------------------------------------------

    report.append("### 6.4 Largest Python Classes")
    report.append("")

    report.append(
        "| Rank | Class | File | Lines | Methods |"
    )

    report.append(
        "|---:|---|---|---:|---:|"
    )

    for i, cls in enumerate(
        hotspots.large_classes[:50],
        start=1,
    ):

        report.append(
            f"| {i} | `{md(cls.qualname)}` | "
            f"`{md(cls.path)}` | "
            f"{cls.lines:,} | "
            f"{cls.methods:,} |"
        )

    report.append("")

    # ==================================================================
    # Architecture
    # ==================================================================

    report.append("## 7. Python Architecture")
    report.append("")

    report.append(
        f"**Python modules discovered:** "
        f"{len(architecture.module_graph):,}"
    )

    report.append("")

    report.append(
        f"**Circular dependency components:** "
        f"{len(architecture.cycles):,}"
    )

    report.append("")

    if architecture.cycles:

        report.append("### Circular Dependency Components")
        report.append("")

        for i, cycle in enumerate(
            architecture.cycles[:50],
            start=1,
        ):

            report.append(
                f"{i}. " + " → ".join(
                    f"`{md(module)}`"
                    for module in cycle
                )
            )

        report.append("")

    # ------------------------------------------------------------------
    # Fan-in
    # ------------------------------------------------------------------

    report.append("### 7.1 Highest Fan-In Modules")
    report.append("")

    report.append(
        "| Rank | Module | Imported By |"
    )

    report.append(
        "|---:|---|---:|"
    )

    for i, (module, count) in enumerate(
        architecture.fan_in.most_common(40),
        start=1,
    ):

        report.append(
            f"| {i} | `{md(module)}` | {count:,} |"
        )

    report.append("")

    # ------------------------------------------------------------------
    # Fan-out
    # ------------------------------------------------------------------

    report.append("### 7.2 Highest Fan-Out Modules")
    report.append("")

    report.append(
        "| Rank | Module | Dependencies |"
    )

    report.append(
        "|---:|---|---:|"
    )

    for i, (module, count) in enumerate(
        sorted(
            architecture.fan_out.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:40],
        start=1,
    ):

        report.append(
            f"| {i} | `{md(module)}` | {count:,} |"
        )

    report.append("")

    # ==================================================================
    # Testing
    # ==================================================================

    report.append("## 8. Testing")
    report.append("")

    report.append("| Metric | Value |")
    report.append("|---|---:|")

    report.append(
        f"| Test source files | {test.files:,} |"
    )

    report.append(
        f"| Test source LOC | {test.code:,} |"
    )

    report.append(
        f"| Test fixture files | "
        f"{analysis.test_fixtures.files:,} |"
    )

    report.append(
        f"| Test fixture size | "
        f"{human_bytes(analysis.test_fixtures.bytes)} |"
    )

    report.append(
        f"| Test / production source LOC | "
        f"{test_ratio * 100:.2f}% |"
    )

    report.append("")

    report.append(
        "This analyzer intentionally separates executable test source "
        "from fixtures. Large JSON fixtures are not interpreted as "
        "hundreds of thousands of lines of test code."
    )

    report.append("")

    # ==================================================================
    # Directory distribution
    # ==================================================================

    report.append("## 9. Source Directory Distribution")
    report.append("")

    report.append(
        "| Directory | Files | Code LOC | Share of Source |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    source_directories = []

    for directory, metrics in analysis.directories.items():

        # Only source/test-source directories.
        source_code = metrics.code

        if source_code <= 0:
            continue

        source_directories.append(
            (directory, metrics)
        )

    source_directories.sort(
        key=lambda item: item[1].code,
        reverse=True,
    )

    for directory, metrics in source_directories[:50]:

        report.append(
            f"| `{md(directory)}` | "
            f"{metrics.files:,} | "
            f"{metrics.code:,} | "
            f"{percentage(metrics.code, source_plus_tests.code)} |"
        )

    report.append("")

    # ==================================================================
    # Data / runtime
    # ==================================================================

    report.append("## 10. Data & Generated Material")
    report.append("")

    report.append("| Category | Files | Size |")
    report.append("|---|---:|---:|")

    for name, metrics in [
        ("Data", analysis.data),
        ("Runtime artifacts", analysis.runtime),
        ("Generated artifacts", analysis.artifacts),
        ("Test fixtures", analysis.test_fixtures),
        ("Binary files", analysis.binaries),
    ]:

        report.append(
            f"| {name} | "
            f"{metrics.files:,} | "
            f"{human_bytes(metrics.bytes)} |"
        )

    report.append("")

    # ==================================================================
    # Documentation
    # ==================================================================

    report.append("## 11. Documentation")
    report.append("")

    report.append("| Metric | Value |")
    report.append("|---|---:|")

    report.append(
        f"| Documentation files | "
        f"{analysis.documentation.files:,} |"
    )

    report.append(
        f"| Documentation LOC | "
        f"{analysis.documentation.code:,} |"
    )

    report.append("")

    # ==================================================================
    # Framework
    # ==================================================================

    report.append("## 12. Detected Tooling / Framework Signals")
    report.append("")

    if analysis.framework_signals:

        for name in sorted(
            analysis.framework_signals
        ):
            report.append(
                f"- **{md(name)}**"
            )

    else:
        report.append(
            "No known framework/tooling signals detected."
        )

    report.append("")

    # ==================================================================
    # Engineering signals
    # ==================================================================

    report.append("## 13. Engineering Signals")
    report.append("")

    report.append(
        "These are signals requiring engineering interpretation; "
        "they are not automatic quality scores."
    )

    report.append("")

    signals_list = []

    # Large modules.
    if len(hotspots.huge_files) > 0:
        signals_list.append(
            (
                "Attention",
                f"{len(hotspots.huge_files)} Python source "
                f"files contain at least 2,000 code LOC.",
            )
        )

    # Complexity.
    if hotspots.high_complexity_functions:
        signals_list.append(
            (
                "Attention",
                f"{len(hotspots.high_complexity_functions)} "
                f"functions have cyclomatic complexity ≥ 10.",
            )
        )

    # Cycles.
    if architecture.cycles:
        signals_list.append(
            (
                "Attention",
                f"{len(architecture.cycles)} circular dependency "
                f"components were detected.",
            )
        )
    else:
        signals_list.append(
            (
                "Positive signal",
                "No module-level circular dependencies were detected "
                "by the static import resolver.",
            )
        )

    # Typing.
    if signals.functions:
        annotation_ratio = safe_ratio(
            signals.annotated_functions,
            signals.functions,
        )

        if annotation_ratio >= 0.80:
            signals_list.append(
                (
                    "Positive signal",
                    f"{annotation_ratio * 100:.1f}% of analyzed Python "
                    f"functions have at least one type annotation or "
                    f"return annotation.",
                )
            )

    # Broad exception handling.
    if signals.broad_except:
        signals_list.append(
            (
                "Attention",
                f"{signals.broad_except} broad "
                f"Exception/BaseException handlers detected.",
            )
        )

    # Bare except.
    if signals.bare_except:
        signals_list.append(
            (
                "Attention",
                f"{signals.bare_except} bare `except:` handlers detected.",
            )
        )

    # TODO.
    if signals.todo:
        signals_list.append(
            (
                "Informational",
                f"{signals.todo:,} TODO markers detected.",
            )
        )

    # Test ratio.
    if production.code:

        ratio_value = test.code / production.code

        if ratio_value >= 0.20:
            signals_list.append(
                (
                    "Positive signal",
                    f"Test source is approximately "
                    f"{ratio_value * 100:.1f}% of production source LOC.",
                )
            )
        else:
            signals_list.append(
                (
                    "Attention",
                    f"Test source is approximately "
                    f"{ratio_value * 100:.1f}% of production source LOC. "
                    f"Review actual coverage before interpreting this "
                    f"ratio.",
                )
            )

    report.append("| Type | Signal |")
    report.append("|---|---|")

    for signal_type, message in signals_list:

        report.append(
            f"| **{signal_type}** | {md(message)} |"
        )

    report.append("")

    # ==================================================================
    # Largest repository files
    # ==================================================================

    report.append("## 14. Largest Repository Files")
    report.append("")

    report.append(
        "These are listed separately from source hotspots because "
        "large repository files may simply be data or generated artifacts."
    )

    report.append("")

    report.append(
        "| Rank | File | Classification | "
        "Language | Size |"
    )

    report.append(
        "|---:|---|---|---|---:|"
    )

    largest_files = sorted(
        analysis.all_files,
        key=lambda r: r.size_bytes,
        reverse=True,
    )

    for i, record in enumerate(
        largest_files[:50],
        start=1,
    ):

        report.append(
            f"| {i} | `{md(record.path)}` | "
            f"{md(record.classification)} | "
            f"{md(record.language)} | "
            f"{human_bytes(record.size_bytes)} |"
        )

    report.append("")

    # ==================================================================
    # Excluded directories
    # ==================================================================

    report.append("## 15. Excluded Directories")
    report.append("")

    if analysis.excluded_dirs:

        report.append("| Directory | Occurrences |")
        report.append("|---|---:|")

        for name, count in (
            analysis.excluded_dirs.most_common()
        ):

            report.append(
                f"| `{md(name)}` | {count:,} |"
            )

    else:

        report.append(
            "No excluded directories encountered."
        )

    report.append("")

    # ==================================================================
    # Parse errors
    # ==================================================================

    parse_errors = [
        pa
        for pa in analysis.python_analysis
        if pa.parse_error
    ]

    report.append("## 16. Analysis Errors")
    report.append("")

    report.append(
        f"Python AST parse errors: **{len(parse_errors):,}**"
    )

    report.append("")

    if parse_errors:

        report.append(
            "| File | Error |"
        )

        report.append(
            "|---|---|"
        )

        for pa in parse_errors[:100]:

            report.append(
                f"| `{md(pa.path)}` | "
                f"`{md(pa.parse_error)}` |"
            )

        report.append("")

    if analysis.errors:

        report.append(
            f"File-system/read errors: **{len(analysis.errors):,}**"
        )

        report.append("")

        for error in analysis.errors[:100]:
            report.append(
                f"- `{md(error)}`"
            )

        report.append("")

    # ==================================================================
    # Methodology
    # ==================================================================

    report.append("## 17. Methodology")
    report.append("")

    report.append(
        "### Source classification"
    )

    report.append("")

    report.append(
        "The analyzer distinguishes executable source from data, "
        "fixtures, runtime artifacts, generated artifacts, "
        "documentation, configuration, and binaries."
    )

    report.append("")

    report.append(
        "### Python analysis"
    )

    report.append("")

    report.append(
        "Python source is parsed with the standard-library `ast` "
        "module. Functions, classes, imports, decorators, type "
        "annotations, async functions, exception handlers, "
        "comprehensions, docstrings, and approximate cyclomatic "
        "complexity are extracted structurally."
    )

    report.append("")

    report.append(
        "### LOC"
    )

    report.append("")

    report.append(
        "LOC is used as a size measurement. JSON, CSV, datasets, "
        "runtime manifests, generated reports, and other repository "
        "artifacts are not treated as production source."
    )

    report.append("")

    report.append(
        "### Architecture"
    )

    report.append("")

    report.append(
        "Python imports are resolved against modules discovered "
        "inside the repository. Fan-in, fan-out, and strongly "
        "connected components are used to identify dependency "
        "hotspots and circular dependencies."
    )

    report.append("")

    # ==================================================================
    # Limitations
    # ==================================================================

    report.append("## 18. Limitations")
    report.append("")

    limitations = [
        "Cyclomatic complexity is an approximation based on Python AST control-flow constructs.",
        "Static import resolution cannot perfectly model dynamic imports, plugin systems, runtime module loading, or generated imports.",
        "Test LOC is a size metric and does not represent test effectiveness or coverage.",
        "Large files are investigation candidates, not automatically architectural defects.",
        "TODO/FIXME/HACK counts are indicators, not quality scores.",
        "JavaScript/TypeScript analysis is currently lexical rather than AST-based.",
        "Runtime behavior, performance, correctness, security, and data quality cannot be established from static analysis alone.",
    ]

    for item in limitations:
        report.append(
            f"- {item}"
        )

    report.append("")

    # ==================================================================
    # Reproducibility
    # ==================================================================

    report.append("## 19. Reproducibility")
    report.append("")

    report.append("```text")
    report.append(
        f"Analyzer version: {ANALYZER_VERSION}"
    )
    report.append(
        f"Python version: {sys.version.split()[0]}"
    )
    report.append(
        f"Root: {analysis.root}"
    )
    report.append(
        f"Generated: {now.isoformat(timespec='seconds')}"
    )
    report.append("```")
    report.append("")

    report.append("---")
    report.append("")

    report.append(
        "*This report provides engineering evidence rather than "
        "an automatic software-quality verdict.*"
    )

    return "\n".join(report)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "AST-driven repository codebase health analyzer."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to current directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("CODEBASE_HEALTH_REPORT.md"),
        help="Markdown output path.",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    args = parse_args()

    root = args.root.expanduser().resolve()

    if not root.exists():
        print(
            f"ERROR: Repository does not exist: {root}",
            file=sys.stderr,
        )
        return 1

    if not root.is_dir():
        print(
            f"ERROR: Repository root is not a directory: {root}",
            file=sys.stderr,
        )
        return 1

    print("=" * 72)
    print("CODEBASE HEALTH ANALYZER")
    print("=" * 72)
    print(f"Version : {ANALYZER_VERSION}")
    print(f"Root    : {root}")
    print()
    print("Scanning repository...")
    print("Python files will be parsed using AST.")
    print()

    analysis = scan_repository(root)

    print(
        f"Files scanned       : {len(analysis.all_files):,}"
    )

    print(
        f"Production files    : {analysis.source.files:,}"
    )

    print(
        f"Production code LOC : {analysis.source.code:,}"
    )

    print(
        f"Test source files   : {analysis.test_source.files:,}"
    )

    print(
        f"Test source LOC     : {analysis.test_source.code:,}"
    )

    print(
        f"Test fixtures       : {analysis.test_fixtures.files:,}"
    )

    print(
        f"Runtime artifacts   : {analysis.runtime.files:,}"
    )

    print(
        f"Data files          : {analysis.data.files:,}"
    )

    print()

    print("Analyzing Python architecture...")

    architecture = analyze_architecture(
        analysis
    )

    print(
        f"Python modules      : "
        f"{len(architecture.module_graph):,}"
    )

    print(
        f"Cycles detected     : "
        f"{len(architecture.cycles):,}"
    )

    print()

    print("Analyzing hotspots...")

    hotspots = analyze_hotspots(
        analysis
    )

    print(
        f"Large Python files  : "
        f"{len(hotspots.large_files):,}"
    )

    print(
        f"Large functions     : "
        f"{len(hotspots.large_functions):,}"
    )

    print(
        f"Complex functions   : "
        f"{len(hotspots.high_complexity_functions):,}"
    )

    print()

    print("Calculating engineering signals...")

    signals = calculate_quality_signals(
        analysis
    )

    report = generate_report(
        analysis,
        architecture,
        hotspots,
        signals,
    )

    output = args.output

    if not output.is_absolute():
        output = root / output

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print()
    print(
        f"Production source : "
        f"{analysis.source.code:,} LOC"
    )
    print(
        f"Test source       : "
        f"{analysis.test_source.code:,} LOC"
    )
    print(
        f"Test fixtures     : "
        f"{analysis.test_fixtures.files:,} files"
    )
    print(
        f"Data              : "
        f"{human_bytes(analysis.data.bytes)}"
    )
    print(
        f"Runtime artifacts : "
        f"{human_bytes(analysis.runtime.bytes)}"
    )
    print(
        f"Python modules    : "
        f"{len(architecture.module_graph):,}"
    )
    print(
        f"Cycles            : "
        f"{len(architecture.cycles):,}"
    )
    print()
    print(
        f"Report:"
    )
    print(
        f"  {output}"
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())