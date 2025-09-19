"""Environment configuration support for storyctl.

This module loads declarative environment definitions from YAML files and
provides helpers to resolve them into concrete environment variables as well
as pre-flight checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import shlex

import os

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - yaml is a required runtime dep
    raise RuntimeError("PyYAML is required to load storyctl environments") from exc


@dataclass
class EnvVarSpec:
    """Declarative description of an environment variable."""

    key: str
    value: Optional[str] = None
    from_env: Optional[str] = None
    required: Optional[bool] = None
    description: Optional[str] = None
    sensitive: Optional[bool] = None

    def copy(self) -> "EnvVarSpec":
        return replace(self)

    def merge(self, other: "EnvVarSpec") -> "EnvVarSpec":
        """Return a new spec with ``other`` overriding unset fields."""

        return EnvVarSpec(
            key=self.key,
            value=other.value if other.value is not None else self.value,
            from_env=other.from_env if other.from_env is not None else self.from_env,
            required=self.required if other.required is None else other.required,
            description=other.description if other.description is not None else self.description,
            sensitive=self.sensitive if other.sensitive is None else other.sensitive,
        )

    @property
    def is_required(self) -> bool:
        return bool(self.required)

    @property
    def is_sensitive(self) -> bool:
        if self.sensitive is not None:
            return self.sensitive
        return bool(self.from_env)


@dataclass
class PreflightCheck:
    """A single environment validation step."""

    name: str
    check_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    optional: bool = False
    description: Optional[str] = None

    def copy(self) -> "PreflightCheck":
        return PreflightCheck(
            name=self.name,
            check_type=self.check_type,
            params=dict(self.params),
            optional=self.optional,
            description=self.description,
        )


@dataclass
class EnvVarStatus:
    """Resolved state of a single environment variable."""

    key: str
    value: Optional[str]
    source: Optional[str]
    required: bool
    sensitive: bool
    missing: bool
    description: Optional[str]
    from_env: Optional[str]


@dataclass
class EnvironmentDefinition:
    """Fully merged environment definition."""

    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    extends: Optional[str] = None
    env_vars: Dict[str, EnvVarSpec] = field(default_factory=dict)
    checks: List[PreflightCheck] = field(default_factory=list)

    def copy(self) -> "EnvironmentDefinition":
        return EnvironmentDefinition(
            name=self.name,
            description=self.description,
            tags=list(self.tags),
            extends=self.extends,
            env_vars={k: v.copy() for k, v in self.env_vars.items()},
            checks=[c.copy() for c in self.checks],
        )

    def resolve(
        self,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "EnvironmentResolution":
        """Resolve declared variables into concrete values.

        Args:
            environ: Optional mapping to use when reading ``from_env`` entries.
                Defaults to :data:`os.environ`.

        Returns:
            EnvironmentResolution with concrete values and missing metadata.
        """

        if environ is None:
            environ = os.environ
        values: Dict[str, str] = {}
        statuses: Dict[str, EnvVarStatus] = {}
        missing_required: List[str] = []
        missing_optional: List[str] = []

        for key, spec in self.env_vars.items():
            resolved_value: Optional[str] = None
            source: Optional[str] = None
            if spec.from_env:
                env_key = spec.from_env
                candidate = environ.get(env_key)
                if candidate is not None:
                    resolved_value = str(candidate)
                    source = f"env:{env_key}"
                elif spec.value is not None:
                    resolved_value = str(spec.value)
                    source = "default"
                else:
                    if spec.is_required:
                        missing_required.append(env_key)
                    else:
                        missing_optional.append(env_key)
            elif spec.value is not None:
                resolved_value = str(spec.value)
                source = "default"

            statuses[key] = EnvVarStatus(
                key=key,
                value=resolved_value,
                source=source,
                required=spec.is_required,
                sensitive=spec.is_sensitive,
                missing=spec.is_required and resolved_value is None,
                description=spec.description,
                from_env=spec.from_env,
            )

            if resolved_value is not None:
                values[key] = resolved_value

        # Second pass: interpolate references in values (e.g., ${LM_ENDPOINT})
        interpolation_context: Dict[str, str] = dict(os.environ)
        interpolation_context.update(values)

        for key, val in list(values.items()):
            interpolated = _interpolate(val, interpolation_context)
            if interpolated != val:
                values[key] = interpolated
            current_status = statuses[key]
            statuses[key] = EnvVarStatus(
                key=current_status.key,
                value=values[key],
                source=current_status.source,
                required=current_status.required,
                sensitive=current_status.sensitive,
                missing=current_status.missing,
                description=current_status.description,
                from_env=current_status.from_env,
            )

        return EnvironmentResolution(
            definition=self,
            values=values,
            statuses=statuses,
            missing_required=missing_required,
            missing_optional=missing_optional,
        )


@dataclass
class EnvironmentResolution:
    """Resolved environment alongside metadata used by the CLI."""

    definition: EnvironmentDefinition
    values: Dict[str, str]
    statuses: Dict[str, EnvVarStatus]
    missing_required: List[str]
    missing_optional: List[str]

    def as_shell(self) -> List[str]:
        """Render resolved variables as ``export`` statements."""

        exports: List[str] = []
        for key, value in self.values.items():
            exports.append(f"export {key}={shlex.quote(value)}")
        return exports

    def as_dict(self) -> Dict[str, str]:
        return dict(self.values)

    def has_missing_required(self) -> bool:
        return bool(self.missing_required)


class EnvironmentRegistry:
    """Loader for environment definitions."""

    def __init__(self, paths: Optional[Sequence[Path | str]] = None) -> None:
        self._raw: Dict[str, Dict[str, Any]] = {}
        self._cache: Dict[str, EnvironmentDefinition] = {}
        self._default: Optional[str] = None
        paths = list(paths or [])
        if not paths:
            paths.append(_default_env_dir())

        for entry in paths:
            self._ingest_path(Path(entry))

        if not self._raw:
            raise RuntimeError("No environments found for storyctl")

        if self._default is None:
            self._default = self._raw[next(iter(self._raw))]["name"]

    @property
    def default(self) -> str:
        assert self._default is not None
        return self._default

    def list(self) -> List[EnvironmentDefinition]:
        return [self.get(name) for name in sorted(self._raw.keys())]

    def get(self, name: Optional[str] = None) -> EnvironmentDefinition:
        target = name or self.default
        if target not in self._raw:
            raise KeyError(f"Unknown environment '{target}'")
        return self._build_environment(target)

    # Internal helpers -------------------------------------------------

    def _ingest_path(self, path: Path) -> None:
        if path.is_file():
            self._load_file(path)
            return
        if path.is_dir():
            for file_path in sorted(path.glob("*.yml")):
                self._load_file(file_path)
            for file_path in sorted(path.glob("*.yaml")):
                self._load_file(file_path)
            return
        raise FileNotFoundError(f"Environment path not found: {path}")

    def _load_file(self, path: Path) -> None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Environment file {path} must contain a mapping")

        name = data.get("name") or path.stem
        data["name"] = name
        self._raw[name] = data
        if data.get("default"):
            self._default = name

    def _build_environment(self, name: str, stack: Optional[List[str]] = None) -> EnvironmentDefinition:
        if name in self._cache:
            return self._cache[name].copy()

        if name not in self._raw:
            raise KeyError(f"Environment '{name}' not defined")

        stack = list(stack or [])
        if name in stack:
            raise RuntimeError(f"Cyclic environment inheritance detected: {' -> '.join(stack + [name])}")
        stack.append(name)

        raw = self._raw[name]
        parent_name = raw.get("extends")
        parent_def: Optional[EnvironmentDefinition]
        if parent_name:
            parent_def = self._build_environment(parent_name, stack)
        else:
            parent_def = None

        env_def = self._compose_environment(raw, parent_def)
        self._cache[name] = env_def
        stack.pop()
        return env_def.copy()

    def _compose_environment(
        self,
        raw: Dict[str, Any],
        parent: Optional[EnvironmentDefinition],
    ) -> EnvironmentDefinition:
        if parent is not None:
            base = parent.copy()
        else:
            base = EnvironmentDefinition(name=raw.get("name", ""))

        base.name = raw.get("name", base.name)
        if raw.get("description"):
            base.description = str(raw["description"])
        if raw.get("tags"):
            for tag in raw["tags"]:
                if tag not in base.tags:
                    base.tags.append(str(tag))
        base.extends = raw.get("extends")

        env_section = raw.get("env") or {}
        parsed_env = _parse_env_section(env_section)
        for key, spec in parsed_env.items():
            if key in base.env_vars:
                base.env_vars[key] = base.env_vars[key].merge(spec)
            else:
                base.env_vars[key] = spec

        checks_section = raw.get("checks") or []
        parsed_checks = _parse_checks_section(checks_section)
        if parsed_checks:
            existing = {chk.name: idx for idx, chk in enumerate(base.checks)}
            for chk in parsed_checks:
                if chk.name in existing:
                    base.checks[existing[chk.name]] = chk
                else:
                    base.checks.append(chk)

        return base


def _parse_env_section(data: Mapping[str, Any]) -> Dict[str, EnvVarSpec]:
    result: Dict[str, EnvVarSpec] = {}
    for key, raw in data.items():
        if isinstance(raw, dict):
            spec = EnvVarSpec(
                key=key,
                value=str(raw.get("value")) if raw.get("value") is not None else None,
                from_env=str(raw.get("from_env")) if raw.get("from_env") is not None else None,
                required=raw.get("required"),
                description=str(raw.get("description")) if raw.get("description") is not None else None,
                sensitive=raw.get("sensitive"),
            )
        else:
            spec = EnvVarSpec(key=key, value=str(raw) if raw is not None else None)
        result[key] = spec
    return result


def _parse_checks_section(data: Iterable[Any]) -> List[PreflightCheck]:
    checks: List[PreflightCheck] = []
    for item in data:
        if not isinstance(item, Mapping):
            raise ValueError("Each check must be a mapping")
        name = str(item.get("name") or item.get("type") or "check")
        check_type = str(item.get("type") or "command")
        params = dict(item.get("params") or {})
        optional = bool(item.get("optional", False))
        description = item.get("description")
        checks.append(
            PreflightCheck(
                name=name,
                check_type=check_type,
                params=params,
                optional=optional,
                description=str(description) if description is not None else None,
            )
        )
    return checks


def _interpolate(value: str, context: Mapping[str, str]) -> str:
    if not isinstance(value, str) or "${" not in value:
        return value
    try:
        return Template(value).safe_substitute(context)
    except Exception:
        return value


def _default_env_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "environments"
