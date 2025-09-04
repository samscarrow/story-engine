"""
Shared domain models for Story Engine.
Centralizes common dataclasses used across pipelines and engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StoryRequest:
    title: str
    premise: str
    genre: str
    tone: str
    characters: List[Dict[str, Any]]
    setting: str
    structure: str = "three_act"


@dataclass
class NarrativeArc:
    title: str
    beats: List[Dict[str, Any]]


@dataclass
class SceneDescriptor:
    id: int
    name: str
    situation: str
    location: str
    characters: List[str]
    tension: float
    goals: Dict[str, str]
    emphasis: Dict[str, str]
    sensory: Dict[str, str] = field(default_factory=dict)

