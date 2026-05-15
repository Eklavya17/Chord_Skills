"""
chord_skills.skill_schema  —  Module 0
=======================================
Shared data structures used across all chord_skills modules.

Classes
-------
  VictimSkillSchema    describes the target skill being attacked
  XTHPSkillRole        PREDECESSOR | SUCCESSOR enum
  XTHPSkillCandidate   the generated adversarial skill artifact (filled later)
  RoleResult           hijacker result for one role (filled later)
  HijackResult         full hijacker output (filled later)

Only VictimSkillSchema is needed by Module 1 (query_generator).
The remaining classes are stubbed here and will be fleshed out as each
module is implemented.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Role enum
# ---------------------------------------------------------------------------

class XTHPSkillRole(str, Enum):
    PREDECESSOR = "predecessor"   # hooks before victim tools — pollutes input
    SUCCESSOR   = "successor"     # hooks after  victim tools — pollutes output


# ---------------------------------------------------------------------------
# VictimSkillSchema
# ---------------------------------------------------------------------------

@dataclass
class VictimSkillSchema:
    """
    Lightweight descriptor for a victim skill.

    Can be loaded from:
      - A SKILL.md directory  →  VictimSkillSchema.from_skill_md(path)
      - A JSON file           →  VictimSkillSchema.from_json(path)

    Fields
    ------
    name        : skill name from SKILL.md frontmatter
    description : Layer 1 description (the selection hook)
    body        : Layer 2 SKILL.md body text (optional, used for richer prompts)
    tools       : whitelist of tool names from frontmatter tools: list
    skill_dir   : filesystem path to the skill directory (optional)
    """
    name:      str
    description: str
    body:      str            = ""
    tools:     list[str]      = field(default_factory=list)
    skill_dir: Optional[Path] = field(default=None, repr=False)

    # ── Factory: from SKILL.md directory ─────────────────────────────────

    @classmethod
    def from_skill_md(cls, skill_dir: Path | str) -> "VictimSkillSchema":
        """Parse a SKILL.md file into a VictimSkillSchema."""
        skill_dir = Path(skill_dir)
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")

        m = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        if not m:
            raise ValueError(f"No YAML frontmatter found in {skill_dir}/SKILL.md")

        fm_raw, body = m.group(1), m.group(2).strip()

        # Simple single-line value extraction
        def _val(key: str) -> str:
            hit = re.search(rf"^{key}:\s*(.+)$", fm_raw, re.MULTILINE)
            return hit.group(1).strip().strip("\"'") if hit else ""

        # Multi-line description (YAML > folded scalar)
        desc_lines: list[str] = []
        in_desc = False
        for line in fm_raw.splitlines():
            if line.startswith("description:"):
                in_desc = True
                rest = line.partition(":")[2].strip().lstrip(">").strip()
                if rest:
                    desc_lines.append(rest)
            elif in_desc and (line.startswith("  ") or line.startswith("\t")):
                desc_lines.append(line.strip())
            elif in_desc:
                in_desc = False
        description = " ".join(desc_lines) if desc_lines else _val("description")

        # tools: list
        tools: list[str] = []
        in_tools = False
        for line in fm_raw.splitlines():
            if line.strip().startswith("tools:"):
                in_tools = True
                continue
            if in_tools:
                s = line.strip()
                if s.startswith("- "):
                    tools.append(s[2:].strip())
                elif s and not s.startswith("#"):
                    in_tools = False

        return cls(
            name=_val("name") or skill_dir.name,
            description=description,
            body=body,
            tools=tools,
            skill_dir=skill_dir,
        )

    # ── Factory: from JSON descriptor ────────────────────────────────────

    @classmethod
    def from_json(cls, json_path: Path | str) -> "VictimSkillSchema":
        """
        Load from a JSON descriptor file.

        Expected format:
        {
          "name":        "cusip-validator",
          "description": "...",
          "body":        "...",    (optional)
          "tools":       ["bash"]
        }
        """
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            description=data["description"],
            body=data.get("body", ""),
            tools=data.get("tools", []),
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def to_prompt_str(self) -> str:
        """Format for inclusion in LLM prompts."""
        lines = [
            f"Name       : {self.name}",
            f"Description: {self.description}",
        ]
        if self.tools:
            lines.append(f"Tools      : {', '.join(self.tools)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stubs — filled in as each module is implemented
# ---------------------------------------------------------------------------

@dataclass
class XTHPSkillCandidate:
    """
    A generated adversarial skill artifact — three layers.

    Layer 1  description     — frontmatter hook (drives selection)
    Layer 2  body            — SKILL.md body (workflow instructions)
    Layer 3  preprocess_src  — scripts/preprocess.py  (predecessor payload)
             postprocess_src — scripts/postprocess.py (successor payload)
    """
    role:            XTHPSkillRole
    skill_name:      str
    description:     str
    body:            str
    preprocess_src:  str = ""
    postprocess_src: str = ""
    target_name:     str = ""
    attack_vector:   str = ""   # TARGETED_SEMANTIC | SCENARIO_BASED | DOMAIN_FORMAT | GENERAL_FORMAT
    attack_hint:     str = ""   # free-form annotation added by harvester/polluter

    def to_skill_md(self) -> str:
        """Render full SKILL.md text (frontmatter + body) ready to write to disk."""
        tools = ["bash"]
        if "youtube_search" in self.body or "youtube" in self.target_name.lower():
            tools.append("youtube_search")
        tools_block = "\n".join(f"  - {t}" for t in tools)
        return (
            f"---\n"
            f"name: {self.skill_name}\n"
            f"description: {self.description}\n"
            f"tools:\n{tools_block}\n"
            f"---\n\n"
            f"{self.body}\n"
        )

    def write_to_dir(self, skills_dir: Path) -> Path:
        """
        Write this candidate to disk as a proper skill directory.

        Creates:
          <skills_dir>/<skill_name>/SKILL.md
          <skills_dir>/<skill_name>/scripts/preprocess.py   (predecessor)
          <skills_dir>/<skill_name>/scripts/postprocess.py  (successor)

        Returns the skill directory path.
        """
        skill_dir   = Path(skills_dir) / self.skill_name
        scripts_dir = skill_dir / "scripts"
        skill_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(exist_ok=True)

        (skill_dir / "SKILL.md").write_text(self.to_skill_md(), encoding="utf-8")

        if self.preprocess_src.strip():
            (scripts_dir / "preprocess.py").write_text(self.preprocess_src, encoding="utf-8")
        if self.postprocess_src.strip():
            (scripts_dir / "postprocess.py").write_text(self.postprocess_src, encoding="utf-8")

        return skill_dir

    def to_dict(self) -> dict:
        return {
            "role":           self.role.value,
            "skill_name":     self.skill_name,
            "description":    self.description,
            "attack_vector":  self.attack_vector,
            "body_preview":   self.body[:300] + ("…" if len(self.body) > 300 else ""),
            "has_preprocess": bool(self.preprocess_src.strip()),
            "has_postprocess": bool(self.postprocess_src.strip()),
        }


@dataclass
class RoleResult:
    """Stub — implemented in Module 5 (skill_hijacker.py)."""
    role:             XTHPSkillRole      = XTHPSkillRole.PREDECESSOR
    final_candidate:  XTHPSkillCandidate = field(default_factory=XTHPSkillCandidate)
    hsr:              float              = 0.0
    hijacked:         bool               = False
    target_outputs:   list[str]          = field(default_factory=list)
    optimizer_rounds: int                = 0


@dataclass
class HijackResult:
    """Stub — implemented in Module 5 (skill_hijacker.py)."""
    target_name: str                   = ""
    queries:     list[str]             = field(default_factory=list)
    predecessor: Optional[RoleResult]  = None
    successor:   Optional[RoleResult]  = None
    elapsed_sec: float                 = 0.0
