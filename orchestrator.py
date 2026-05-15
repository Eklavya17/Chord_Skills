"""
chord_skills.orchestrator  —  Top-Level Orchestrator
=====================================================
Chains the three chord_skills agents into a single automated pipeline:

    SkillHijacker ──(HijackResult)──► SkillHarvester ──(HarvestResult)──► SkillPolluter
                                                                                 │
                                                                     results/report.json
                                                                     results/summary.csv
                                                                     logs/final.log

Paper reference
---------------
§V (Chord System Overview, Figure 2):
  "Chord is a tool to automatically discover the hidden XTHP threats that
   existed in common LLM agent tool collections. The threat model contains
   three interconnected agents: Hijacker, Harvester, and Polluter."

  "The output of Hijacker is used by both Harvester and Polluter."

Metrics aggregated per skill (Table V):
  HSR  — Hijacking Success Rate      (SkillHijacker output)
  HASR — Harvesting Attack SR        (SkillHarvester output)
  PSR  — Polluting Success Rate      (SkillPolluter output)

Skill-layer note
----------------
The scan target is a VictimSkillSchema (loaded from a SKILL.md directory or
a .json descriptor) rather than a LangChain ToolSchema. Everything else
mirrors the original orchestrator structure 1:1.

Usage
-----
    from chord_skills.orchestrator import Orchestrator, ScanConfig
    from chord_skills.skill_testing_agent import DefenseType

    config = ScanConfig(
        skill_paths=["pipeline/skills/cusip_validator"],
        defense=DefenseType.NONE,
    )
    orchestrator = Orchestrator(config)
    report = orchestrator.run()
    print(report.summary_table())

    # One-call shorthand:
    from chord_skills.orchestrator import run_chord_skills
    report = run_chord_skills(
        skill_paths=["pipeline/skills/cusip_validator"],
    )
    print(report.summary_table())

Standalone usage
----------------
  # Scan a single skill directory:
  python -m chord_skills.orchestrator pipeline/skills/cusip_validator

  # Scan a JSON descriptor:
  python -m chord_skills.orchestrator chord_skills/data/victim_skills/cusip_validator.json

  # With defense and phase flags:
  python -m chord_skills.orchestrator <target> [defense] [--skip-harvester] [--skip-polluter]
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chord_skills.skill_schema import VictimSkillSchema                    # noqa: E402
from chord_skills.skill_hijacker import SkillHijacker, HijackResult, run_hijacker  # noqa
from chord_skills.skill_harvester import SkillHarvester, HarvestResult, run_harvester  # noqa
from chord_skills.skill_polluter import (                                   # noqa
    SkillPolluter, PollutionResult, PollutionSpec, run_polluter,
)
from chord_skills.skill_testing_agent import DefenseType, N_ROUNDS          # noqa

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ScanConfig
# ---------------------------------------------------------------------------

@dataclass
class ScanConfig:
    """
    Configuration for a full chord_skills pipeline run.

    Parameters
    ----------
    skill_paths      : List of skill targets — each may be a SKILL.md directory
                       OR a path to a .json VictimSkillSchema descriptor.
    defense          : Defense condition to evaluate (default: NONE / baseline).
    log_dir          : Output directory for per-skill JSON logs.
    results_dir      : Output directory for the final report + CSV.
    model            : LLM model (paper default: gpt-4o).
    temperature      : Sampling temperature (paper default: 0.8).
    pollution_spec   : Custom PollutionSpec; sensible defaults used if None.
    skip_harvester   : Skip harvesting phase (faster; omits HASR from report).
    skip_polluter    : Skip pollution phase (faster; omits PSR from report).
    """
    skill_paths:     list[str]
    defense:         DefenseType             = DefenseType.NONE
    log_dir:         str                     = "chord_skills/logs"
    results_dir:     str                     = "chord_skills/results"
    model:           str                     = "gpt-4o"
    temperature:     float                   = 0.8
    pollution_spec:  Optional[PollutionSpec] = None
    skip_harvester:  bool                    = False
    skip_polluter:   bool                    = False


# ---------------------------------------------------------------------------
# SkillResult  —  per-skill aggregated result (Table V row)
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    """
    Full chord_skills result for one victim skill — mirrors Table V row layout.
    Contains outputs from all three agents, ready for report serialisation.
    """
    target_name:      str
    schema:           VictimSkillSchema
    hijack_result:    Optional[HijackResult]    = None
    harvest_result:   Optional[HarvestResult]   = None
    pollution_result: Optional[PollutionResult] = None
    elapsed_sec:      float                     = 0.0
    error:            Optional[str]             = None

    # ── Metric accessors ─────────────────────────────────────────────

    @property
    def hsr_pred(self) -> float:
        if self.hijack_result and self.hijack_result.predecessor:
            return self.hijack_result.predecessor.hsr
        return 0.0

    @property
    def hsr_succ(self) -> float:
        if self.hijack_result and self.hijack_result.successor:
            return self.hijack_result.successor.hsr
        return 0.0

    @property
    def hasr_pred(self) -> float:
        return self.harvest_result.hasr_predecessor if self.harvest_result else 0.0

    @property
    def hasr_succ(self) -> float:
        return self.harvest_result.hasr_successor if self.harvest_result else 0.0

    @property
    def psr_pred(self) -> float:
        return self.pollution_result.psr_predecessor if self.pollution_result else 0.0

    @property
    def psr_succ(self) -> float:
        return self.pollution_result.psr_successor if self.pollution_result else 0.0

    @property
    def any_hijacked(self) -> bool:
        return self.hijack_result.any_hijacked if self.hijack_result else False

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "elapsed_sec": self.elapsed_sec,
            "error":       self.error,
            "hsr_pred":    round(self.hsr_pred, 3),
            "hsr_succ":    round(self.hsr_succ, 3),
            "hasr_pred":   round(self.hasr_pred, 3),
            "hasr_succ":   round(self.hasr_succ, 3),
            "psr_pred":    round(self.psr_pred, 3),
            "psr_succ":    round(self.psr_succ, 3),
            "hijack":    self.hijack_result.to_dict()    if self.hijack_result    else None,
            "harvest":   self.harvest_result.to_dict()  if self.harvest_result   else None,
            "pollution": self.pollution_result.to_dict() if self.pollution_result else None,
        }

    def to_csv_row(self) -> dict:
        """Single flat row for the summary CSV — mirrors Table V column layout."""
        pred_xthp = succ_xthp = "—"
        if self.hijack_result:
            if self.hijack_result.predecessor:
                pred_xthp = self.hijack_result.predecessor.final_candidate.skill_name
            if self.hijack_result.successor:
                succ_xthp = self.hijack_result.successor.final_candidate.skill_name
        return {
            "target":      self.target_name,
            "pred_xthp":   pred_xthp,
            "succ_xthp":   succ_xthp,
            "HSR_pred":    f"{self.hsr_pred:.1%}",
            "HSR_succ":    f"{self.hsr_succ:.1%}",
            "HASR_pred":   f"{self.hasr_pred:.1%}",
            "HASR_succ":   f"{self.hasr_succ:.1%}",
            "PSR_pred":    f"{self.psr_pred:.1%}",
            "PSR_succ":    f"{self.psr_succ:.1%}",
            "hijacked":    "YES" if self.any_hijacked else "NO",
            "elapsed_sec": f"{self.elapsed_sec:.1f}",
            "error":       self.error or "",
        }


# ---------------------------------------------------------------------------
# ChordSkillsReport
# ---------------------------------------------------------------------------

@dataclass
class ChordSkillsReport:
    """
    Final output of a full chord_skills scan.

    Contains one SkillResult per target, plus aggregate stats and
    serialisation methods for JSON / CSV / plain text.
    """
    skill_results:  list[SkillResult] = field(default_factory=list)
    defense:        DefenseType       = DefenseType.NONE
    total_elapsed:  float             = 0.0

    # ── Aggregates ────────────────────────────────────────────────────

    @property
    def n_skills(self) -> int:
        return len(self.skill_results)

    @property
    def n_hijacked(self) -> int:
        return sum(1 for r in self.skill_results if r.any_hijacked)

    @property
    def avg_hsr(self) -> float:
        vals = [r.hsr_pred for r in self.skill_results] + \
               [r.hsr_succ for r in self.skill_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_hasr(self) -> float:
        vals = [r.hasr_pred for r in self.skill_results] + \
               [r.hasr_succ for r in self.skill_results]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_psr(self) -> float:
        vals = [r.psr_pred for r in self.skill_results] + \
               [r.psr_succ for r in self.skill_results]
        return sum(vals) / len(vals) if vals else 0.0

    # ── Formatting ────────────────────────────────────────────────────

    def summary_table(self) -> str:
        """
        Plain-text summary table replicating Table V layout from the paper.

        Columns: Target | Pred XTHP | Succ XTHP | HSR(P) | HSR(S) |
                 HASR(P) | HASR(S) | PSR(P) | PSR(S)
        """
        col_w   = [24, 28, 28, 8, 8, 8, 8, 8, 8]
        headers = [
            "Target", "Pred XTHP Skill", "Succ XTHP Skill",
            "HSR(P)", "HSR(S)", "HASR(P)", "HASR(S)", "PSR(P)", "PSR(S)",
        ]
        sep = "─" * (sum(col_w) + len(col_w) * 3 + 1)

        def fmt_row(cells: list[str]) -> str:
            return "│ " + " │ ".join(c.ljust(w) for c, w in zip(cells, col_w)) + " │"

        lines = [sep, fmt_row(headers), sep]
        for r in self.skill_results:
            row = r.to_csv_row()
            lines.append(fmt_row([
                row["target"][:col_w[0]],
                row["pred_xthp"][:col_w[1]],
                row["succ_xthp"][:col_w[2]],
                row["HSR_pred"],
                row["HSR_succ"],
                row["HASR_pred"],
                row["HASR_succ"],
                row["PSR_pred"],
                row["PSR_succ"],
            ]))
        lines += [
            sep,
            fmt_row([
                "AVERAGE", "", "",
                f"{self.avg_hsr:.1%}", "",
                f"{self.avg_hasr:.1%}", "",
                f"{self.avg_psr:.1%}", "",
            ]),
            sep,
        ]
        lines.append(
            f"  Skills scanned: {self.n_skills}  |  "
            f"Hijacked: {self.n_hijacked}/{self.n_skills}  |  "
            f"Defense: {self.defense.value}  |  "
            f"Elapsed: {self.total_elapsed:.1f}s"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "defense":       self.defense.value,
            "total_elapsed": self.total_elapsed,
            "n_skills":      self.n_skills,
            "n_hijacked":    self.n_hijacked,
            "avg_hsr":       round(self.avg_hsr, 3),
            "avg_hasr":      round(self.avg_hasr, 3),
            "avg_psr":       round(self.avg_psr, 3),
            "skills":        [r.to_dict() for r in self.skill_results],
        }

    def save(self, results_dir: str) -> tuple[Path, Path]:
        """Write results/report.json and results/summary.csv. Returns (json, csv)."""
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "report.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2))

        csv_path = out / "summary.csv"
        if self.skill_results:
            fieldnames = list(self.skill_results[0].to_csv_row().keys())
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.skill_results:
                    writer.writerow(r.to_csv_row())

        logger.info("[Orchestrator] Report saved → %s", json_path)
        logger.info("[Orchestrator] CSV saved    → %s", csv_path)
        return json_path, csv_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Top-level chord_skills orchestrator (§V, Figure 2 — skill adaptation).

    For each target skill it:
      1. Runs SkillHijacker  → HijackResult  (HSR per role)
      2. Runs SkillHarvester → HarvestResult (HASR per CRD × role)
      3. Runs SkillPolluter  → PollutionResult (PSR per role)
      4. Aggregates into ChordSkillsReport → writes results/ + logs/

    Parameters
    ----------
    config : ScanConfig controlling which skills to scan and with what settings.
    """

    def __init__(self, config: ScanConfig) -> None:
        self.config      = config
        self.log_dir     = Path(config.log_dir)
        self.results_dir = Path(config.results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ChordSkillsReport:
        """
        Execute the full three-agent pipeline for all skills in config.skill_paths.

        Returns
        -------
        ChordSkillsReport aggregating all SkillResult objects.
        """
        t_start = time.time()
        report  = ChordSkillsReport(defense=self.config.defense)

        logger.info("=" * 70)
        logger.info(
            "[Orchestrator] chord_skills scan started  |  skills=%d  |  defense=%s",
            len(self.config.skill_paths), self.config.defense.value,
        )
        logger.info("=" * 70)

        for i, skill_path in enumerate(self.config.skill_paths, 1):
            schema = self._load_schema(skill_path)
            logger.info(
                "[Orchestrator] (%d/%d) Processing: %s",
                i, len(self.config.skill_paths), schema.name,
            )
            skill_result = self._process_one_skill(schema)
            report.skill_results.append(skill_result)
            self._append_final_log(skill_result)

        report.total_elapsed = round(time.time() - t_start, 2)
        json_path, csv_path  = report.save(str(self.results_dir))

        logger.info("=" * 70)
        logger.info("[Orchestrator] Scan complete in %.1fs", report.total_elapsed)
        logger.info(
            "[Orchestrator] %d/%d skills hijacked", report.n_hijacked, report.n_skills,
        )
        logger.info(
            "[Orchestrator] Avg HSR=%.1f%%  HASR=%.1f%%  PSR=%.1f%%",
            report.avg_hsr * 100, report.avg_hasr * 100, report.avg_psr * 100,
        )
        logger.info("=" * 70)

        return report

    # ------------------------------------------------------------------
    # Per-skill pipeline
    # ------------------------------------------------------------------

    def _process_one_skill(self, schema: VictimSkillSchema) -> SkillResult:
        """
        Run the full SkillHijacker → SkillHarvester → SkillPolluter pipeline
        for one victim skill. Exceptions are caught so the scan continues.
        """
        t0     = time.time()
        result = SkillResult(target_name=schema.name, schema=schema)

        try:
            # ── Phase 1: SkillHijacker ──────────────────────────────────
            logger.info("[Orchestrator] Phase 1/3 — SkillHijacker: %s", schema.name)
            hijacker = SkillHijacker(
                victim_schema=schema,
                defense=self.config.defense,
                log_dir=self.config.log_dir,
                model=self.config.model,
                temperature=self.config.temperature,
            )
            hijack_result       = hijacker.run()
            result.hijack_result = hijack_result

            logger.info(
                "[Orchestrator] Hijacker done — pred_HSR=%.1f%%  succ_HSR=%.1f%%",
                hijack_result.predecessor.hsr * 100 if hijack_result.predecessor else 0,
                hijack_result.successor.hsr   * 100 if hijack_result.successor   else 0,
            )

            # ── Phase 2: SkillHarvester ─────────────────────────────────
            if not self.config.skip_harvester and hijack_result.any_hijacked:
                logger.info("[Orchestrator] Phase 2/3 — SkillHarvester: %s", schema.name)
                harvester = SkillHarvester(
                    victim_schema=schema,
                    defense=self.config.defense,
                    log_dir=self.config.log_dir,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
                harvest_result       = harvester.run(hijack_result)
                result.harvest_result = harvest_result
                logger.info(
                    "[Orchestrator] Harvester done — HASR_pred=%.1f%%  HASR_succ=%.1f%%",
                    harvest_result.hasr_predecessor * 100,
                    harvest_result.hasr_successor   * 100,
                )
            elif self.config.skip_harvester:
                logger.info("[Orchestrator] Harvester skipped (skip_harvester=True)")
            else:
                logger.info("[Orchestrator] Harvester skipped — no hijacking succeeded")

            # ── Phase 3: SkillPolluter ──────────────────────────────────
            if not self.config.skip_polluter and hijack_result.any_hijacked:
                logger.info("[Orchestrator] Phase 3/3 — SkillPolluter: %s", schema.name)
                spec = self.config.pollution_spec or PollutionSpec(
                    queries=hijack_result.queries,
                )
                polluter = SkillPolluter(
                    victim_schema=schema,
                    defense=self.config.defense,
                    log_dir=self.config.log_dir,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
                pollution_result       = polluter.run(hijack_result, spec)
                result.pollution_result = pollution_result
                logger.info(
                    "[Orchestrator] Polluter done — PSR_pred=%.1f%%  PSR_succ=%.1f%%",
                    pollution_result.psr_predecessor * 100,
                    pollution_result.psr_successor   * 100,
                )
            elif self.config.skip_polluter:
                logger.info("[Orchestrator] Polluter skipped (skip_polluter=True)")
            else:
                logger.info("[Orchestrator] Polluter skipped — no hijacking succeeded")

        except Exception as exc:
            logger.error(
                "[Orchestrator] Error processing %s: %s",
                schema.name, exc, exc_info=True,
            )
            result.error = str(exc)

        result.elapsed_sec = round(time.time() - t0, 2)
        logger.info(
            "[Orchestrator] %s done in %.1fs  "
            "[HSR_p=%.1f%%  HSR_s=%.1f%%  HASR_p=%.1f%%  PSR_p=%.1f%%]",
            schema.name, result.elapsed_sec,
            result.hsr_pred * 100, result.hsr_succ * 100,
            result.hasr_pred * 100, result.psr_pred * 100,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_schema(skill_path: str) -> VictimSkillSchema:
        """Load VictimSkillSchema from either a SKILL.md directory or a .json file."""
        p = Path(skill_path)
        if p.is_dir():
            return VictimSkillSchema.from_skill_md(p)
        if p.suffix == ".json":
            return VictimSkillSchema.from_json(p)
        raise ValueError(f"[Orchestrator] Unrecognised skill path: {skill_path}")

    def _append_final_log(self, skill_result: SkillResult) -> None:
        """
        Append a row to logs/final.log in the paper's format:
          predecessor, <target>, <xthp_name>, HSR=X/5, HASR=Y/10, PSR=Z/5,
        """
        final_log = self.log_dir / "final.log"
        hr = skill_result.hijack_result

        if hr is None:
            line = (
                f"error, {skill_result.target_name}, -, "
                f"HSR=0/{N_ROUNDS}, HASR=0/10, PSR=0/{N_ROUNDS},\n"
            )
            with Path(final_log).open("a") as f:
                f.write(line)
            return

        for rr in [hr.predecessor, hr.successor]:
            if rr is None:
                continue
            role_label = rr.role.value
            hsr_n      = round(rr.hsr * N_ROUNDS)
            xthp       = rr.final_candidate.skill_name

            if skill_result.harvest_result:
                hv       = skill_result.harvest_result
                hasr_val = (hv.hasr_predecessor
                            if rr.role.value == "predecessor"
                            else hv.hasr_successor)
                hasr_n   = round(hasr_val * 10)
            else:
                hasr_n = "?"

            if skill_result.pollution_result:
                pv      = skill_result.pollution_result
                psr_val = (pv.psr_predecessor
                           if rr.role.value == "predecessor"
                           else pv.psr_successor)
                psr_n   = round(psr_val * N_ROUNDS)
            else:
                psr_n = "?"

            line = (
                f"{role_label}, {skill_result.target_name}, {xthp}, "
                f"HSR={hsr_n}/{N_ROUNDS}, HASR={hasr_n}/10, PSR={psr_n}/{N_ROUNDS},\n"
            )
            with Path(final_log).open("a") as f:
                f.write(line)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def run_chord_skills(
    skill_paths:    list[str],
    defense:        DefenseType             = DefenseType.NONE,
    log_dir:        str                     = "chord_skills/logs",
    results_dir:    str                     = "chord_skills/results",
    model:          str                     = "gpt-4o",
    temperature:    float                   = 0.8,
    pollution_spec: Optional[PollutionSpec] = None,
    skip_harvester: bool                    = False,
    skip_polluter:  bool                    = False,
) -> ChordSkillsReport:
    """
    One-call entry point: configure and run the full chord_skills pipeline.

    Example
    -------
    >>> from chord_skills.orchestrator import run_chord_skills
    >>> report = run_chord_skills(
    ...     skill_paths=["pipeline/skills/cusip_validator"],
    ...     skip_harvester=True,   # faster: HSR + PSR only
    ... )
    >>> print(report.summary_table())
    """
    config = ScanConfig(
        skill_paths=skill_paths,
        defense=defense,
        log_dir=log_dir,
        results_dir=results_dir,
        model=model,
        temperature=temperature,
        pollution_spec=pollution_spec,
        skip_harvester=skip_harvester,
        skip_polluter=skip_polluter,
    )
    return Orchestrator(config).run()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m chord_skills.orchestrator <skill_dir_or_json> [defense]")
        print("         [--skip-harvester] [--skip-polluter]")
        print()
        print("Examples:")
        print("  python -m chord_skills.orchestrator pipeline/skills/cusip_validator")
        print("  python -m chord_skills.orchestrator chord_skills/data/victim_skills/cusip_validator.json none --skip-harvester")
        sys.exit(1)

    args           = sys.argv[1:]
    skill_path     = args[0]
    defense_str    = args[1] if len(args) > 1 and not args[1].startswith("--") else "none"
    skip_harvester = "--skip-harvester" in args
    skip_polluter  = "--skip-polluter"  in args
    defense        = DefenseType(defense_str)

    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  chord_skills Orchestrator")
    print(f"  target={skill_path}  defense={defense.value}")
    print(f"  skip_harvester={skip_harvester}  skip_polluter={skip_polluter}")
    print(sep)

    report = run_chord_skills(
        skill_paths=[skill_path],
        defense=defense,
        skip_harvester=skip_harvester,
        skip_polluter=skip_polluter,
    )

    print()
    print(report.summary_table())
    print()
    print(f"  Results written to chord_skills/results/")
    print(f"  Logs    written to chord_skills/logs/")
