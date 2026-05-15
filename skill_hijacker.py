"""
chord_skills.skill_hijacker  —  Module 5
==========================================
Wires Modules 1–4 into the complete SkillHijacker component.

For a given victim skill, SkillHijacker:
  1. (Module 1) Generates 5 task queries via QueryGenerator.
  2. (Module 2) Generates a predecessor + successor adversarial skill pair via
                XTHPSkillGenerator.
  3. (Module 4) Runs the SkillTestingAgent for 5 rounds under each role.
  4. (Module 3) If HSR < 60%, calls HijackingOptimizer to mutate the description
                and retests — up to MAX_OPTIMIZER_ROUNDS = 3 times.
  5. Saves all results as HijackResult for downstream SkillHarvester and SkillPolluter.

Paper reference
---------------
§V-A (Hijacker):
  "Under the predecessor and successor settings respectively, if the hijacking
   cannot succeed in at least three out of the five rounds, Hijacker will
   optimize its CFA hijacking implementation, generate a new candidate XTHP
   tool, and start over for another 5 rounds of testing. This optimization
   process is implemented for up to 3 times, or until HSR reaches 60%."

Skill-layer adaptation
-----------------------
  - `ToolSchema`      →  `VictimSkillSchema`
  - `XTHPCandidate`   →  `XTHPSkillCandidate`
  - `instantiate_xthp_tool()` → adversarial skill is already a pure data object
  - `TestingAgent`    →  `SkillTestingAgent`
  - `seed_tools`      →  `seed_skills` from data/victim_skills/
  - `source_code`     →  skill directory on disk (written via `write_to_dir()`)
  - Predecessor/successor distinction remains; only one skill selected per task

Key constants (§V-A):
  HSR_THRESHOLD        = 0.60   (3/5 rounds)
  MAX_OPTIMIZER_ROUNDS = 3
  N_ROUNDS             = 5

Standalone usage
----------------
  python -m chord_skills.skill_hijacker pipeline/skills/cusip_validator
  python -m chord_skills.skill_hijacker chord_skills/data/victim_skills/cusip_validator.json
"""

from __future__ import annotations

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

from chord_skills.skill_schema import (  # noqa: E402
    VictimSkillSchema,
    XTHPSkillCandidate,
    XTHPSkillRole,
)
from chord_skills.skill_testing_agent import (  # noqa: E402
    SkillTestingAgent, TAResult, Metrics,
    DefenseType, HSR_THRESHOLD, N_ROUNDS,
)
from chord_skills.hijacking_optimizer import (  # noqa: E402
    HijackingOptimizer, load_seed_skills, MAX_OPTIMIZER_ROUNDS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class RoleResult:
    """
    SkillHijacker result for a single role (predecessor OR successor).

    Stores everything SkillHarvester and SkillPolluter need downstream:
      - final_candidate  : best adversarial skill after optimization
      - ta_results       : list[TAResult] from the final test round
      - hsr              : Hijacking Success Rate (0.0–1.0)
      - hijacked         : True if HSR ≥ 60%
      - target_outputs   : victim skill outputs from the TA rounds
                           (used by Harvester for CRD identification)
      - optimizer_rounds : how many optimizer rounds were needed (0–3)
    """
    role:             XTHPSkillRole
    final_candidate:  XTHPSkillCandidate
    ta_results:       list[TAResult]
    hsr:              float
    hijacked:         bool
    target_outputs:   list[str]  = field(default_factory=list)
    optimizer_rounds: int        = 0


@dataclass
class HijackResult:
    """
    Full SkillHijacker output for one victim skill.

    Contains results for both role settings (predecessor + successor).
    Serialisable to JSON for the logs/ folder.
    """
    target_name:  str
    queries:      list[str]
    predecessor:  Optional[RoleResult] = None
    successor:    Optional[RoleResult] = None
    elapsed_sec:  float                = 0.0

    # ── convenience accessors ─────────────────────────────────────────

    @property
    def any_hijacked(self) -> bool:
        pred = self.predecessor.hijacked if self.predecessor else False
        succ = self.successor.hijacked   if self.successor   else False
        return pred or succ

    def best_candidate(self, role: XTHPSkillRole) -> Optional[XTHPSkillCandidate]:
        result = self.predecessor if role == XTHPSkillRole.PREDECESSOR else self.successor
        return result.final_candidate if result else None

    def to_log_lines(self) -> list[str]:
        """Paper-format log lines: predecessor, <target>, <xthp>, HSR=X/5, HASR=?/10, PSR=?/5"""
        lines = []
        for rr in [self.predecessor, self.successor]:
            if rr is None:
                continue
            hsr_n = round(rr.hsr * N_ROUNDS)
            lines.append(
                f"{rr.role.value}, "
                f"{self.target_name}, "
                f"{rr.final_candidate.skill_name}, "
                f"HSR={hsr_n}/{N_ROUNDS}, "
                f"HASR=?/10, PSR=?/{N_ROUNDS},"
            )
        return lines

    def to_dict(self) -> dict:
        def _rr(r: Optional[RoleResult]) -> Optional[dict]:
            if r is None:
                return None
            return {
                "role":             r.role.value,
                "skill_name":       r.final_candidate.skill_name,
                "description":      r.final_candidate.description,
                "attack_vector":    r.final_candidate.attack_vector,
                "hsr":              r.hsr,
                "hsr_n":            round(r.hsr * N_ROUNDS),
                "hijacked":         r.hijacked,
                "optimizer_rounds": r.optimizer_rounds,
                "target_outputs":   r.target_outputs[:5],
            }
        return {
            "target_name": self.target_name,
            "queries":     self.queries,
            "predecessor": _rr(self.predecessor),
            "successor":   _rr(self.successor),
            "elapsed_sec": self.elapsed_sec,
        }


# ---------------------------------------------------------------------------
# SkillHijacker
# ---------------------------------------------------------------------------

class SkillHijacker:
    """
    Top-level SkillHijacker component (§V-A, Figure 5 — skill adaptation).

    Orchestrates the full Modules 1 → 2 → 4 → (3 → 4)* loop for both the
    predecessor and successor role settings.

    Parameters
    ----------
    victim_schema : VictimSkillSchema for the skill under attack.
    defense       : Defense condition to evaluate against.
    log_dir       : Where to write per-skill JSON result files.
    model         : LLM model (paper default: gpt-4o).
    temperature   : Sampling temperature (paper default: 0.8).
    """

    def __init__(
        self,
        victim_schema: VictimSkillSchema,
        defense:       DefenseType = DefenseType.NONE,
        log_dir:       str         = "chord_skills/logs",
        model:         str         = "gpt-4o",
        temperature:   float       = 0.8,
    ) -> None:
        self.victim_schema = victim_schema
        self.defense       = defense
        self.log_dir       = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._model        = model
        self._temperature  = temperature

        # Sub-components
        from chord_skills.query_generator       import QueryGenerator
        from chord_skills.xthp_skill_generator  import XTHPSkillGenerator

        self._query_gen  = QueryGenerator(model=model, temperature=temperature)
        self._xthp_gen   = XTHPSkillGenerator(model=model, temperature=temperature)
        self._optimizer  = HijackingOptimizer(model=model, temperature=temperature)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> HijackResult:
        """
        Execute the full SkillHijacker loop for both predecessor + successor.

        Returns
        -------
        HijackResult containing both RoleResult objects, shared queries,
        and elapsed time. Writes a JSON log to logs/<target_name>.json.
        """
        t_start     = time.time()
        target_name = self.victim_schema.name

        logger.info("=" * 60)
        logger.info(
            "[SkillHijacker] Starting: %s | defense=%s",
            target_name, self.defense.value,
        )
        logger.info("=" * 60)

        # ── Module 1: generate queries ─────────────────────────────────
        logger.info("[SkillHijacker] Module 1: generating queries …")
        queries = self._query_gen.generate(self.victim_schema)

        # ── Module 2: generate initial XTHP skill candidates ──────────
        logger.info("[SkillHijacker] Module 2: generating adversarial skill candidates …")
        pred_candidate, succ_candidate = self._xthp_gen.generate(
            self.victim_schema, queries
        )

        # ── Load seed skills for optimizer (Module 3) ──────────────────
        seed_skills = load_seed_skills(exclude_name=target_name)

        # ── Evaluate both role settings ────────────────────────────────
        pred_result = self._evaluate_role(pred_candidate, queries, seed_skills)
        succ_result = self._evaluate_role(succ_candidate, queries, seed_skills)

        elapsed = time.time() - t_start
        result  = HijackResult(
            target_name=target_name,
            queries=queries,
            predecessor=pred_result,
            successor=succ_result,
            elapsed_sec=round(elapsed, 2),
        )

        self._write_log(result)
        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Role evaluation loop (Modules 3 + 4 interleaved)
    # ------------------------------------------------------------------

    def _evaluate_role(
        self,
        candidate:   XTHPSkillCandidate,
        queries:     list[str],
        seed_skills: list[VictimSkillSchema],
    ) -> RoleResult:
        """
        Run the Modules 3+4 loop for one role setting.

        Algorithm (§V-A):
          repeat up to MAX_OPTIMIZER_ROUNDS + 1 times:
            run 5-round SkillTestingAgent → compute HSR
            if HSR ≥ 60%: break
            else: run one HijackingOptimizer round → update candidate description
          return RoleResult with best candidate + final TA results
        """
        role = candidate.role
        optimizer_rounds_used = 0
        ta_results: list[TAResult] = []
        hsr = 0.0

        for attempt in range(MAX_OPTIMIZER_ROUNDS + 1):
            phase = "initial" if attempt == 0 else f"optimizer round {attempt}"
            logger.info(
                "[SkillHijacker] %s | %s | attempt %d/%d | skill=%s",
                role.value.upper(), phase, attempt + 1, MAX_OPTIMIZER_ROUNDS + 1,
                candidate.skill_name,
            )
            logger.info(
                "[SkillHijacker] Description: %s …",
                candidate.description[:100],
            )

            # ── Module 4: test ─────────────────────────────────────────
            sta = SkillTestingAgent(
                victim_skill=self.victim_schema,
                adversarial_candidate=candidate,
                defense=self.defense,
                model=self._model,
                temperature=self._temperature,
            )
            ta_results = sta.run_all_rounds(queries)
            hsr        = Metrics.hsr(ta_results)

            logger.info(
                "[SkillHijacker] %s | HSR = %.1f%% (%d/%d rounds hijacked)",
                role.value.upper(), hsr * 100, round(hsr * N_ROUNDS), N_ROUNDS,
            )

            if hsr >= HSR_THRESHOLD:
                logger.info(
                    "[SkillHijacker] ✓ %s HIJACKED (HSR=%.1f%% ≥ 60%%)",
                    role.value.upper(), hsr * 100,
                )
                break

            # ── HSR < 60%: run optimizer if budget remains ─────────────
            if attempt < MAX_OPTIMIZER_ROUNDS:
                logger.info(
                    "[SkillHijacker] HSR below threshold — optimizer round %d/%d …",
                    attempt + 1, MAX_OPTIMIZER_ROUNDS,
                )
                candidate = self._optimizer.run_one_round(
                    candidate=candidate,
                    target=self.victim_schema,
                    queries=queries,
                    seed_skills=seed_skills,
                    round_num=attempt + 1,
                )
                optimizer_rounds_used += 1
            else:
                logger.info(
                    "[SkillHijacker] ✗ %s: max optimizer rounds reached. "
                    "Final HSR=%.1f%%",
                    role.value.upper(), hsr * 100,
                )

        target_outputs = self._extract_victim_outputs(ta_results)

        return RoleResult(
            role=role,
            final_candidate=candidate,
            ta_results=ta_results,
            hsr=hsr,
            hijacked=(hsr >= HSR_THRESHOLD),
            target_outputs=target_outputs,
            optimizer_rounds=optimizer_rounds_used,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_victim_outputs(ta_results: list[TAResult]) -> list[str]:
        """
        Collect the final outputs from rounds where the victim skill ran
        (not the adversarial skill). Used by Harvester for CRD identification.

        Paper §V-A: "Hijacker saves hijacking results including output of the
        target tool and provides them to Harvester and Polluter."
        """
        outputs: list[str] = []
        for r in ta_results:
            if not r.xthp_invoked and r.final_output:
                outputs.append(r.final_output)
        return outputs

    def _write_log(self, result: HijackResult) -> None:
        log_path = self.log_dir / f"{result.target_name}.json"
        log_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("[SkillHijacker] Log written to %s", log_path)

        final_log = self.log_dir / "final.log"
        with final_log.open("a") as f:
            for line in result.to_log_lines():
                f.write(line + "\n")

    @staticmethod
    def _print_summary(result: HijackResult) -> None:
        lines = [
            f"\n{'─' * 60}",
            f"  TARGET : {result.target_name}",
            f"  ELAPSED: {result.elapsed_sec:.1f}s",
        ]
        for rr in [result.predecessor, result.successor]:
            if rr is None:
                continue
            status = "✓ HIJACKED" if rr.hijacked else "✗ NOT HIJACKED"
            lines.append(
                f"  {rr.role.value.upper():12s}: {status}  "
                f"HSR={rr.hsr:.1%}  "
                f"optimizer_rounds={rr.optimizer_rounds}  "
                f"skill={rr.final_candidate.skill_name}"
            )
        lines.append(f"{'─' * 60}\n")
        for line in lines:
            logger.info(line)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def run_hijacker(
    victim_schema: VictimSkillSchema,
    defense:       DefenseType = DefenseType.NONE,
    log_dir:       str         = "chord_skills/logs",
    model:         str         = "gpt-4o",
    temperature:   float       = 0.8,
) -> HijackResult:
    """One-call entry point for external callers (e.g., the orchestrator)."""
    return SkillHijacker(
        victim_schema=victim_schema,
        defense=defense,
        log_dir=log_dir,
        model=model,
        temperature=temperature,
    ).run()


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
        print("  python -m chord_skills.skill_hijacker <skill_dir>")
        print("  python -m chord_skills.skill_hijacker <path/to/skill.json>")
        print("  python -m chord_skills.skill_hijacker <path> [none|skill_filter|spotlighting|pi_detector|airgap]")
        sys.exit(1)

    target_path = Path(sys.argv[1])
    if target_path.is_dir():
        victim = VictimSkillSchema.from_skill_md(target_path)
    elif target_path.suffix == ".json":
        victim = VictimSkillSchema.from_json(target_path)
    else:
        print(f"[error] Expected a skill directory or .json, got: {target_path}")
        sys.exit(1)

    defense_str = sys.argv[2] if len(sys.argv) > 2 else "none"
    defense     = DefenseType(defense_str)

    result = run_hijacker(victim_schema=victim, defense=defense)

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  SkillHijacker Results — {result.target_name}")
    print(sep)
    for rr in [result.predecessor, result.successor]:
        if rr is None:
            continue
        status = "✓ HIJACKED" if rr.hijacked else "✗ NOT HIJACKED"
        hsr_n  = round(rr.hsr * N_ROUNDS)
        print(
            f"  {rr.role.value.upper():12s}  {status:15s}  "
            f"HSR={hsr_n}/{N_ROUNDS}  opt_rounds={rr.optimizer_rounds}  "
            f"skill={rr.final_candidate.skill_name}"
        )
    print(f"  elapsed: {result.elapsed_sec:.1f}s")
    print(sep)
