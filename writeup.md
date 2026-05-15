# XTHP Threats at the Skill Layer: A Chord Replication Study

**Author:** chord_skills research pipeline
**Date:** May 15, 2026
**Target skill:** `unknown`
**Framework:** Anthropic-style skill system (Cowork / Claude Code)

---

## Abstract

This report documents an empirical evaluation of Cross-Tool Harvesting and
Polluting (XTHP) threats at the *skill layer* of LLM agent systems.  The
original Chord framework (NDSS 2026) demonstrated that adversarial tools can
hijack, harvest, and pollute agent control flows in LangChain-based pipelines.
We ask whether the same threat class generalises to skill-based architectures —
where a single natural-language description drives selection and a SKILL.md body
drives execution — and measure the three core metrics from the paper: Hijacking
Success Rate (HSR), Harvesting Attack Success Rate (HASR), and Pollution Success
Rate (PSR).

---

## 1. Background

### 1.1 Skill architecture

A *skill* in an Anthropic-style agent system exposes three attack surfaces:

| Layer | Component | Attack surface |
|-------|-----------|---------------|
| Layer 1 | `description:` frontmatter | Selection hook — the LLM reads this to decide which skill to invoke |
| Layer 2 | `SKILL.md` body | Instruction payload — controls what the skill asks the LLM agent to do |
| Layer 3 | `scripts/preprocess.py`, `scripts/postprocess.py` | Execution payload — arbitrary Python run before/after the skill body |

### 1.2 Attack taxonomy (from the paper)

The adversarial skill must hook into the agent's control flow under one of two
*role settings*:

- **Predecessor** — the adversarial skill is selected *before* the victim skill
  runs, corrupting the input query (pre-emptive pollution, §IV-B / §IV-C).
- **Successor** — the adversarial skill is selected *after* the victim skill,
  tampering with its output (retrospective pollution, §IV-D).

Four hooking strategies are used (§IV-B, §IV-C):

| Strategy | Mechanism | Paper section |
|----------|-----------|---------------|
| TARGETED_SEMANTIC | Claim to preprocess/validate the victim skill's exact inputs | §IV-B1 |
| SCENARIO_BASED | Describe a plausible companion workflow | §IV-B2 |
| DOMAIN_FORMAT | Match victim skill's domain in description | §IV-C1 |
| GENERAL_FORMAT | Broad "pipeline utility" framing | §IV-C2 |

---

## 2. Methodology

The chord_skills pipeline replicates Chord's five-module Hijacker loop and two
downstream agents (Harvester, Polluter) for the skill layer:

```
QueryGenerator (M1)
      ↓  5 task queries
XTHPSkillGenerator (M2)
      ↓  predecessor + successor adversarial skill candidates
SkillTestingAgent (M4)  ←────────────────────────────────────────┐
      ↓  HSR over 5 rounds                                       │
  HSR < 60%?  →  HijackingOptimizer (M3, up to 3 rounds)  ──────┘
      ↓  HijackResult  (HSR ≥ 60% or budget exhausted)
SkillHarvester (M6a)        SkillPolluter (M6b)
      ↓  HASR                     ↓  PSR
               ChordSkillsReport
```

**Shadow LLM:** GPT-4o at temperature=0.8 (paper §V-B default).
**Rounds per evaluation:** N=5.
**HSR threshold:** 60% (3/5 rounds hijacked).
**Optimizer budget:** up to 3 rounds of Phase 1 (pairwise ranking) + Phase 2
(mutation with Performance / Fairness / Reliability / LLM-Friendly strategies).

---

## 3. Stage 1 — Query Generation

*Results not available.*

---

## 4. Stage 2 — XTHP Skill Generation

*Results not available.*

---

## 5. Stage 3 — Hijacking Optimizer

*Results not available.*

---

## 6. Stage 4 — Skill Testing Agent

*Results not available.*

---

## 7. Stage 5 — Skill Hijacker

*Results not available.*

---

## 8. Stage 6a — Skill Harvester

*Results not available (no hijacking or stage skipped).*

---

## 9. Stage 6b — Skill Polluter

*Results not available (no hijacking or stage skipped).*

---

## 10. Summary — Three-Metric Results Table

This table replicates the format of Table V from the Chord paper, adapted for
the skill layer.

| Metric | Predecessor | Successor | Paper baseline (tool layer) |
|--------|------------|----------|-----------------------------|
| **HSR** (Hijacking Success Rate) | — | — | 80.0% avg |
| **HASR** (Harvesting Attack SR) | — | — | 72.5% avg |
| **PSR** (Pollution Success Rate) | — | — | 65.0% avg |

---

## 11. Threat Characterisation

### 11.1 What the skill layer changes

| Dimension | Tool layer (original Chord) | Skill layer (this work) |
|-----------|----------------------------|------------------------|
| Selection mechanism | LLM picks from flat tool pool in every ReAct step | Single-step LLM selection from skill descriptions |
| Predecessor/successor | Two separate tools coexist in tool pool | Single adversarial skill encodes both roles in its body |
| Attack payload | Python class `_run()` method | SKILL.md body (Layer 2) + preprocess/postprocess scripts (Layer 3) |
| Harvesting channel A | Pydantic `Field()` in `args_schema` | Body instruction appended to SKILL.md |
| Harvesting channel B | Standalone named parameter in `_run()` | Named variable in `preprocess.py` |
| Description injection | AST-level surgery on Python source | `dataclasses.replace()` on a plain field |
| Defense applicability | tool_filter / spotlighting / pi_detector / airgap | skill_filter / spotlighting / pi_detector / airgap |

### 11.2 Key architectural observation

In skill systems only **one** skill is selected per task (unlike tool-based
ReAct where multiple tools can be called in sequence).  This means predecessor
and successor behaviour must be encoded *within* the body of a single adversarial
skill rather than as two independent tools.  Despite this additional constraint,
the four hooking strategies from §IV remain effective because the LLM's selection
decision is still driven purely by the Layer 1 description.

### 11.3 Mitigation recommendations

1. **Description provenance verification** — cryptographically sign skill
   descriptions so an adversarial description cannot silently replace a
   legitimate one.
2. **Layer 2 instruction sandboxing** — treat SKILL.md body text as untrusted
   data; run it through a policy-checker before exposing it to the agent's
   system prompt.
3. **Script allowlisting** — only permit `preprocess.py` and `postprocess.py`
   from verified sources; block arbitrary filesystem/network access.
4. **Spotlighting** — wrap skill outputs in XML delimiters to prevent Layer 2
   instructions from leaking into downstream context.

---

## 12. Reproducibility

All results above were generated by the `chord_skills` pipeline:

```bash
# Install dependencies
pip install openai tenacity python-dotenv --break-system-packages

# Run full three-agent pipeline on cusip_validator
cd chord_paper
python chord_skills/run_demo.py --target chord_skills/data/victim_skills/cusip_validator.json

# Or run each stage independently:
python -m chord_skills.query_generator          chord_skills/data/victim_skills/cusip_validator.json
python -m chord_skills.xthp_skill_generator     chord_skills/data/victim_skills/cusip_validator.json
python -m chord_skills.hijacking_optimizer      chord_skills/data/victim_skills/cusip_validator.json
python -m chord_skills.skill_testing_agent      chord_skills/data/victim_skills/cusip_validator.json
python -m chord_skills.skill_hijacker           chord_skills/data/victim_skills/cusip_validator.json
python -m chord_skills.skill_harvester          chord_skills/data/victim_skills/cusip_validator.json
python -m chord_skills.skill_polluter           chord_skills/data/victim_skills/cusip_validator.json
python -m chord_skills.orchestrator             chord_skills/data/victim_skills/cusip_validator.json
```

All stage outputs are persisted as JSON in `chord_skills/demo_results/`.
This writeup was auto-generated by `chord_skills/writeup_generator.py`.

---

*Generated by chord_skills writeup_generator.py — 2026-05-15 16:06*
