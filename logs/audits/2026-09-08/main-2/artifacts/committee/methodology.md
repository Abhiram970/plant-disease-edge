## Methodology Transparency Review (SRQR-aware)

### MUST-FIX (submission blockers)
- No methodology blocker was surfaced by the fallback pass.

### SHOULD-FIX (quality improvements)
- (abstract) "First, frozen encoders from 11.4 to 86.3 million parameters diagnose unseen crops at 2.2–13.9 times chance across nested held-out sets of 16, 34 and 51 classes, accuracy nearly flat across that range." — Multiple sections contain numeric claims. Confirm that the same quantities reconcile across main text, tables, and appendix material.
- (method) "SigLIP 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features." — Comparative evaluation language was detected. Deep review should verify that baseline tuning, data splits, and reporting conventions are described symmetrically.

### SRQR Checklist Deltas
- Sampling rationale: clarify how the evidence base supports the paper's strongest claims.
- Data collection details (time/place/duration): add context when results depend on specific settings.
- Coding process (stages, coders, disagreement resolution): specify if qualitative or hybrid analysis is used.
- Saturation: state whether the evidence scope is exhaustive or bounded.
- Triangulation: explain whether multiple evidence sources were reconciled.
- Reflexivity: acknowledge researcher choices that shape interpretation.
