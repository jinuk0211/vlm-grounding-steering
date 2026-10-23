# Draft validation — 2026-10-23

## Document

- English anonymous review draft for a prospective CVPR 2027 submission.
- Official CVPR 2026 style used provisionally; no 2027-specific style release
  was confirmed. See TEMPLATE_PROVENANCE.md.
- Final PDF: 8 letter-sized pages, with 7 main-text pages and references
  beginning on page 8.
- 18 bibliography entries; no unresolved citation or cross-reference markers.
- One vector pipeline diagram, four tables, and three displayed equations.
- Experimental table cells remain unmeasured placeholders.

## Verification

- Independent read-only review checked the paper against the implemented
  patch score, head indexing, bounded gain, KL direction, token positions,
  inference inputs, and the implementation/planned-experiment boundary.
- The review's evaluation-table issue was fixed: the main table explicitly
  contains forced-choice results and includes achieved TPR beside FPR.
  Generated-answer and caption metrics are separate.
- Tectonic 0.17.0 completed successfully with BibTeX after the three-model
  protocol and transfer table were added.
- Final compiler log contains no overfull boxes, undefined references,
  undefined citations, or undefined font-shape warnings.
- Times-compatible Nimbus Roman text fonts are embedded; the original
  conference geometry, font size, and review line-numbering are preserved.
- The new transfer table and adjacent result layouts were visually checked
  for clipping; the previous diagram and equation review remains unchanged.

## Remaining environment diagnostics

The native Codex preview compiler could not launch because Windows returned
access denied (error 5). The source project remains editable, and a separate
verified official portable compiler produced the PDF.

The fallback environment prints a Fontconfig configuration diagnostic and
an invalid-UTF-8 warning in the external lineno.sty package. Compilation
succeeds; the PDF has embedded fonts and no visible replacement glyphs.
These package/environment diagnostics were not hidden by modifying the
conference style.

## Scientific status

This validation concerns writing, software correspondence, references, and
document rendering. It supplies no pretrained-model benchmark results,
newness guarantee, or evidence that EPHS outperforms any baseline.
The reported 74 CPU tests are implementation checks, not
scientific measurements performed while writing this paper.
