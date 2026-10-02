# Constitution

## Purpose

This project builds and trains ML surrogate models of particle-shape contact
interactions for DEM. Each interaction is modular: code, data, and outputs
never mix across interactions.

The end goal is a new LIGGGHTS particle-shape type whose overlap is computed
by a trained surrogate, evaluated from C++ at runtime. This repo delivers
the trained artifacts and their interface contract; the LIGGGHTS integration
is a separate project.

## Durable principles

1. **Scientific accuracy.** Training labels come from the true reference
   solution, never from a geometric approximation.
2. **Reproducibility and traceable provenance.** Generators use fixed,
   recorded random seeds. Each dataset records how it was generated: script
   version, seed, sampling strategy, and ranges.
3. **Distinct epistemic status.** Verified facts, hypotheses, and proposed
   designs are kept distinct. Statements are labelled as *Confirmed*,
   *Assumption*, *Proposal*, or *Open question*.
4. **Ask before deciding.** Consequential unresolved decisions are raised
   with the user, not guessed.
5. **Preserve existing work.** No deletion, overwrite, move, or rename of
   existing files without approval.
6. **Modular interactions.** Each interaction lives in its own folder under
   `interactions/`. Code, data, and outputs are never mixed across
   interactions.
7. **Documentation per interaction.** Every interaction documents: geometry
   and size convention, coordinate frames, units, quaternion order and
   canonicalization, overlap definition and sign, normalization and
   rescaling, scaler format, and the provenance of the reference solution.
8. **Evaluation.** Evaluation uses explicitly documented test cases, not
   training performance alone. A successful run or build is not scientific
   validation. Record commands, inputs, outputs, and observations.
9. **Cost control.** No full-scale data generation, training run, commit, or
   push without the user's request or approval. Small smoke tests are
   allowed once a phase is approved, reported as smoke tests, not results.
10. **Configuration-specific models.** Each model is specific to one fixed
    shape and one fixed plane. No generalization across shapes or walls.
11. **Sign convention.** The surrogate predicts the signed distance between
    shape and wall, defined through the shape's support function (for a
    polyhedron and a plane: the height of the lowest vertex above the
    plane). Positive = gap (no contact), negative = penetration. LIGGGHTS
    `deltan` is negative for actual overlap, so `deltan = signed_distance`.
12. **Accuracy first.** Overlap accuracy takes priority over extra outputs
    or convenience. Accuracy targets are stated as explicit, measurable
    tolerances in the specs. Contact point and normal are a later feature.
13. **Deployable without Python.** Every artifact and preprocessing step
    must be reproducible in C++ at runtime (portable formats, documented
    transforms, no Python dependency).
14. **Explicit size convention.** Models are trained at a canonical size.
    Rescaling to real particles uses a documented, explicit size definition
    (e.g. edge length), never an implicit one.
