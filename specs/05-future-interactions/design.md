# Spec 05 — Design

## Template approach

The cube-wall interaction serves as the reference template. When adding a new interaction:

1. Copy the scaffold structure.
2. Adapt the generator to the new shape's support function.
3. Adapt the training script if the input/output contract changes.
4. Write new specs under `specs/`.

## Shape-shape interactions

Shape-shape interactions will require:
- Two poses (one per shape) instead of pose + fixed plane.
- A relative position/orientation representation.
- A different overlap definition (minimum distance between two shapes).

These are open design questions to be resolved when the first shape-shape interaction is requested.
