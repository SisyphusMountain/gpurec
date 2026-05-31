# Workflow Transition Mechanics Cleanup Plan, 2026-05-31

## Target

`gpurec/workflow/_transitions.py` still had small post-collapse duplication in
private transition execution mechanics, especially the active-batch
`next_batch` return path and transition-result forwarding.

## Scope

- Keep transition classification order and all transition branches in place.
- Collapse duplicated `next_batch` return mechanics without changing checkpoint
  save, runtime-cache clear, batch selection, or solver-stage configuration
  timing.
- Remove only unused private parameters and duplicate no-op state calls.
- Do not change row fields, checkpoint status fields, optimizer phase handling,
  public workflow exports, or transition DTOs.

## Verification

- Existing direct transition tests cover checkpointing and step-stop status.
- Extend the `next_batch` direct test to cover the `checkpoint_every=0` branch.
- Run focused transition/workflow selectors plus the full CPU unit marker gate
  before committing.
