# Scheduler Roadmap

## Current State
- `EulerDiscreteScheduler` is implemented, wired through the Swift pipelines and CLI, and validated with package tests plus macOS smoke testing.
- The img2img initialization bug uncovered during Euler work has been fixed: image-to-image now starts from unit Gaussian noise and lets the scheduler apply the selected start-step sigma.
- The Swift package has direct diffusers-reference fixture coverage for representative Euler scheduler math.

## Next Goal
Add `EulerAncestralDiscreteScheduler` to evaluate whether ancestral Euler sampling gives better low-step results than `dpmpp` or plain Euler in this codebase.

This is a new feature, not unfinished `EulerDiscreteScheduler` work.

Status:
- In progress.
- `EulerAncestralDiscreteScheduler` now exists in the Swift package and is wired into the SD/SDXL scheduler selection path.
- The scheduler uses the pipeline RNG for ancestral step noise so seeded behavior stays aligned with the repo's existing RNG options.
- Diffusers-reference fixture coverage is in place for representative ancestral Euler timesteps, input scaling, add-noise behavior, and stochastic step outputs.
- Package tests and Xcode build validation currently pass.

## Delivery Phases

### Phase 1: Text-to-image integration
- Implement `EulerAncestralDiscreteScheduler` for text-to-image first.
- Match diffusers ancestral Euler behavior closely enough for practical parity.
- Reuse the current float timestep scheduler plumbing.
- Expose the scheduler through the existing pipeline and CLI interfaces.
- Add math-level unit tests and diffusers-reference parity coverage.

Success criteria:
- The scheduler builds and runs through the existing SD/SDXL denoising loops.
- Package tests pass in Xcode MCP.
- macOS smoke testing shows deterministic seeded outputs and usable low-step results.

Status:
- Partially completed.
- Implementation, CLI exposure, and automated parity coverage are done.
- Manual macOS low-step validation is still pending.

### Phase 2: Image-to-image support
- Extend ancestral Euler behavior to image-to-image only after text-to-image is stable.
- Validate that img2img strength/start-step handling works sensibly with ancestral noise injection.
- Add targeted tests for img2img initialization and early denoising behavior.

Success criteria:
- Low-strength img2img preserves source structure.
- Higher-strength img2img transforms the source without collapsing into raw noise.

Status:
- Not yet validated.
- The current implementation reuses the existing img2img initialization path, but this phase is still pending explicit testing.

## Why This Feature
- Plain Euler is now correct and usable, but in this project it is not dramatically different from `dpmpp` at moderate step counts.
- `EulerAncestralDiscreteScheduler` is a better next candidate because it may converge to a recognizable image in fewer steps.
- That makes it more likely to provide a user-visible benefit rather than just compatibility surface area.

## Main Risks
- Ancestral Euler injects fresh noise during stepping, so it is not just a trivial variant of the current Euler scheduler.
- Determinism and seed handling matter more because the scheduler itself becomes stochastic.
- Diffusers parity is more sensitive because step-to-step randomness must line up with the scheduler math, not just the sigma schedule.
- Img2img may need extra care because both scheduler start-step logic and ancestral step noise affect behavior.

## Required Changes

### 1. Add the scheduler implementation
- Create `swift/StableDiffusion/pipeline/EulerAncestralDiscreteScheduler.swift`.
- Implement sigma schedule generation, model input scaling, ancestral Euler stepping, and diffusers-style noise handling.
- Start with epsilon-prediction support only unless the repo already requires more.

### 2. Extend scheduler selection
- Add `.eulerAncestralDiscreteScheduler` to `StableDiffusionScheduler`.
- Register the new scheduler in:
  - `swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift`
  - `swift/StableDiffusion/pipeline/StableDiffusionXLPipeline.swift`
- Consider whether SD3 should remain unchanged unless there is a clear compatibility story.

### 3. Expose it in the CLI
- Add a CLI option such as `euler_a` in `swift/StableDiffusionCLI/main.swift`.
- Update help text to list the new scheduler.

### 4. Add tests
- Add scheduler math tests under `swift/StableDiffusionTests/`.
- Add diffusers-reference fixtures and parity tests, following the pattern now used for plain Euler.
- Add seed/determinism tests if the ancestral implementation depends on scheduler-internal randomness.

### 5. Manual validation
- Run Xcode MCP package tests.
- Run low-step macOS smoke tests, especially around `10` and `15` steps.
- Compare outputs against `dpmpp` and plain Euler for the same prompt, seed, and model.

## Validation Strategy

### Automated
- Package tests should cover:
  - timestep/sigma generation
  - `scaleModelInput(...)`
  - `step(...)` against fixed reference outputs
  - any scheduler-specific noise sampling behavior
- Diffusers-reference fixtures should be used for representative ancestral scheduler operations.

### Manual
- Text-to-image:
  - compare `dpmpp`, `euler`, and `euler_a` at low step counts
  - use the same prompt, seed, and model
- Image-to-image, if Phase 2 is attempted:
  - compare behavior at `strength 0.2`, `0.5`, and `0.8`
  - verify preservation vs transformation remains sensible

## Files Likely To Change
- `swift/StableDiffusion/pipeline/EulerAncestralDiscreteScheduler.swift` (new)
- `swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift`
- `swift/StableDiffusion/pipeline/StableDiffusionXLPipeline.swift`
- `swift/StableDiffusionCLI/main.swift`
- `swift/StableDiffusionTests/StableDiffusionTests.swift`
- `swift/StableDiffusionTests/Resources/...` (new fixture files)
- `scripts/...` (fixture generator, if needed)

## Out Of Scope For The First Pass
- Broad scheduler refactors unrelated to ancestral Euler
- Multiple prediction modes unless clearly required
- SD3 integration unless there is a concrete use case
- Large benchmark infrastructure beyond focused smoke tests and parity fixtures
