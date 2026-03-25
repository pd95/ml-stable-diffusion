# Euler Scheduler Plan

## Goal
Add an Euler scheduler to the Swift package for Stable Diffusion and expose it through the existing pipeline and CLI interfaces.

## Delivery Phases

### Phase 1: Basic package integration
- Implement a pragmatic Euler scheduler that fits the current Swift scheduler architecture.
- Focus on text-to-image support first.
- Keep the existing `Int` timestep plumbing unless it blocks a usable implementation.
- Add CLI and pipeline integration so the scheduler can be selected and tested on macOS.
- Add math-level unit tests for the new scheduler.
- This phase is explicitly **not guaranteed to match Hugging Face diffusers exactly**.

Status:
- Completed.
- The Swift package now includes `EulerDiscreteScheduler`.
- The scheduler is wired into the SD and SDXL pipelines and exposed through the CLI as `--scheduler euler`.
- Build validation succeeded on the Apple toolchain through Xcode MCP.
- Package tests pass, including Euler-specific unit tests.
- Manual CLI smoke testing on macOS produced valid output images.

### Phase 2: Diffusers parity
- Revisit timestep representation and move to `Float` timesteps if required.
- Revisit image-to-image noise injection behavior.
- Tighten scheduler math and input scaling to match diffusers more closely.
- Add parity-oriented validation against known Python/diffusers outputs.

Status:
- In progress.
- Float timestep plumbing has now been implemented across the SD and SDXL scheduler path.
- `Scheduler`, `Unet`, and `ControlNet` now accept `Float` timesteps where the pipeline uses scheduler-emitted values.
- `EulerDiscreteScheduler` now preserves fractional timesteps and interpolates sigma values for `scaleModelInput(...)`.
- Added test coverage for fractional timestep behavior.
- Text-to-image Euler behavior is now closer to diffusers than the original integer-rounded implementation.

Still open:
- Revisit `addNoise(...)` and timestep/strength behavior for image-to-image parity.
- Validate scheduler behavior against Python/diffusers reference outputs rather than only smoke tests and math-level tests.
- Decide whether further sigma/timestep interpolation should be applied in more places for stricter parity.

## Current State
- The Swift package already supports `PNDMScheduler`, `DPMSolverMultistepScheduler`, and `DiscreteFlowScheduler`.
- Scheduler selection is wired through `StableDiffusionScheduler` and `SchedulerOption`.
- The denoising loops in `StableDiffusionPipeline.swift` and `StableDiffusionXLPipeline.swift` now call `scheduler.scaleModelInput(...)` before UNet execution.
- The SD and SDXL scheduler path now uses `Float` timesteps end-to-end.
- The current Euler implementation is validated for text-to-image, but image-to-image parity remains unfinished.

## Required Changes

### 1. Add the scheduler implementation
- Create `swift/StableDiffusion/pipeline/EulerDiscreteScheduler.swift`.
- Mirror the Hugging Face Euler discrete algorithm closely enough to preserve expected behavior for epsilon-prediction Stable Diffusion models.
- Store sigma schedule, convert model output to denoised sample form, and implement Euler stepping.

Status:
- Completed for the current Swift implementation.

### 2. Extend the scheduler abstraction
- Update `swift/StableDiffusion/pipeline/Scheduler.swift`.
- Add a default `scaleModelInput(sample:timeStep:)` method that returns the input unchanged for existing schedulers.
- Consider whether `initNoiseSigma` should become scheduler-specific for Euler instead of using the protocol default.

Status:
- Completed.
- `scaleModelInput(...)` was added.
- Scheduler timestep plumbing has since been expanded from `Int` to `Float` for SD and SDXL compatibility with fractional Euler timesteps.

### 3. Integrate scheduler scaling into denoising
- Update `swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift`.
- Update `swift/StableDiffusion/pipeline/StableDiffusionXLPipeline.swift`.
- Before each UNet call, pass latents through `scheduler.scaleModelInput(...)`.
- Keep `DiscreteFlowScheduler` unchanged for SD3 unless we later decide to unify the denoising loop further.

Status:
- Completed.
- SD3 already used float timesteps; SD and SDXL now follow the same general direction.

### 4. Register the scheduler
- Add `.eulerDiscreteScheduler` to `StableDiffusionScheduler` in `swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift`.
- Add switch cases in the scheduler factory blocks in:
  - `swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift`
  - `swift/StableDiffusion/pipeline/StableDiffusionXLPipeline.swift`
- Add `euler` to `SchedulerOption` in `swift/StableDiffusionCLI/main.swift`.
- Update CLI help text to mention Euler.

Status:
- Completed.

### 5. Add tests
- Add a new XCTest file under `swift/StableDiffusionTests/`.
- Test sigma/timestep generation.
- Test `scaleModelInput(...)`.
- Test one or two scheduler `step(...)` transitions against fixed expected values.
- Prefer math-level unit tests over full inference tests so they remain runnable without model assets.

Status:
- Completed, using the existing `swift/StableDiffusionTests/StableDiffusionTests.swift` file.
- Current coverage includes timestep generation, input scaling, Euler stepping behavior, add-noise behavior, and fractional timestep handling.

## Open Design Decision
The largest design question is timestep type.

### Option A: Minimal first pass
- Keep `Int` timesteps in the protocol and round any Euler-derived indices as needed.
- Lower refactor cost.
- Higher risk of mismatch with diffusers behavior.

### Option B: Parity-oriented refactor
- Change scheduler timesteps and UNet timestep plumbing from `Int` to `Float`.
- Touch `Scheduler`, the denoising loops, and `Unet.predictNoise(...)`.
- Better match to diffusers Euler scheduling.

Outcome:
- Phase 1 started with Option A.
- Phase 2 has now partially executed Option B.
- The Swift SD/SDXL scheduler path uses `Float` timesteps, which was necessary for fractional Euler timesteps and closer diffusers alignment.

## Files Expected To Change
- `swift/StableDiffusion/pipeline/Scheduler.swift`
- `swift/StableDiffusion/pipeline/EulerDiscreteScheduler.swift` (new)
- `swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift`
- `swift/StableDiffusion/pipeline/StableDiffusionXLPipeline.swift`
- `swift/StableDiffusionCLI/main.swift`
- `swift/StableDiffusionTests/...` (new or updated test file)
- `swift/StableDiffusion/pipeline/Unet.swift`
- `swift/StableDiffusion/pipeline/ControlNet.swift`

Additional follow-up files likely for remaining parity work:
- `swift/StableDiffusion/pipeline/StableDiffusion3Pipeline.swift` if scheduler protocol unification continues
- Python reference code or test fixtures for parity comparisons

## Review Of This Plan
- The Phase 1 integration work is complete and validated.
- The timestep-representation risk turned out to be real; `Float` timesteps were required for the next parity step, and `Unet.swift` plus `ControlNet.swift` became part of that work.
- The main remaining technical risk is still image-to-image parity, especially `addNoise(...)` behavior and exact strength/timestep handling.
- Another remaining gap is validation quality: we have build coverage, package tests, and smoke tests, but not diffusers reference-parity tests yet.
- The plan remains sound: the project has moved from “usable Euler integration” into “incremental parity tightening,” with clear next work on image-to-image and reference validation.
