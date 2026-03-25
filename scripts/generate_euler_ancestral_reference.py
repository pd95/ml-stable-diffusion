#!/usr/bin/env python3

import json
from pathlib import Path

import torch
from diffusers import EulerAncestralDiscreteScheduler


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().flatten().tolist()]


def make_case(step_count: int) -> dict:
    scheduler = EulerAncestralDiscreteScheduler(
        num_train_timesteps=10,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
    )
    scheduler.set_timesteps(step_count)

    case = {
        "timesteps": tensor_to_list(scheduler.timesteps),
        "sigmas": tensor_to_list(scheduler.sigmas),
    }

    if step_count == 4:
        sample = torch.tensor([1.0, -2.0])
        original = torch.tensor([3.0])
        noise = torch.tensor([2.0])
        output = torch.tensor([2.0])
        step_sample = torch.tensor([10.0])

        case["scale_model_input"] = {
            "sample": tensor_to_list(sample),
            "timeStep": float(scheduler.timesteps[0]),
            "result": tensor_to_list(scheduler.scale_model_input(sample.clone(), scheduler.timesteps[0])),
        }
        case["add_noise_full"] = {
            "original": tensor_to_list(original),
            "noise": tensor_to_list(noise),
            "timeStep": float(scheduler.timesteps[0]),
            "result": tensor_to_list(
                scheduler.add_noise(original, noise, torch.tensor([scheduler.timesteps[0]]))
            ),
        }
        case["add_noise_strength_half"] = {
            "original": tensor_to_list(original),
            "noise": tensor_to_list(noise),
            "timeStep": float(scheduler.timesteps[2]),
            "result": tensor_to_list(
                scheduler.add_noise(original, noise, torch.tensor([scheduler.timesteps[2]]))
            ),
        }
        step_result = scheduler.step(
            output,
            scheduler.timesteps[0],
            step_sample,
            generator=torch.Generator().manual_seed(123)
        )
        case["step_first"] = {
            "sample": tensor_to_list(step_sample),
            "output": tensor_to_list(output),
            "timeStep": float(scheduler.timesteps[0]),
            "prevSample": tensor_to_list(step_result.prev_sample),
            "predOriginalSample": tensor_to_list(step_result.pred_original_sample),
            "seed": 123,
        }

    if step_count == 3:
        sample = torch.tensor([4.0])
        original = torch.tensor([3.0])
        noise = torch.tensor([2.0])
        output = torch.tensor([2.0])
        step_sample = torch.tensor([10.0])

        case["scale_model_input_fractional"] = {
            "sample": tensor_to_list(sample),
            "timeStep": float(scheduler.timesteps[1]),
            "result": tensor_to_list(scheduler.scale_model_input(sample.clone(), scheduler.timesteps[1])),
        }
        case["add_noise_fractional"] = {
            "original": tensor_to_list(original),
            "noise": tensor_to_list(noise),
            "timeStep": float(scheduler.timesteps[1]),
            "result": tensor_to_list(
                scheduler.add_noise(original, noise, torch.tensor([scheduler.timesteps[1]]))
            ),
        }
        step_result = scheduler.step(
            output,
            scheduler.timesteps[1],
            step_sample,
            generator=torch.Generator().manual_seed(123)
        )
        case["step_fractional"] = {
            "sample": tensor_to_list(step_sample),
            "output": tensor_to_list(output),
            "timeStep": float(scheduler.timesteps[1]),
            "prevSample": tensor_to_list(step_result.prev_sample),
            "predOriginalSample": tensor_to_list(step_result.pred_original_sample),
            "seed": 123,
        }

    return case


def main() -> None:
    output_path = Path("swift/StableDiffusionTests/Resources/euler_ancestral_reference.json")
    fixture = {
        "diffusersVersion": "0.37.1",
        "stepCount4": make_case(4),
        "stepCount3": make_case(3),
    }
    output_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
