// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.

import Accelerate
import CoreML

/// A pragmatic Euler discrete scheduler for Stable Diffusion.
///
/// This is a phase 1 implementation that follows the current Swift scheduler
/// architecture and uses integer timesteps. It is intended to be usable first
/// and tightened against diffusers behavior in a later phase.
@available(iOS 16.2, macOS 13.1, *)
public final class EulerDiscreteScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Int]

    private let sigmas: [Float]

    public private(set) var modelOutputs: [MLShapedArray<Float32>] = []

    private var counter = 0

    public init(
        stepCount: Int = 50,
        trainStepCount: Int = 1000,
        betaSchedule: BetaSchedule = .scaledLinear,
        betaStart: Float = 0.00085,
        betaEnd: Float = 0.012
    ) {
        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount

        switch betaSchedule {
        case .linear:
            self.betas = linspace(betaStart, betaEnd, trainStepCount)
        case .scaledLinear:
            self.betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map { $0 * $0 }
        }

        self.alphas = betas.map { 1.0 - $0 }
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i - 1]
        }
        self.alphasCumProd = alphasCumProd

        if stepCount == 1 {
            self.timeSteps = [trainStepCount - 1]
        } else {
            self.timeSteps = linspace(0, Float(trainStepCount - 1), stepCount)
                .reversed()
                .map { Int(round($0)) }
        }

        self.sigmas = timeSteps.map { Self.sigma(from: alphasCumProd[$0]) } + [0]
    }

    public var initNoiseSigma: Float {
        sigmas.first ?? 1
    }

    public func scaleModelInput(sample: MLShapedArray<Float32>, timeStep t: Int) -> MLShapedArray<Float32> {
        let sigma = Self.sigma(from: alphasCumProd[min(max(t, 0), alphasCumProd.count - 1)])
        let scale = 1.0 / sqrt(sigma * sigma + 1.0)
        return MLShapedArray(unsafeUninitializedShape: sample.shape) { scalars, _ in
            sample.withUnsafeShapedBufferPointer { buffer, _, _ in
                for i in 0..<sample.scalarCount {
                    scalars.initializeElement(at: i, to: buffer[i] * scale)
                }
            }
        }
    }

    public func addNoise(
        originalSample: MLShapedArray<Float32>,
        noise: [MLShapedArray<Float32>],
        strength: Float
    ) -> [MLShapedArray<Float32>] {
        let startStep = max(inferenceStepCount - Int(Float(inferenceStepCount) * strength), 0)
        guard startStep < timeSteps.count else {
            return noise.map { _ in originalSample }
        }

        let sigma = sigmas[startStep]
        return noise.map {
            weightedSum([1.0, Double(sigma)], [originalSample, $0])
        }
    }

    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Int,
        sample s: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        _ = t
        let stepIndex = min(counter, timeSteps.count - 1)
        let sigma = sigmas[stepIndex]
        let prevSigma = sigmas[min(stepIndex + 1, sigmas.count - 1)]
        let dt = prevSigma - sigma

        let denoisedSample = weightedSum(
            [1.0, Double(-sigma)],
            [s, output]
        )
        modelOutputs.append(denoisedSample)

        counter += 1
        return weightedSum(
            [1.0, Double(dt)],
            [s, output]
        )
    }
}

@available(iOS 16.2, macOS 13.1, *)
private extension EulerDiscreteScheduler {
    static func sigma(from alphaCumProd: Float) -> Float {
        sqrt((1 - alphaCumProd) / alphaCumProd)
    }
}
