// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.

import Accelerate
import CoreML

/// An ancestral Euler scheduler for Stable Diffusion.
@available(iOS 16.2, macOS 13.1, *)
public final class EulerAncestralDiscreteScheduler: Scheduler {
    public let trainStepCount: Int
    public let inferenceStepCount: Int
    public let betas: [Float]
    public let alphas: [Float]
    public let alphasCumProd: [Float]
    public let timeSteps: [Float]

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
            self.timeSteps = [Float(trainStepCount - 1)]
        } else {
            self.timeSteps = linspace(0, Float(trainStepCount - 1), stepCount).reversed()
        }

        self.sigmas = timeSteps.map { Self.interpolatedSigma(for: $0, alphasCumProd: alphasCumProd) } + [0]
    }

    public var initNoiseSigma: Float {
        sigmas.first ?? 1
    }

    public func scaleModelInput(sample: MLShapedArray<Float32>, timeStep t: Float) -> MLShapedArray<Float32> {
        let sigma = Self.interpolatedSigma(for: t, alphasCumProd: alphasCumProd)
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
        let startStep = startStep(for: strength)
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
        timeStep t: Float,
        sample s: MLShapedArray<Float32>
    ) -> MLShapedArray<Float32> {
        fatalError("EulerAncestralDiscreteScheduler requires pipeline RNG")
    }

    public func step(
        output: MLShapedArray<Float32>,
        timeStep t: Float,
        sample s: MLShapedArray<Float32>,
        random: inout RandomSource
    ) -> MLShapedArray<Float32> {
        let stepIndex = timeSteps.firstIndex(of: t) ?? min(counter, timeSteps.count - 1)
        let sigmaFrom = sigmas[stepIndex]
        let sigmaTo = sigmas[min(stepIndex + 1, sigmas.count - 1)]

        let predOriginalSample = weightedSum(
            [1.0, Double(-sigmaFrom)],
            [s, output]
        )
        modelOutputs.append(predOriginalSample)

        let sigmaUp: Float
        if sigmaFrom > 0 {
            sigmaUp = sqrt(max(0, (sigmaTo * sigmaTo) * (sigmaFrom * sigmaFrom - sigmaTo * sigmaTo) / (sigmaFrom * sigmaFrom)))
        } else {
            sigmaUp = 0
        }
        let sigmaDown = sqrt(max(0, sigmaTo * sigmaTo - sigmaUp * sigmaUp))
        let dt = sigmaDown - sigmaFrom

        let deterministicPart = weightedSum(
            [1.0, Double(dt)],
            [s, output]
        )

        let stochasticPart: MLShapedArray<Float32>
        if sigmaUp > 0 {
            stochasticPart = MLShapedArray<Float32>(
                converting: random.normalShapedArray(s.shape, mean: 0.0, stdev: 1.0)
            )
        } else {
            stochasticPart = MLShapedArray<Float32>(repeating: 0, shape: s.shape)
        }

        counter += 1
        return weightedSum(
            [1.0, Double(sigmaUp)],
            [deterministicPart, stochasticPart]
        )
    }
}

@available(iOS 16.2, macOS 13.1, *)
private extension EulerAncestralDiscreteScheduler {
    static func sigma(from alphaCumProd: Float) -> Float {
        sqrt((1 - alphaCumProd) / alphaCumProd)
    }

    static func interpolatedSigma(for timestep: Float, alphasCumProd: [Float]) -> Float {
        let clipped = min(max(timestep, 0), Float(alphasCumProd.count - 1))
        let lowerIndex = Int(floor(clipped))
        let upperIndex = min(lowerIndex + 1, alphasCumProd.count - 1)
        let lowerSigma = sigma(from: alphasCumProd[lowerIndex])
        let upperSigma = sigma(from: alphasCumProd[upperIndex])
        let fraction = clipped - Float(lowerIndex)
        return lowerSigma + (upperSigma - lowerSigma) * fraction
    }
}
