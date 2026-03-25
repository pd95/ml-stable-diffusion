// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import XCTest
import CoreML
@testable import StableDiffusion

@available(iOS 16.2, macOS 13.1, *)
final class StableDiffusionTests: XCTestCase {

    var vocabFileInBundleURL: URL {
        let fileName = "vocab"
        guard let url = Bundle.module.url(forResource: fileName, withExtension: "json") else {
            fatalError("BPE tokenizer vocabulary file is missing from bundle")
        }
        return url
    }

    var mergesFileInBundleURL: URL {
        let fileName = "merges"
        guard let url = Bundle.module.url(forResource: fileName, withExtension: "txt") else {
            fatalError("BPE tokenizer merges file is missing from bundle")
        }
        return url
    }

    func testBPETokenizer() throws {

        let tokenizer = try BPETokenizer(mergesAt: mergesFileInBundleURL, vocabularyAt: vocabFileInBundleURL)

        func testPrompt(prompt: String, expectedIds: [Int]) {

            let (tokens, ids) = tokenizer.tokenize(input: prompt)

            print("Tokens          = \(tokens)\n")
            print("Expected tokens = \(expectedIds.map({ tokenizer.token(id: $0) }))")
            print("ids             = \(ids)\n")
            print("Expected Ids    = \(expectedIds)\n")

            XCTAssertEqual(ids,expectedIds)
        }

        testPrompt(prompt: "a photo of an astronaut riding a horse on mars",
                   expectedIds: [49406, 320, 1125, 539, 550, 18376, 6765, 320, 4558, 525, 7496, 49407])

        testPrompt(prompt: "Apple CoreML developer tools on a Macbook Air are fast",
                   expectedIds: [49406,  3055, 19622,  5780, 10929,  5771,   525,   320, 20617,
                                 1922,   631,  1953, 49407])
    }

    func test_randomNormalValues_matchNumPyRandom() {
        var random = NumPyRandomSource(seed: 12345)
        let samples = random.normalArray(count: 10_000)
        let last5 = samples.suffix(5)

        // numpy.random.seed(12345); print(numpy.random.randn(10000)[-5:])
        let expected = [-0.86285345, 2.15229409, -0.00670556, -1.21472309, 0.65498866]

        for (value, expected) in zip(last5, expected) {
            XCTAssertEqual(value, expected, accuracy: .ulpOfOne.squareRoot())
        }
    }

    func testEulerDiscreteSchedulerTimestepsAndScaling() {
        let scheduler = EulerDiscreteScheduler(stepCount: 4, trainStepCount: 10)

        XCTAssertEqual(scheduler.timeSteps, [9, 6, 3, 0])

        let sample = MLShapedArray<Float32>(scalars: [Float32(1), Float32(-2)], shape: [2])
        let scaled = scheduler.scaleModelInput(sample: sample, timeStep: 9)

        let sigma = scheduler.initNoiseSigma
        let scale = 1.0 / sqrt(sigma * sigma + 1.0)
        XCTAssertEqual(scaled.scalars[0], sample.scalars[0] * scale, accuracy: 1e-6)
        XCTAssertEqual(scaled.scalars[1], sample.scalars[1] * scale, accuracy: 1e-6)
    }

    func testEulerDiscreteSchedulerStepUsesEulerUpdate() {
        let scheduler = EulerDiscreteScheduler(stepCount: 4, trainStepCount: 10)
        let sample = MLShapedArray<Float32>(scalars: [Float32(10)], shape: [1])
        let output = MLShapedArray<Float32>(scalars: [Float32(2)], shape: [1])

        let next = scheduler.step(output: output, timeStep: 9, sample: sample)

        let sigma = scheduler.initNoiseSigma
        let nextSigma = scheduler.timeSteps.count > 1
            ? sqrt((1 - scheduler.alphasCumProd[scheduler.timeSteps[1]]) / scheduler.alphasCumProd[scheduler.timeSteps[1]])
            : 0
        let expectedDenoised = sample.scalars[0] - sigma * output.scalars[0]
        let expectedNext = sample.scalars[0] + (nextSigma - sigma) * output.scalars[0]
        XCTAssertEqual(next.scalars[0], expectedNext, accuracy: 1e-6)

        guard let last = scheduler.modelOutputs.last else {
            return XCTFail("Expected a denoised sample to be recorded")
        }
        XCTAssertEqual(last.scalars[0], expectedDenoised, accuracy: 1e-6)
    }

    func testEulerDiscreteSchedulerAddNoiseUsesSigmaSchedule() {
        let scheduler = EulerDiscreteScheduler(stepCount: 4, trainStepCount: 10)
        let original = MLShapedArray<Float32>(scalars: [Float32(3)], shape: [1])
        let noise = MLShapedArray<Float32>(scalars: [Float32(2)], shape: [1])

        let noisy = scheduler.addNoise(originalSample: original, noise: [noise], strength: 1.0)

        let expected = original.scalars[0] + scheduler.initNoiseSigma * noise.scalars[0]
        XCTAssertEqual(noisy.count, 1)
        XCTAssertEqual(noisy[0].scalars[0], expected, accuracy: 1e-6)
    }
}
