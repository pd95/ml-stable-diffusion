// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import XCTest
import CoreML
import Foundation
@testable import StableDiffusion

@available(iOS 16.2, macOS 13.1, *)
final class StableDiffusionTests: XCTestCase {

    struct EulerReferenceFixture: Decodable {
        struct Case: Decodable {
            let timesteps: [Float]
            let sigmas: [Float]
            let scale_model_input: TensorCase?
            let add_noise_full: TensorCase?
            let add_noise_strength_half: TensorCase?
            let step_first: StepCase?
            let scale_model_input_fractional: TensorCase?
            let add_noise_fractional: TensorCase?
            let step_fractional: StepCase?
        }

        struct TensorCase: Decodable {
            let sample: [Float]?
            let original: [Float]?
            let noise: [Float]?
            let timeStep: Float
            let result: [Float]
        }

        struct StepCase: Decodable {
            let sample: [Float]
            let output: [Float]
            let timeStep: Float
            let prevSample: [Float]
            let predOriginalSample: [Float]
            let seed: UInt32?
        }

        let diffusersVersion: String
        let stepCount4: Case
        let stepCount3: Case
    }

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

    var eulerReferenceFixture: EulerReferenceFixture {
        guard let url = Bundle.module.url(forResource: "euler_reference", withExtension: "json") else {
            fatalError("Euler reference fixture is missing from bundle")
        }
        do {
            return try JSONDecoder().decode(EulerReferenceFixture.self, from: Data(contentsOf: url))
        } catch {
            fatalError("Failed to decode Euler reference fixture: \(error)")
        }
    }

    var eulerAncestralReferenceFixture: EulerReferenceFixture {
        guard let url = Bundle.module.url(forResource: "euler_ancestral_reference", withExtension: "json") else {
            fatalError("Euler ancestral reference fixture is missing from bundle")
        }
        do {
            return try JSONDecoder().decode(EulerReferenceFixture.self, from: Data(contentsOf: url))
        } catch {
            fatalError("Failed to decode Euler ancestral reference fixture: \(error)")
        }
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

        let expectedTimesteps: [Float] = [9, 6, 3, 0]
        XCTAssertEqual(scheduler.timeSteps.count, expectedTimesteps.count)
        for (actual, expected) in zip(scheduler.timeSteps, expectedTimesteps) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }

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
            ? sqrt((1 - scheduler.alphasCumProd[Int(round(scheduler.timeSteps[1]))]) / scheduler.alphasCumProd[Int(round(scheduler.timeSteps[1]))])
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

    func testEulerDiscreteSchedulerSupportsFractionalTimesteps() {
        let scheduler = EulerDiscreteScheduler(stepCount: 3, trainStepCount: 10)
        let sample = MLShapedArray<Float32>(scalars: [Float32(4)], shape: [1])

        XCTAssertEqual(scheduler.timeSteps[1], 4.5, accuracy: 1e-6)

        let scaled = scheduler.scaleModelInput(sample: sample, timeStep: 4.5)

        let lowerSigma = sqrt((1 - scheduler.alphasCumProd[4]) / scheduler.alphasCumProd[4])
        let upperSigma = sqrt((1 - scheduler.alphasCumProd[5]) / scheduler.alphasCumProd[5])
        let sigma = lowerSigma + (upperSigma - lowerSigma) * 0.5
        let expectedScale = 1.0 / sqrt(sigma * sigma + 1.0)
        XCTAssertEqual(scaled.scalars[0], sample.scalars[0] * expectedScale, accuracy: 1e-6)
    }

    func testEulerDiscreteSchedulerImageToImageStrengthUsesMatchingStartStep() {
        let scheduler = EulerDiscreteScheduler(stepCount: 4, trainStepCount: 10)
        let strength: Float = 0.5
        let original = MLShapedArray<Float32>(scalars: [Float32(3)], shape: [1])
        let noise = MLShapedArray<Float32>(scalars: [Float32(2)], shape: [1])

        let timesteps = scheduler.calculateTimesteps(strength: strength)
        XCTAssertEqual(timesteps.count, 2)
        XCTAssertEqual(timesteps.first ?? -1, 3, accuracy: 1e-6)

        let noisy = scheduler.addNoise(originalSample: original, noise: [noise], strength: strength)

        let timestep = timesteps[0]
        let lowerSigma = sqrt((1 - scheduler.alphasCumProd[3]) / scheduler.alphasCumProd[3])
        let upperSigma = sqrt((1 - scheduler.alphasCumProd[4]) / scheduler.alphasCumProd[4])
        let sigma = lowerSigma + (upperSigma - lowerSigma) * (timestep - 3)
        let expected = original.scalars[0] + sigma * noise.scalars[0]
        XCTAssertEqual(noisy[0].scalars[0], expected, accuracy: 1e-6)
    }

    func testEulerDiscreteSchedulerMatchesDiffusersReferenceFixture() {
        let fixture = eulerReferenceFixture
        XCTAssertEqual(fixture.diffusersVersion, "0.37.1")

        let scheduler4 = EulerDiscreteScheduler(stepCount: 4, trainStepCount: 10)
        assertEqualFloatValues(scheduler4.timeSteps, fixture.stepCount4.timesteps)
        assertEqualShapedArrayScalars(scheduler4.scaleModelInput(
            sample: shapedArray(fixture.stepCount4.scale_model_input!.sample!),
            timeStep: fixture.stepCount4.scale_model_input!.timeStep
        ).scalars, fixture.stepCount4.scale_model_input!.result)
        assertEqualShapedArrayScalars(
            scheduler4.addNoise(
                originalSample: shapedArray(fixture.stepCount4.add_noise_full!.original!),
                noise: [shapedArray(fixture.stepCount4.add_noise_full!.noise!)],
                strength: 1.0
            )[0].scalars,
            fixture.stepCount4.add_noise_full!.result
        )
        assertEqualShapedArrayScalars(
            scheduler4.addNoise(
                originalSample: shapedArray(fixture.stepCount4.add_noise_strength_half!.original!),
                noise: [shapedArray(fixture.stepCount4.add_noise_strength_half!.noise!)],
                strength: 0.5
            )[0].scalars,
            fixture.stepCount4.add_noise_strength_half!.result
        )

        let step4 = scheduler4.step(
            output: shapedArray(fixture.stepCount4.step_first!.output),
            timeStep: fixture.stepCount4.step_first!.timeStep,
            sample: shapedArray(fixture.stepCount4.step_first!.sample)
        )
        assertEqualShapedArrayScalars(step4.scalars, fixture.stepCount4.step_first!.prevSample)
        assertEqualShapedArrayScalars(scheduler4.modelOutputs.last!.scalars, fixture.stepCount4.step_first!.predOriginalSample)

        let scheduler3 = EulerDiscreteScheduler(stepCount: 3, trainStepCount: 10)
        assertEqualFloatValues(scheduler3.timeSteps, fixture.stepCount3.timesteps)
        assertEqualShapedArrayScalars(scheduler3.scaleModelInput(
            sample: shapedArray(fixture.stepCount3.scale_model_input_fractional!.sample!),
            timeStep: fixture.stepCount3.scale_model_input_fractional!.timeStep
        ).scalars, fixture.stepCount3.scale_model_input_fractional!.result)
        assertEqualShapedArrayScalars(
            scheduler3.addNoise(
                originalSample: shapedArray(fixture.stepCount3.add_noise_fractional!.original!),
                noise: [shapedArray(fixture.stepCount3.add_noise_fractional!.noise!)],
                strength: 2.0 / 3.0
            )[0].scalars,
            fixture.stepCount3.add_noise_fractional!.result
        )

        let step3 = scheduler3.step(
            output: shapedArray(fixture.stepCount3.step_fractional!.output),
            timeStep: fixture.stepCount3.step_fractional!.timeStep,
            sample: shapedArray(fixture.stepCount3.step_fractional!.sample)
        )
        assertEqualShapedArrayScalars(step3.scalars, fixture.stepCount3.step_fractional!.prevSample)
        assertEqualShapedArrayScalars(scheduler3.modelOutputs.last!.scalars, fixture.stepCount3.step_fractional!.predOriginalSample)
    }

    func testEulerAncestralDiscreteSchedulerMatchesDiffusersReferenceFixture() {
        let fixture = eulerAncestralReferenceFixture
        XCTAssertEqual(fixture.diffusersVersion, "0.37.1")

        let scheduler4 = EulerAncestralDiscreteScheduler(stepCount: 4, trainStepCount: 10)
        assertEqualFloatValues(scheduler4.timeSteps, fixture.stepCount4.timesteps)
        assertEqualShapedArrayScalars(scheduler4.scaleModelInput(
            sample: shapedArray(fixture.stepCount4.scale_model_input!.sample!),
            timeStep: fixture.stepCount4.scale_model_input!.timeStep
        ).scalars, fixture.stepCount4.scale_model_input!.result)
        assertEqualShapedArrayScalars(
            scheduler4.addNoise(
                originalSample: shapedArray(fixture.stepCount4.add_noise_full!.original!),
                noise: [shapedArray(fixture.stepCount4.add_noise_full!.noise!)],
                strength: 1.0
            )[0].scalars,
            fixture.stepCount4.add_noise_full!.result
        )
        assertEqualShapedArrayScalars(
            scheduler4.addNoise(
                originalSample: shapedArray(fixture.stepCount4.add_noise_strength_half!.original!),
                noise: [shapedArray(fixture.stepCount4.add_noise_strength_half!.noise!)],
                strength: 0.5
            )[0].scalars,
            fixture.stepCount4.add_noise_strength_half!.result
        )

        var random4: RandomSource = TorchRandomSource(seed: fixture.stepCount4.step_first!.seed!)
        let step4 = scheduler4.step(
            output: shapedArray(fixture.stepCount4.step_first!.output),
            timeStep: fixture.stepCount4.step_first!.timeStep,
            sample: shapedArray(fixture.stepCount4.step_first!.sample),
            random: &random4
        )
        assertEqualShapedArrayScalars(step4.scalars, fixture.stepCount4.step_first!.prevSample)
        assertEqualShapedArrayScalars(scheduler4.modelOutputs.last!.scalars, fixture.stepCount4.step_first!.predOriginalSample)

        let scheduler3 = EulerAncestralDiscreteScheduler(stepCount: 3, trainStepCount: 10)
        assertEqualFloatValues(scheduler3.timeSteps, fixture.stepCount3.timesteps)
        assertEqualShapedArrayScalars(scheduler3.scaleModelInput(
            sample: shapedArray(fixture.stepCount3.scale_model_input_fractional!.sample!),
            timeStep: fixture.stepCount3.scale_model_input_fractional!.timeStep
        ).scalars, fixture.stepCount3.scale_model_input_fractional!.result)
        assertEqualShapedArrayScalars(
            scheduler3.addNoise(
                originalSample: shapedArray(fixture.stepCount3.add_noise_fractional!.original!),
                noise: [shapedArray(fixture.stepCount3.add_noise_fractional!.noise!)],
                strength: 2.0 / 3.0
            )[0].scalars,
            fixture.stepCount3.add_noise_fractional!.result
        )

        var random3: RandomSource = TorchRandomSource(seed: fixture.stepCount3.step_fractional!.seed!)
        let step3 = scheduler3.step(
            output: shapedArray(fixture.stepCount3.step_fractional!.output),
            timeStep: fixture.stepCount3.step_fractional!.timeStep,
            sample: shapedArray(fixture.stepCount3.step_fractional!.sample),
            random: &random3
        )
        assertEqualShapedArrayScalars(step3.scalars, fixture.stepCount3.step_fractional!.prevSample)
        assertEqualShapedArrayScalars(scheduler3.modelOutputs.last!.scalars, fixture.stepCount3.step_fractional!.predOriginalSample)
    }

    private func shapedArray(_ scalars: [Float]) -> MLShapedArray<Float32> {
        MLShapedArray<Float32>(scalars: scalars.map { Float32($0) }, shape: [scalars.count])
    }

    private func assertEqualShapedArrayScalars(_ actual: [Float32], _ expected: [Float], accuracy: Float = 1e-5, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: accuracy, file: file, line: line)
        }
    }

    private func assertEqualFloatValues(_ actual: [Float], _ expected: [Float], accuracy: Float = 1e-6, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: accuracy, file: file, line: line)
        }
    }
}
