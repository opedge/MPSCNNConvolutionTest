//
//  MPSCNNConvolutionTest.swift
//  MPSCNNConvolutionTest
//
//  Created by Oleg Poyaganov on 30/11/2017.
//  Copyright © 2017 Oleg Poyaganov. All rights reserved.
//

import XCTest
import Metal
import Accelerate
import MetalPerformanceShaders

class WeightsDataSource: NSObject, MPSCNNConvolutionDataSource {
    let desc: MPSCNNConvolutionDescriptor
    let weight: [Float]
    let bias: [Float]
    
    init(desc: MPSCNNConvolutionDescriptor, weight: [Float], bias: [Float]) {
        self.desc = desc
        self.weight = weight
        self.bias = bias
    }
    
    func descriptor() -> MPSCNNConvolutionDescriptor {
        return desc
    }
    
    func dataType() -> MPSDataType {
        return .float32
    }
    
    func weights() -> UnsafeMutableRawPointer {
        return UnsafeMutableRawPointer(mutating: weight)
    }
    
    func biasTerms() -> UnsafeMutablePointer<Float>? {
        return UnsafeMutablePointer(mutating: bias)
    }
    
    func load() -> Bool {
        return true
    }
    
    func purge() {
    }
    
    func label() -> String? {
        return "Conv"
    }
}

func generateRandom(size: Int) -> [Float] {
    var array = [Float](repeating: 0, count: size)
    for i in 0..<size {
        array[i] = Float(arc4random()) / 0xFFFFFFFF
    }
    return array
}

func copyDataToImage(data: [Float], shape: [Int], image: MPSImage) {
    assert(shape.count == 3)
    
    let sliceCount = (shape[0] + 3) >> 2
    for slice in 0..<sliceCount {
        var sliceData = [Float](repeating: 0, count: 4 * shape[1] * shape[2])
        for row in 0..<shape[1] {
            for col in 0..<shape[2] {
                for c in 0..<4 {
                    let channel = 4 * slice + c
                    if channel < shape[0] {
                        sliceData[row * shape[2] * 4 + col * 4 + c] = data[channel * shape[1] * shape[2] + row * shape[1] + col]
                    }
                }
            }
        }
        
        image.texture.replace(
            region: MTLRegionMake2D(0, 0, shape[2], shape[1]),
            mipmapLevel: 0,
            slice: slice,
            withBytes: UnsafeRawPointer(sliceData),
            bytesPerRow: 4 * shape[2] * MemoryLayout<Float>.size,
            bytesPerImage: 4 * shape[1] * shape[2] * MemoryLayout<Float>.size
        )
    }
}

func getDataFromImage(_ image: MPSImage) -> [Float] {
    let sliceCount = (image.featureChannels + 3) >> 2
    var result = [Float](repeating: 0, count: image.featureChannels * image.width * image.height)
    for slice in 0..<sliceCount {
        var sliceData = [Float](repeating: 0, count: 4 * image.height * image.width)
        image.texture.getBytes(
            &sliceData,
            bytesPerRow: 4 * image.width * MemoryLayout<Float>.size,
            bytesPerImage: 4 * image.height * image.width * MemoryLayout<Float>.size,
            from: MTLRegionMake2D(0, 0, image.width, image.height),
            mipmapLevel: 0,
            slice: slice
        )
        for row in 0..<image.height {
            for col in 0..<image.width {
                for c in 0..<4 {
                    let channel = 4 * slice + c
                    if channel < image.featureChannels {
                        result[channel * image.height * image.width + row * image.height + col] = sliceData[row * image.width * 4 + col * 4 + c]
                    }
                }
            }
        }
    }
    return result
}

func getMetalDevice(isLowPower: Bool) -> MTLDevice {
    for device in MTLCopyAllDevices() {
        if device.isLowPower == isLowPower {
            return device
        }
    }
    XCTFail("Metal device is not available")
    fatalError()
}

func transposeWeights(_ weights: [Float], kernelWidth: Int, kernelHeight: Int, inputChannels: Int, outputChannels: Int) -> [Float] {
    var transposed = [Float](repeating: 0, count: weights.count)
    var i = 0
    for oc in 0..<outputChannels {
        for ky in 0..<kernelHeight {
            for kx in 0..<kernelWidth {
                for ic in 0..<inputChannels {
                    transposed[kx + kernelWidth*(ky + kernelHeight*(ic + inputChannels*oc))] = weights[i]
                    i += 1
                }
            }
        }
    }
    return transposed
}

func getBNNSOutput(params: BNNSConvolutionLayerParameters, inputShape: [Int], input: [Float], outputShape: [Int]) -> [Float] {
    var convParams = params
    
    var inputDescriptor = BNNSImageStackDescriptor(
        width: inputShape[2],
        height: inputShape[1],
        channels: inputShape[0],
        row_stride: inputShape[2],
        image_stride: inputShape[2] * inputShape[1],
        data_type: .float
    )
    
    var outputDescriptor = BNNSImageStackDescriptor(
        width: outputShape[2],
        height: outputShape[1],
        channels: outputShape[0],
        row_stride: outputShape[2],
        image_stride: outputShape[2] * outputShape[1],
        data_type: .float
    )
    
    var filterParams = BNNSFilterParameters(flags: 0, n_threads: 0, alloc_memory: nil, free_memory: nil)
    let conv = BNNSFilterCreateConvolutionLayer(
        &inputDescriptor,
        &outputDescriptor,
        &convParams,
        &filterParams
    )!
    defer { BNNSFilterDestroy(conv) }
    
    let output = [Float](repeating: 0, count: outputShape.reduce(1, *))
    
    let applyResult = BNNSFilterApply(conv, UnsafeRawPointer(input), UnsafeMutableRawPointer(mutating: output))
    XCTAssertEqual(applyResult, 0)
    
    return output
}

func printArray(_ arr: [Float], count: Int = 16) {
    print(arr[0..<min(arr.count, count)])
}

func assertAlmostEqual(arr1: [Float], arr2: [Float], accuracy: Float) {
    print("First:")
    printArray(arr1)
    print("Second:")
    printArray(arr2)
    XCTAssertEqual(arr1.count, arr2.count, "Array sizes do not equal")
    for i in 0..<arr1.count {
        XCTAssertEqual(arr1[i], arr2[i], accuracy: accuracy)
    }
}

func testWithDevice(_ device: MTLDevice) {
    // Conv params
    let inputShape = [3, 32, 32]
    let kernelShape = [3, 3]
    let outputChannels = 3
    let strides = [1, 1]
    let pads = [2, 2]
    let outputHeight = (inputShape[1] - 1) * strides[0] + kernelShape[0] - 2 * pads[0]
    let outputWidth = (inputShape[2] - 1) * strides[1] + kernelShape[1] - 2 * pads[1]
    let outputShape = [outputChannels, outputHeight, outputWidth]
    
    let input = generateRandom(size: inputShape.reduce(1, *))
    let weight = generateRandom(size: outputChannels * kernelShape[0] * kernelShape[1] * inputShape[0])
    let bias = generateRandom(size: outputChannels)
    
    // Get reference output from BNNS convolution
    let transposedWeight = transposeWeights(
        weight,
        kernelWidth: kernelShape[1], kernelHeight: kernelShape[0],
        inputChannels: inputShape[0], outputChannels: outputChannels
    )
    let params = BNNSConvolutionLayerParameters(
        x_stride: strides[1],
        y_stride: strides[0],
        x_padding: pads[1],
        y_padding: pads[0],
        k_width: kernelShape[1],
        k_height: kernelShape[0],
        in_channels: inputShape[0],
        out_channels: outputChannels,
        weights: BNNSLayerData(data: UnsafeRawPointer(transposedWeight), data_type: .float),
        bias: BNNSLayerData(data: UnsafeRawPointer(bias), data_type: .float),
        activation: .identity
    )
    
    let bnnsOutput = getBNNSOutput(params: params, inputShape: inputShape, input: input, outputShape: outputShape)
    
    // Run MPS Convolution
    guard let commandQueue = device.makeCommandQueue() else {
        XCTFail("Couldn't init metal command queue")
        return
    }
    
    let convDesc = MPSCNNConvolutionDescriptor(
        kernelWidth: kernelShape[1], kernelHeight: kernelShape[0],
        inputFeatureChannels: inputShape[0], outputFeatureChannels: outputChannels
    )
    convDesc.strideInPixelsX = strides[1]
    convDesc.strideInPixelsY = strides[0]
    let dataSource = WeightsDataSource(desc: convDesc, weight: weight, bias: bias)
    let conv = MPSCNNConvolution(device: device, weights: dataSource)
    conv.edgeMode = .zero
    conv.offset = MPSOffset(
        x: kernelShape[1] / 2 - pads[1],
        y: kernelShape[0] / 2 - pads[0],
        z: 0
    )
    
    let inputDesc = MPSImageDescriptor(channelFormat: .float32, width: inputShape[2], height: inputShape[1], featureChannels: inputShape[0])
    inputDesc.storageMode = .managed
    let inputImage = MPSImage(device: device, imageDescriptor: inputDesc)
    copyDataToImage(data: input, shape: inputShape, image: inputImage)
    
    let commandBuffer = commandQueue.makeCommandBuffer()!
    var blit = commandBuffer.makeBlitCommandEncoder()!
    blit.synchronize(resource: inputImage.texture)
    blit.endEncoding()
    
    let outputDesc = MPSImageDescriptor(channelFormat: .float32, width: outputShape[2], height: outputShape[1], featureChannels: outputShape[0])
    let outputImage = MPSImage(device: device, imageDescriptor: outputDesc)
    
    conv.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)
    
    blit = commandBuffer.makeBlitCommandEncoder()!
    blit.synchronize(resource: outputImage.texture)
    blit.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    let mpsOutput = getDataFromImage(outputImage)
    
    assertAlmostEqual(arr1: mpsOutput, arr2: bnnsOutput, accuracy: 0.001)
}

class MPSCNNConvolutionTest: XCTestCase {
    override func setUp() {
        super.setUp()
        continueAfterFailure = false
    }
    
    override func tearDown() {
        continueAfterFailure = true
        super.tearDown()
    }
    
    func testStandardDevice() {
        testWithDevice(getMetalDevice(isLowPower: false))
    }
    
    func testLowPoweredDevice() {
        testWithDevice(getMetalDevice(isLowPower: true))
    }
}
