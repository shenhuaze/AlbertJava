package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * 全连接层
 */
public class DenseLayer {
    private Matrix kernel;
    private Matrix bias;
    private String activationStr;

    public DenseLayer(String kernelFile, String biasFile, int inputDim, int outputDim, String activationStr) {
        this.kernel = MatrixLoader.loadTextFile(kernelFile, inputDim, outputDim);
        this.bias = MatrixLoader.loadTextFile(biasFile, 1, outputDim);
        this.activationStr = activationStr;
    }

    public Matrix forward(Matrix input) {
        Matrix output = input.times(this.kernel).plusBroadcastAxis0(this.bias).activate(activationStr);
        return output;
    }
}
