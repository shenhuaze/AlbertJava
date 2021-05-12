package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * 层归一化
 */
public class LayerNormalization {
    private Matrix beta;
    private Matrix gamma;
    private double epsilon = 1e-12;

    public LayerNormalization(String betaFile, String gammaFile, int dimension) {
        this.beta = MatrixLoader.loadTextFile(betaFile, 1, dimension);
        this.gamma = MatrixLoader.loadTextFile(gammaFile, 1, dimension);
    }

    public Matrix forward(Matrix input) {
        Matrix mean = input.meanAxis1();
        Matrix var = input.minus(mean).square().meanAxis1();
        Matrix std = var.squareRoot().plusScalar(this.epsilon);
        Matrix norm = input.minus(mean).arrayRightDivide(std);
        norm = norm.arrayTimesBroadcastAxis0(this.gamma).plusBroadcastAxis0(this.beta);
        return norm;
    }
}
