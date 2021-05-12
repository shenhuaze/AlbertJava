package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * TransformerEncoder的前向全连接层
 */
public class FeedForwardLayer {
    private DenseLayer intermediateLayer;
    private DenseLayer outputLayer;
    private LayerNormalization layerNormalization;

    private Matrix ffnIntermediate;
    private Matrix ffnOutputBeforeLayerNorm;
    private Matrix ffnOutput;

    public FeedForwardLayer(FilePathConfig filePathConfig,
                            AlbertConfig albertConfig) {
        this.intermediateLayer =
                new DenseLayer(filePathConfig.getFfnIntermediateKernelFile(), filePathConfig.getFfnIntermediateBiasFile(), albertConfig.getHiddenSize(), albertConfig.getIntermediateSize(), albertConfig.getHiddenAct());
        this.outputLayer =
                new DenseLayer(filePathConfig.getFfnOutputKernelFile(), filePathConfig.getFfnOutputBiasFile(), albertConfig.getIntermediateSize(), albertConfig.getHiddenSize(), "linear");
        this.layerNormalization =
                new LayerNormalization(filePathConfig.getFfnLayerNormBetaFile(), filePathConfig.getFfnLayerNormGammaFile(), albertConfig.getHiddenSize());
    }

    public Matrix forward(Matrix attentionOutput) {
        ffnIntermediate = intermediateLayer.forward(attentionOutput);
        ffnOutputBeforeLayerNorm = outputLayer.forward(ffnIntermediate);
        ffnOutput = layerNormalization.forward(ffnOutputBeforeLayerNorm.plus(attentionOutput));
        return ffnOutput;
    }

    public Matrix getFfnIntermediate() {
        return ffnIntermediate;
    }

    public Matrix getFfnOutputBeforeLayerNorm() {
        return ffnOutputBeforeLayerNorm;
    }

    public Matrix getFfnOutput() {
        return ffnOutput;
    }

    public static void main(String[] args) {
        // [batch_size * seq_length, hidden_size]
        Matrix attentionOutput = new Matrix(4096, 64);

        FilePathConfig filePathConfig = new FilePathConfig("/file_path_config.properties");
        AlbertConfig albertConfig = new AlbertConfig("/albert_config.properties");

        FeedForwardLayer feedForwardLayer = new FeedForwardLayer(filePathConfig, albertConfig);
        Matrix encoderOutput = feedForwardLayer.forward(attentionOutput);
        System.out.println();
    }
}
