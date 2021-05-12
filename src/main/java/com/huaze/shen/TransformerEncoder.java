package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * TransformerEncoder
 */
public class TransformerEncoder {
    private MultiHeadSelfAttentionLayer multiHeadSelfAttentionLayer;
    private FeedForwardLayer feedForwardLayer;

    private Matrix attentionOutput;
    private Matrix encoderOutput;

    public TransformerEncoder(FilePathConfig filePathConfig,
                              AlbertConfig albertConfig) {
        this.multiHeadSelfAttentionLayer = new MultiHeadSelfAttentionLayer(filePathConfig, albertConfig);
        this.feedForwardLayer = new FeedForwardLayer(filePathConfig, albertConfig);
    }

    public Matrix forward(Matrix encoderInput, Matrix inputMask) {
        // encoderInput: [batch_size * seq_length, hidden_size]
        // inputMask: [batch_size, seq_length]

        // [batch_size * seq_length, hidden_size]
        attentionOutput = multiHeadSelfAttentionLayer.forward(encoderInput, inputMask);
        // [batch_size * seq_length, hidden_size]
        encoderOutput = feedForwardLayer.forward(attentionOutput);
        return encoderOutput;
    }

    public MultiHeadSelfAttentionLayer getMultiHeadSelfAttentionLayer() {
        return multiHeadSelfAttentionLayer;
    }

    public FeedForwardLayer getFeedForwardLayer() {
        return feedForwardLayer;
    }

    public Matrix getAttentionOutput() {
        return attentionOutput;
    }

    public Matrix getEncoderOutput() {
        return encoderOutput;
    }

    public static void main(String[] args) {
        // [batch_size * seq_length, hidden_size]
        Matrix encoderInput = new Matrix(4096, 64);
        // [batch_size, seq_length]
        Matrix inputMask = new Matrix(16, 256);

        FilePathConfig filePathConfig = new FilePathConfig("/file_path_config.properties");
        AlbertConfig albertConfig = new AlbertConfig("/albert_config.properties");

        TransformerEncoder transformerEncoder =
                new TransformerEncoder(filePathConfig, albertConfig);
        Matrix attentionOutput = transformerEncoder.forward(encoderInput, inputMask);
        System.out.println();
    }
}
