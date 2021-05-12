package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * TransformerEncoder的多头注意力层
 */
public class MultiHeadSelfAttentionLayer {
    private DenseLayer attentionQuery;
    private DenseLayer attentionKey;
    private DenseLayer attentionValue;
    private DenseLayer attentionOutputLayer;
    private LayerNormalization layerNormalization;
    private int batchSize;
    private int seqLength;
    private int hiddenSize;
    private int numAttentionHeads;
    private int sizePerHead;

    private Matrix queryLayer;
    private Matrix keyLayer;
    private Matrix valueLayer;
    private Matrix contextLayer2D;
    private Matrix attentionOutputBeforeLayerNorm;
    private Matrix attentionOutput;

    public MultiHeadSelfAttentionLayer(FilePathConfig filePathConfig,
                                       AlbertConfig albertConfig) {
        this.attentionQuery =
                new DenseLayer(filePathConfig.getAttentionQueryKernelFile(), filePathConfig.getAttentionQueryBiasFile(), albertConfig.getHiddenSize(), albertConfig.getHiddenSize(), "linear");
        this.attentionKey =
                new DenseLayer(filePathConfig.getAttentionKeyKernelFile(), filePathConfig.getAttentionKeyBiasFile(), albertConfig.getHiddenSize(), albertConfig.getHiddenSize(), "linear");
        this.attentionValue =
                new DenseLayer(filePathConfig.getAttentionValueKernelFile(), filePathConfig.getAttentionValueBiasFile(), albertConfig.getHiddenSize(), albertConfig.getHiddenSize(), "linear");
        this.attentionOutputLayer =
                new DenseLayer(filePathConfig.getAttentionOutputKernelFile(), filePathConfig.getAttentionOutputBiasFile(), albertConfig.getHiddenSize(), albertConfig.getHiddenSize(), "linear");
        this.layerNormalization =
                new LayerNormalization(filePathConfig.getAttentionLayerNormBetaFile(), filePathConfig.getAttentionLayerNormGammaFile(), albertConfig.getHiddenSize());
        this.batchSize = albertConfig.getBatchSize();
        this.seqLength = albertConfig.getSeqLength();
        this.hiddenSize = albertConfig.getHiddenSize();
        this.numAttentionHeads = albertConfig.getNumAttentionHeads();
        this.sizePerHead = this.hiddenSize / this.numAttentionHeads;
    }

    public Matrix forward(Matrix encoderInput, Matrix inputMask) {
        // self-attention block
        // encoderInput: [batch_size * seq_length, hidden_size]
        // inputMask: [batch_size, seq_length]
        Matrix3D attentionMask = createAttentionMaskFromInputMask(inputMask);
        queryLayer = attentionQuery.forward(encoderInput);
        keyLayer = attentionKey.forward(encoderInput);
        valueLayer = attentionValue.forward(encoderInput);
        // [batch_size, num_attention_heads, seq_length, size_per_head]
        Matrix4D transposedQueryLayer =
                transposeForScores(queryLayer, this.batchSize, this.seqLength, this.numAttentionHeads, this.sizePerHead);
        // [batch_size, num_attention_heads, seq_length, size_per_head]
        Matrix4D transposedKeyLayer =
                transposeForScores(keyLayer, this.batchSize, this.seqLength, this.numAttentionHeads, this.sizePerHead);
        // [batch_size, num_attention_heads, seq_length, seq_length]
        Matrix4D attentionScores = calculateAttentionScores(transposedQueryLayer, transposedKeyLayer);
        // [batch_size, num_attention_heads, seq_length, seq_length]
        attentionScores = combineAttentionScoresAndMask(attentionScores, attentionMask);
        // [batch_size, num_attention_heads, seq_length, seq_length]
        Matrix4D attentionProbs = attentionScores.softmax();
        // [batch_size, num_attention_heads, seq_length, size_per_head]
        Matrix4D transposedValueLayer =
                transposeForScores(valueLayer, this.batchSize, this.seqLength, this.numAttentionHeads, this.sizePerHead);
        // [batch_size, num_attention_heads, seq_length, size_per_head]
        Matrix4D contextLayer = calculateContextLayer(attentionProbs, transposedValueLayer);
        // [batch_size, seq_length, num_attention_heads, size_per_head]
        contextLayer = contextLayer.transpose_0213();
        // [batch_size * seq_length, hidden_size]
        contextLayer2D = contextLayer.reshapeTo2D(this.batchSize * this.seqLength, this.hiddenSize);

        // output block
        // [batch_size * seq_length, hidden_size]
        attentionOutputBeforeLayerNorm = this.attentionOutputLayer.forward(contextLayer2D);
        // [batch_size * seq_length, hidden_size]
        attentionOutput = layerNormalization.forward(attentionOutputBeforeLayerNorm.plus(encoderInput));
        return attentionOutput;
    }

    public Matrix4D calculateContextLayer(Matrix4D attentionProbs, Matrix4D transposedValueLayer) {
        double[][][][] attentionProbsArray = attentionProbs.getArray();
        double[][][][] transposedValueLayerArray = transposedValueLayer.getArray();
        double[][][][] resultArray =
                new double[this.batchSize][this.numAttentionHeads][this.seqLength][this.sizePerHead];
        for (int i = 0; i < this.batchSize; i++) {
            for (int j = 0; j < this.numAttentionHeads; j++) {
                double[][] perHeadValueArray = transposedValueLayerArray[i][j];
                Matrix perHeadValue = new Matrix(perHeadValueArray);
                for (int k = 0; k < this.seqLength; k++) {
                    double[] currentTokenAttentionProbs = attentionProbsArray[i][j][k];
                    Matrix currentTokenWeightedValue = perHeadValue.weightedSum(currentTokenAttentionProbs);
                    resultArray[i][j][k] = currentTokenWeightedValue.getArray()[0];
                }
            }
        }
        return new Matrix4D(resultArray);
    }

    public Matrix4D combineAttentionScoresAndMask(Matrix4D attentionScores, Matrix3D attentionMask) {
        Matrix4D attentionMaskBroadcast = attentionMask.broadcastAxis1(this.numAttentionHeads);
        attentionMaskBroadcast  = attentionMaskBroadcast.times(-1).plus(1).times(-1000);
        attentionScores = attentionScores.plus(attentionMaskBroadcast);
        return attentionScores;
    }

    public Matrix4D calculateAttentionScores(Matrix4D query, Matrix4D key) {
        double[][][][] queryArray = query.getArray();
        double[][][][] keyArray = key.getArray();
        double[][][][] attentionScoresArray =
                new double[this.batchSize][this.numAttentionHeads][this.seqLength][this.seqLength];
        for (int i = 0; i < this.batchSize; i++) {
            for (int j = 0; j < this.numAttentionHeads; j++) {
                double[][] perHeadQueryArray = queryArray[i][j];
                double[][] perHeadKeyArray = keyArray[i][j];
                Matrix perHeadQueryMatrix = new Matrix(perHeadQueryArray);
                Matrix perHeadKeyMatrix = new Matrix(perHeadKeyArray);
                attentionScoresArray[i][j] = perHeadQueryMatrix.times(perHeadKeyMatrix.transpose()).getArray();
            }
        }
        Matrix4D attentionScores = new Matrix4D(attentionScoresArray);
        attentionScores = attentionScores.times(1.0 / Math.sqrt(this.sizePerHead));
        return attentionScores;
    }

    public Matrix4D transposeForScores(Matrix matrix, int batchSize, int seqLength, int numAttentionHeads, int sizePerHead) {
        Matrix4D matrix4D = matrix.reshapeTo4D(batchSize, seqLength, numAttentionHeads, sizePerHead);
        matrix4D = matrix4D.transpose_0213();
        return matrix4D;
    }

    public Matrix3D createAttentionMaskFromInputMask(Matrix inputMask) {
        int batchSize = inputMask.getRowDimension();
        int seqLength = inputMask.getColumnDimension();
        double[][][] attentionMask = new double[batchSize][seqLength][seqLength];
        double[][] inputMaskArray = inputMask.getArray();
        for (int i = 0; i < batchSize; i++) {
            double[] currentMaskArray = inputMaskArray[i];
            for (int j = 0; j < seqLength; j++) {
                attentionMask[i][j] = currentMaskArray;
            }
        }
        // [batch_size, seq_length, seq_length]
        return new Matrix3D(attentionMask);
    }

    public Matrix getQueryLayer() {
        return queryLayer;
    }

    public Matrix getKeyLayer() {
        return keyLayer;
    }

    public Matrix getValueLayer() {
        return valueLayer;
    }

    public Matrix getContextLayer2D() {
        return contextLayer2D;
    }

    public Matrix getAttentionOutputBeforeLayerNorm() {
        return attentionOutputBeforeLayerNorm;
    }

    public Matrix getAttentionOutput() {
        return attentionOutput;
    }

    public static void main(String[] args) {
        // [batch_size * seq_length, hidden_size]
        Matrix encoderInput = new Matrix(4096, 64);
        Matrix inputMask = new Matrix(16, 256);

        FilePathConfig filePathConfig = new FilePathConfig("/file_path_config.properties");
        AlbertConfig albertConfig = new AlbertConfig("/albert_config.properties");

        MultiHeadSelfAttentionLayer multiHeadSelfAttentionLayer =
                new MultiHeadSelfAttentionLayer(filePathConfig, albertConfig);
        Matrix attentionOutput = multiHeadSelfAttentionLayer.forward(encoderInput, inputMask);
        System.out.println();
    }
}
