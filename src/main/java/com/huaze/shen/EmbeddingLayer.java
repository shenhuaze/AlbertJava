package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * ALBERT的embedding层
 */
public class EmbeddingLayer {
    private LookupTable wordEmbeddingTable;
    private LookupTable tokenTypeEmbeddingTable;
    private LookupTable positionEmbeddingTable;
    private LayerNormalization embeddingLayerNormalization;
    private DenseLayer embeddingHiddenMappingIn;

    private Matrix wordEmbeddings;
    private Matrix tokenTypeEmbeddings;
    private Matrix positionEmbeddings;
    private Matrix embeddingsOutputBeforeLayerNorm;
    private Matrix embeddingsOutput;
    private Matrix mappedEmbeddingsOutput;

    public EmbeddingLayer(FilePathConfig filePathConfig,
                          AlbertConfig albertConfig) {
        this.wordEmbeddingTable =
                new LookupTable(filePathConfig.getWordEmbeddingsFile(), albertConfig.getVocabSize(), albertConfig.getEmbeddingSize());
        this.tokenTypeEmbeddingTable =
                new LookupTable(filePathConfig.getTokenTypeEmbeddingsFile(), albertConfig.getTypeVocabSize(), albertConfig.getEmbeddingSize());
        this.positionEmbeddingTable =
                new LookupTable(filePathConfig.getPositionEmbeddingsFile(), albertConfig.getMaxPositionEmbeddings(), albertConfig.getEmbeddingSize());
        this.embeddingLayerNormalization =
                new LayerNormalization(filePathConfig.getEmbeddingLayerNormBetaFile(), filePathConfig.getEmbeddingLayerNormGammaFile(), albertConfig.getEmbeddingSize());
        this.embeddingHiddenMappingIn =
                new DenseLayer(filePathConfig.getEmbeddingHiddenMappingInKernelFile(), filePathConfig.getEmbeddingHiddenMappingInBiasFile(), albertConfig.getEmbeddingSize(), albertConfig.getHiddenSize(), "linear");
    }

    public Matrix forward(Matrix inputIds, Matrix tokenTypeIds, Matrix positionIds) {
        // [batch_size * seq_length, embedding_size]
        wordEmbeddings = wordEmbeddingTable.lookup(inputIds);
        tokenTypeEmbeddings = tokenTypeEmbeddingTable.lookup(tokenTypeIds);
        positionEmbeddings = positionEmbeddingTable.lookup(positionIds);
        // [batch_size * seq_length, embedding_size]
        embeddingsOutputBeforeLayerNorm = wordEmbeddings.plus(tokenTypeEmbeddings).plus(positionEmbeddings);
        // [batch_size * seq_length, embedding_size]
        embeddingsOutput = embeddingLayerNormalization.forward(embeddingsOutputBeforeLayerNorm);
        // [batch_size * seq_length, hidden_size]
        mappedEmbeddingsOutput = embeddingHiddenMappingIn.forward(embeddingsOutput);
        return mappedEmbeddingsOutput;
    }

    public static void main(String[] args) {
        // [batch_size, seq_length]
        Matrix inputIds = new Matrix(16, 256);
        Matrix segmentIds = new Matrix(16, 256);
        Matrix positionIds = new Matrix(16, 256);

        FilePathConfig filePathConfig = new FilePathConfig("/file_path_config.properties");
        AlbertConfig albertConfig = new AlbertConfig("/albert_config.properties");

        EmbeddingLayer embeddingLayer =
                new EmbeddingLayer(filePathConfig, albertConfig);
        Matrix embeddingOutput = embeddingLayer.forward(inputIds, segmentIds, positionIds);
        System.out.println();
    }

    public Matrix getWordEmbeddings() {
        return wordEmbeddings;
    }

    public Matrix getTokenTypeEmbeddings() {
        return tokenTypeEmbeddings;
    }

    public Matrix getPositionEmbeddings() {
        return positionEmbeddings;
    }

    public Matrix getEmbeddingsOutputBeforeLayerNorm() {
        return embeddingsOutputBeforeLayerNorm;
    }

    public Matrix getEmbeddingsOutput() {
        return embeddingsOutput;
    }

    public Matrix getMappedEmbeddingsOutput() {
        return mappedEmbeddingsOutput;
    }
}
