package com.huaze.shen;

import java.util.Properties;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * ALBERT相关配置参数
 */
public class AlbertConfig {
    private int vocabSize;
    private int embeddingSize;
    private int hiddenSize;
    private int numHiddenLayers;
    private int numHiddenGroups;
    private int numAttentionHeads;
    private int intermediateSize;
    private int innerGroupNum;
    private double downScaleFactor;
    private String hiddenAct;
    private double hiddenDropoutProb;
    private double attentionProbsDropoutProb;
    private int maxPositionEmbeddings;
    private int typeVocabSize;
    private double initializerRange;
    private int batchSize;
    private int seqLength;
    private int numTags;

    public AlbertConfig(int vocabSize,
                        int embeddingSize,
                        int hiddenSize,
                        int numHiddenLayers,
                        int numHiddenGroups,
                        int numAttentionHeads,
                        int intermediateSize,
                        int innerGroupNum,
                        double downScaleFactor,
                        String hiddenAct,
                        double hiddenDropoutProb,
                        double attentionProbsDropoutProb,
                        int maxPositionEmbeddings,
                        int typeVocabSize,
                        double initializerRange,
                        int batchSize,
                        int seqLength,
                        int numTags) {
        this.vocabSize = vocabSize;
        this.embeddingSize = embeddingSize;
        this.hiddenSize = hiddenSize;
        this.numHiddenLayers = numHiddenLayers;
        this.numHiddenGroups = numHiddenGroups;
        this.numAttentionHeads = numAttentionHeads;
        this.intermediateSize = intermediateSize;
        this.innerGroupNum = innerGroupNum;
        this.downScaleFactor = downScaleFactor;
        this.hiddenAct = hiddenAct;
        this.hiddenDropoutProb = hiddenDropoutProb;
        this.attentionProbsDropoutProb = attentionProbsDropoutProb;
        this.maxPositionEmbeddings = maxPositionEmbeddings;
        this.typeVocabSize = typeVocabSize;
        this.initializerRange = initializerRange;
        this.batchSize = batchSize;
        this.seqLength = seqLength;
        this.numTags = numTags;
    }

    public AlbertConfig(String configFile) {
        Properties properties = new Properties();
        try {
            properties.load(AlbertConfig.class.getResourceAsStream(configFile));
            this.vocabSize = Integer.parseInt(properties.getProperty("vocab_size"));
            this.embeddingSize = Integer.parseInt(properties.getProperty("embedding_size"));
            this.hiddenSize = Integer.parseInt(properties.getProperty("hidden_size"));
            this.numHiddenLayers = Integer.parseInt(properties.getProperty("num_hidden_layers"));
            this.numHiddenGroups = Integer.parseInt(properties.getProperty("num_hidden_groups"));
            this.numAttentionHeads = Integer.parseInt(properties.getProperty("num_attention_heads"));
            this.intermediateSize = Integer.parseInt(properties.getProperty("intermediate_size"));
            this.innerGroupNum = Integer.parseInt(properties.getProperty("inner_group_num"));
            this.downScaleFactor = Double.parseDouble(properties.getProperty("down_scale_factor"));
            this.hiddenAct = properties.getProperty("hidden_act");
            this.hiddenDropoutProb = Double.parseDouble(properties.getProperty("hidden_dropout_prob"));
            this.attentionProbsDropoutProb = Double.parseDouble(properties.getProperty("attention_probs_dropout_prob"));
            this.maxPositionEmbeddings = Integer.parseInt(properties.getProperty("max_position_embeddings"));
            this.typeVocabSize = Integer.parseInt(properties.getProperty("type_vocab_size"));
            this.initializerRange = Double.parseDouble(properties.getProperty("initializer_range"));
            this.batchSize = Integer.parseInt(properties.getProperty("batch_size"));
            this.seqLength = Integer.parseInt(properties.getProperty("seq_length"));
            this.numTags = Integer.parseInt(properties.getProperty("num_tags"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public int getEmbeddingSize() {
        return embeddingSize;
    }

    public void setEmbeddingSize(int embeddingSize) {
        this.embeddingSize = embeddingSize;
    }

    public int getHiddenSize() {
        return hiddenSize;
    }

    public void setHiddenSize(int hiddenSize) {
        this.hiddenSize = hiddenSize;
    }

    public int getNumHiddenLayers() {
        return numHiddenLayers;
    }

    public void setNumHiddenLayers(int numHiddenLayers) {
        this.numHiddenLayers = numHiddenLayers;
    }

    public int getNumHiddenGroups() {
        return numHiddenGroups;
    }

    public void setNumHiddenGroups(int numHiddenGroups) {
        this.numHiddenGroups = numHiddenGroups;
    }

    public int getNumAttentionHeads() {
        return numAttentionHeads;
    }

    public void setNumAttentionHeads(int numAttentionHeads) {
        this.numAttentionHeads = numAttentionHeads;
    }

    public int getIntermediateSize() {
        return intermediateSize;
    }

    public void setIntermediateSize(int intermediateSize) {
        this.intermediateSize = intermediateSize;
    }

    public int getInnerGroupNum() {
        return innerGroupNum;
    }

    public void setInnerGroupNum(int innerGroupNum) {
        this.innerGroupNum = innerGroupNum;
    }

    public double getDownScaleFactor() {
        return downScaleFactor;
    }

    public void setDownScaleFactor(double downScaleFactor) {
        this.downScaleFactor = downScaleFactor;
    }

    public String getHiddenAct() {
        return hiddenAct;
    }

    public void setHiddenAct(String hiddenAct) {
        this.hiddenAct = hiddenAct;
    }

    public double getHiddenDropoutProb() {
        return hiddenDropoutProb;
    }

    public void setHiddenDropoutProb(double hiddenDropoutProb) {
        this.hiddenDropoutProb = hiddenDropoutProb;
    }

    public double getAttentionProbsDropoutProb() {
        return attentionProbsDropoutProb;
    }

    public void setAttentionProbsDropoutProb(double attentionProbsDropoutProb) {
        this.attentionProbsDropoutProb = attentionProbsDropoutProb;
    }

    public int getMaxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }

    public void setMaxPositionEmbeddings(int maxPositionEmbeddings) {
        this.maxPositionEmbeddings = maxPositionEmbeddings;
    }

    public int getTypeVocabSize() {
        return typeVocabSize;
    }

    public void setTypeVocabSize(int typeVocabSize) {
        this.typeVocabSize = typeVocabSize;
    }

    public double getInitializerRange() {
        return initializerRange;
    }

    public void setInitializerRange(double initializerRange) {
        this.initializerRange = initializerRange;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getSeqLength() {
        return seqLength;
    }

    public void setSeqLength(int seqLength) {
        this.seqLength = seqLength;
    }

    public int getNumTags() {
        return numTags;
    }
}
