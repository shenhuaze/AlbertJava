package com.huaze.shen;

import java.util.Properties;

/**
 * @author shenhuaze
 * @date 2021-01-27
 *
 * 文件路径相关配置
 */
public class FilePathConfig {
    // 存放权重文件的文件夹
    private String weightsDir;

    // embedding层相关权重文件
    private String wordEmbeddingsFile;
    private String tokenTypeEmbeddingsFile;
    private String positionEmbeddingsFile;
    private String embeddingLayerNormBetaFile;
    private String embeddingLayerNormGammaFile;
    private String embeddingHiddenMappingInKernelFile;
    private String embeddingHiddenMappingInBiasFile;

    // encoder层相关权重文件
    // self-attention
    private String attentionQueryKernelFile;
    private String attentionQueryBiasFile;
    private String attentionKeyKernelFile;
    private String attentionKeyBiasFile;
    private String attentionValueKernelFile;
    private String attentionValueBiasFile;
    private String attentionOutputKernelFile;
    private String attentionOutputBiasFile;
    private String attentionLayerNormBetaFile;
    private String attentionLayerNormGammaFile;
    // ffn
    private String ffnIntermediateKernelFile;
    private String ffnIntermediateBiasFile;
    private String ffnOutputKernelFile;
    private String ffnOutputBiasFile;
    private String ffnLayerNormBetaFile;
    private String ffnLayerNormGammaFile;

    // output层相关权重文件
    private String outputPoolerKernelFile;
    private String outputPoolerBiasFile;
    private String outputSequenceKernelFile;
    private String outputSequenceBiasFile;

    // vocab文件
    private String vocabFile;

    // ALBERT配置文件
    private String albertConfigFile;

    // tag文件
    private String tagFile;

    public FilePathConfig(String configFile) {
        Properties properties = new Properties();
        try {
            properties.load(AlbertConfig.class.getResourceAsStream(configFile));
            // 存放权重文件的文件夹
            this.weightsDir = properties.getProperty("weightsDir");

            // embedding层相关权重文件
            this.wordEmbeddingsFile = this.weightsDir + properties.getProperty("wordEmbeddingsFile");
            this.tokenTypeEmbeddingsFile = this.weightsDir + properties.getProperty("tokenTypeEmbeddingsFile");
            this.positionEmbeddingsFile = this.weightsDir + properties.getProperty("positionEmbeddingsFile");
            this.embeddingLayerNormBetaFile = this.weightsDir + properties.getProperty("embeddingLayerNormBetaFile");
            this.embeddingLayerNormGammaFile = this.weightsDir + properties.getProperty("embeddingLayerNormGammaFile");
            this.embeddingHiddenMappingInKernelFile = this.weightsDir + properties.getProperty("embeddingHiddenMappingInKernelFile");
            this.embeddingHiddenMappingInBiasFile = this.weightsDir + properties.getProperty("embeddingHiddenMappingInBiasFile");

            // encoder层相关权重文件
            // self-attention
            this.attentionQueryKernelFile = this.weightsDir + properties.getProperty("attentionQueryKernelFile");
            this.attentionQueryBiasFile = this.weightsDir + properties.getProperty("attentionQueryBiasFile");
            this.attentionKeyKernelFile = this.weightsDir + properties.getProperty("attentionKeyKernelFile");
            this.attentionKeyBiasFile = this.weightsDir + properties.getProperty("attentionKeyBiasFile");
            this.attentionValueKernelFile = this.weightsDir + properties.getProperty("attentionValueKernelFile");
            this.attentionValueBiasFile = this.weightsDir + properties.getProperty("attentionValueBiasFile");
            this.attentionOutputKernelFile = this.weightsDir + properties.getProperty("attentionOutputKernelFile");
            this.attentionOutputBiasFile = this.weightsDir + properties.getProperty("attentionOutputBiasFile");
            this.attentionLayerNormBetaFile = this.weightsDir + properties.getProperty("attentionLayerNormBetaFile");
            this.attentionLayerNormGammaFile = this.weightsDir + properties.getProperty("attentionLayerNormGammaFile");
            // ffn
            this.ffnIntermediateKernelFile = this.weightsDir + properties.getProperty("ffnIntermediateKernelFile");
            this.ffnIntermediateBiasFile = this.weightsDir + properties.getProperty("ffnIntermediateBiasFile");
            this.ffnOutputKernelFile = this.weightsDir + properties.getProperty("ffnOutputKernelFile");
            this.ffnOutputBiasFile = this.weightsDir + properties.getProperty("ffnOutputBiasFile");
            this.ffnLayerNormBetaFile = this.weightsDir + properties.getProperty("ffnLayerNormBetaFile");
            this.ffnLayerNormGammaFile = this.weightsDir + properties.getProperty("ffnLayerNormGammaFile");

            // output层相关权重文件
            this.outputPoolerKernelFile = this.weightsDir + properties.getProperty("outputPoolerKernelFile");
            this.outputPoolerBiasFile = this.weightsDir + properties.getProperty("outputPoolerBiasFile");
            this.outputSequenceKernelFile = this.weightsDir + properties.getProperty("outputSequenceKernelFile");
            this.outputSequenceBiasFile = this.weightsDir + properties.getProperty("outputSequenceBiasFile");

            // vocab文件
            this.vocabFile = properties.getProperty("vocab_file");

            // ALBERT配置文件
            this.albertConfigFile = properties.getProperty("albert_config_file");

            // tag文件
            this.tagFile = properties.getProperty("tag_file");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String getWeightsDir() {
        return weightsDir;
    }

    public String getWordEmbeddingsFile() {
        return wordEmbeddingsFile;
    }

    public String getTokenTypeEmbeddingsFile() {
        return tokenTypeEmbeddingsFile;
    }

    public String getPositionEmbeddingsFile() {
        return positionEmbeddingsFile;
    }

    public String getEmbeddingLayerNormBetaFile() {
        return embeddingLayerNormBetaFile;
    }

    public String getEmbeddingLayerNormGammaFile() {
        return embeddingLayerNormGammaFile;
    }

    public String getEmbeddingHiddenMappingInKernelFile() {
        return embeddingHiddenMappingInKernelFile;
    }

    public String getEmbeddingHiddenMappingInBiasFile() {
        return embeddingHiddenMappingInBiasFile;
    }

    public String getAttentionQueryKernelFile() {
        return attentionQueryKernelFile;
    }

    public String getAttentionQueryBiasFile() {
        return attentionQueryBiasFile;
    }

    public String getAttentionKeyKernelFile() {
        return attentionKeyKernelFile;
    }

    public String getAttentionKeyBiasFile() {
        return attentionKeyBiasFile;
    }

    public String getAttentionValueKernelFile() {
        return attentionValueKernelFile;
    }

    public String getAttentionValueBiasFile() {
        return attentionValueBiasFile;
    }

    public String getAttentionOutputKernelFile() {
        return attentionOutputKernelFile;
    }

    public String getAttentionOutputBiasFile() {
        return attentionOutputBiasFile;
    }

    public String getAttentionLayerNormBetaFile() {
        return attentionLayerNormBetaFile;
    }

    public String getAttentionLayerNormGammaFile() {
        return attentionLayerNormGammaFile;
    }

    public String getFfnIntermediateKernelFile() {
        return ffnIntermediateKernelFile;
    }

    public String getFfnIntermediateBiasFile() {
        return ffnIntermediateBiasFile;
    }

    public String getFfnOutputKernelFile() {
        return ffnOutputKernelFile;
    }

    public String getFfnOutputBiasFile() {
        return ffnOutputBiasFile;
    }

    public String getFfnLayerNormBetaFile() {
        return ffnLayerNormBetaFile;
    }

    public String getFfnLayerNormGammaFile() {
        return ffnLayerNormGammaFile;
    }

    public String getOutputPoolerKernelFile() {
        return outputPoolerKernelFile;
    }

    public String getOutputPoolerBiasFile() {
        return outputPoolerBiasFile;
    }

    public String getOutputSequenceKernelFile() {
        return outputSequenceKernelFile;
    }

    public String getOutputSequenceBiasFile() {
        return outputSequenceBiasFile;
    }

    public String getVocabFile() {
        return vocabFile;
    }

    public String getAlbertConfigFile() {
        return albertConfigFile;
    }

    public String getTagFile() {
        return tagFile;
    }
}
