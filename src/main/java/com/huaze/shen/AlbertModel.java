package com.huaze.shen;

import java.util.ArrayList;
import java.util.List;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * ALBERT模型
 */
public class AlbertModel {
    private EmbeddingLayer embeddingLayer;
    private List<TransformerEncoder> transformerEncoders;
    private DenseLayer poolerLayer;
    private int numHiddenLayers;
    private int batchSize;
    private int seqLength;
    private int hiddenSize;
    private AlbertOutput albertOutput;

    public AlbertModel(FilePathConfig filePathConfig, AlbertConfig albertConfig) {
        this.numHiddenLayers = albertConfig.getNumHiddenLayers();
        this.batchSize = albertConfig.getBatchSize();
        this.seqLength = albertConfig.getSeqLength();
        this.hiddenSize = albertConfig.getHiddenSize();
        this.embeddingLayer = new EmbeddingLayer(filePathConfig, albertConfig);
        this.transformerEncoders = new ArrayList<>();
        for (int i = 0; i < numHiddenLayers; i++) {
            this.transformerEncoders.add(new TransformerEncoder(filePathConfig, albertConfig));
        }
        this.poolerLayer = new DenseLayer(filePathConfig.getOutputPoolerKernelFile(),
                filePathConfig.getOutputPoolerBiasFile(), albertConfig.getHiddenSize(), albertConfig.getHiddenSize(),
                "tanh");
    }

    public AlbertOutput forward(AlbertInput albertInput) {
        Matrix inputIds = albertInput.getInputIds();
        Matrix segmentIds = albertInput.getSegmentIds();
        Matrix positionIds = albertInput.getPositionIds();
        Matrix inputMask = albertInput.getInputMask();

        Matrix embeddingOutput = embeddingLayer.forward(inputIds, segmentIds, positionIds);
        Matrix encoderOutput = embeddingOutput.copy();
        List<Matrix> allEncoderOutputs = new ArrayList<>();
        for (int i = 0; i < this.numHiddenLayers; i++) {
            encoderOutput = transformerEncoders.get(i).forward(encoderOutput, inputMask);
            allEncoderOutputs.add(encoderOutput.copy());
        }
        // [batch_size, seq_length, hidden_size]
        Matrix3D sequenceOutput = encoderOutput.reshapeTo3D(this.batchSize, this.seqLength, this.hiddenSize);
        // [batch_size, hidden_size]
        Matrix firstTokenOutput = sequenceOutput.extractFirstAlongAxis1();
        // [batch_size, hidden_size]
        Matrix poolerOutput = this.poolerLayer.forward(firstTokenOutput);
        this.albertOutput = new AlbertOutput(embeddingOutput, allEncoderOutputs, sequenceOutput, poolerOutput);
        return this.albertOutput;
    }

    public static void main(String[] args) {
        // [batch_size, seq_length]
        Matrix inputIds = new Matrix(16, 256);
        Matrix segmentIds = new Matrix(16, 256);
        Matrix positionIds = new Matrix(16, 256);
        Matrix inputMask = new Matrix(16, 256);

        FilePathConfig filePathConfig = new FilePathConfig("/file_path_config.properties");
        AlbertConfig albertConfig = new AlbertConfig("/albert_config.properties");

        AlbertModel albertModel = new AlbertModel(filePathConfig, albertConfig);
        AlbertInput albertInput = new AlbertInput(inputIds, segmentIds, positionIds, inputMask);
        AlbertOutput albertOutput = albertModel.forward(albertInput);
        System.out.println();
    }
}
