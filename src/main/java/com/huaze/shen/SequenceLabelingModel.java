package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * 序列标注模型
 */
public class SequenceLabelingModel {
    private int batchSize;
    private int seqLength;
    private int hiddenSize;
    private int numTags;
    private AlbertModel albertModel;
    private DenseLayer sequenceOutputLayer;

    private Matrix3D probabilities;
    private Matrix predictions;

    public SequenceLabelingModel(FilePathConfig filePathConfig, AlbertConfig albertConfig) {
        this.batchSize = albertConfig.getBatchSize();
        this.seqLength = albertConfig.getSeqLength();
        this.hiddenSize = albertConfig.getHiddenSize();
        this.numTags = albertConfig.getNumTags();
        this.albertModel = new AlbertModel(filePathConfig, albertConfig);
        this.sequenceOutputLayer = new DenseLayer(filePathConfig.getOutputSequenceKernelFile(),
                filePathConfig.getOutputSequenceBiasFile(), albertConfig.getHiddenSize(), albertConfig.getNumTags(),
                "linear");
    }

    public Matrix forward(AlbertInput albertInput) {
        AlbertOutput albertOutput = this.albertModel.forward(albertInput);
        Matrix3D albertSequenceOutput3D = albertOutput.getSequenceOutput();
        Matrix albertSequenceOutput = albertSequenceOutput3D.reshapeTo2D(batchSize * seqLength, hiddenSize);
        Matrix logits2D = sequenceOutputLayer.forward(albertSequenceOutput);
        Matrix3D logits3D = logits2D.reshapeTo3D(batchSize, seqLength, numTags);
        predictions = logits3D.argmaxAxis2();
        Matrix probabilities2D = logits2D.softmaxAxis1();
        probabilities = probabilities2D.reshapeTo3D(batchSize, seqLength, numTags);
        return predictions;
    }

    public Matrix3D getProbabilities() {
        return probabilities;
    }

    public Matrix getPredictions() {
        return predictions;
    }

    public static void main(String[] args) {
        // [batch_size, seq_length]
        Matrix inputIds = new Matrix(1, 256);
        Matrix segmentIds = new Matrix(1, 256);
        Matrix positionIds = new Matrix(1, 256);
        Matrix inputMask = new Matrix(1, 256);

        FilePathConfig filePathConfig = new FilePathConfig("/file_path_config.properties");
        AlbertConfig albertConfig = new AlbertConfig("/albert_config.properties");

        SequenceLabelingModel sequenceLabelingModel = new SequenceLabelingModel(filePathConfig, albertConfig);
        AlbertInput albertInput = new AlbertInput(inputIds, segmentIds, positionIds, inputMask);
        Matrix predictions = sequenceLabelingModel.forward(albertInput);
        System.out.println();
    }
}
