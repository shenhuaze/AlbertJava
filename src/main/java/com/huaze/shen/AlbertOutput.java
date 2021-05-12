package com.huaze.shen;

import java.util.List;

/**
 * @author shenhuaze
 * @date 2021-01-28
 *
 * ALBERT的输出
 */
public class AlbertOutput {
    // [batch_size * seq_length, hidden_size]
    Matrix embeddingOutput;
    // List<[batch_size * seq_length, hidden_size]>
    List<Matrix> allEncoderOutputs;
    // [batch_size, seq_length, hidden_size]
    Matrix3D sequenceOutput;
    // [batch_size, hidden_size]
    Matrix pooledOutput;

    public AlbertOutput(Matrix embeddingOutput, List<Matrix> allEncoderOutputs, Matrix3D sequenceOutput, Matrix pooledOutput) {
        this.embeddingOutput = embeddingOutput;
        this.allEncoderOutputs = allEncoderOutputs;
        this.sequenceOutput = sequenceOutput;
        this.pooledOutput = pooledOutput;
    }

    public Matrix getEmbeddingOutput() {
        return embeddingOutput;
    }

    public List<Matrix> getAllEncoderOutputs() {
        return allEncoderOutputs;
    }

    public Matrix3D getSequenceOutput() {
        return sequenceOutput;
    }

    public Matrix getPooledOutput() {
        return pooledOutput;
    }
}
