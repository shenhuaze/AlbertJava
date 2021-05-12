package com.huaze.shen;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * 词向量查询表
 */
public class LookupTable {
    private List<double[]> embeddings;
    private int embeddingSize;

    public LookupTable(String embeddingFile, int vocabSize, int embeddingSize) {
        Matrix matrix = MatrixLoader.loadTextFile(embeddingFile, vocabSize, embeddingSize);
        double[][] array = matrix.getArray();
        this.embeddings = new ArrayList<double[]>();
        this.embeddings.addAll(Arrays.asList(array));
        this.embeddingSize = embeddingSize;
    }

    public Matrix lookup(Matrix ids) {
        // ids: [batch_size, seq_length]
        int batchSize = ids.getRowDimension();
        int seqLength = ids.getColumnDimension();
        double[][] lookupArray = new double[batchSize * seqLength][this.embeddingSize];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLength; j++) {
                int index = i * batchSize + j;
                int vocabId = (int)ids.get(i, j);
                lookupArray[index] = this.embeddings.get(vocabId);
            }
        }
        // [batch_size * seq_length, embeddingSize]
        return new Matrix(lookupArray);
    }
}
