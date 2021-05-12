package com.huaze.shen;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * @author shenhuaze
 * @date 2021-02-01
 *
 * 验证FeedForwardLayer
 */
public class DemoVerifyFeedForwardLayer {
    public Matrix readInputs(String file) throws Exception {
        InputStream inputStream = DemoVerifyFeedForwardLayer.class.getResourceAsStream(file);
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
        String line;
        line = bufferedReader.readLine();
        String[] lineSplit = line.split(" ");
        int numRows = Integer.parseInt(lineSplit[0]);
        int numCols = Integer.parseInt(lineSplit[1]);
        int row = 0;
        double[][] array = new double[numRows][numCols];
        while ((line = bufferedReader.readLine()) != null) {
            lineSplit = line.split(" ");
            for (int i = 0; i < lineSplit.length; i++) {
                array[row][i] = Integer.parseInt(lineSplit[i]);
            }
            row += 1;
        }
        bufferedReader.close();
        return new Matrix(array);
    }

    public void writeOutputs(String file, Matrix output) throws Exception {
        System.out.println("write " + file);
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file));
        int m = output.getRowDimension();
        int n = output.getColumnDimension();
        bufferedWriter.write(String.format("%d %d\n", m, n));
        double[][] outputArray = output.getArray();
        for (int i = 0; i < m; i++) {
            List<String> lineData = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                lineData.add(String.format("%.16f", outputArray[i][j]));
            }
            //System.out.println(String.join(" ", lineData));
            bufferedWriter.write(String.join(" ", lineData) + "\n");
        }
        bufferedWriter.close();
    }

    public void demo() throws Exception {
        // [batch_size, seq_length]

        Matrix inputIds = readInputs("/inputs/input_ids.txt");
        Matrix segmentIds = readInputs("/inputs/segment_ids.txt");
        Matrix positionIds = readInputs("/inputs/input_position_ids.txt");
        Matrix inputMask = readInputs("/inputs/input_mask.txt");

        FilePathConfig filePathConfig = new FilePathConfig("/file_path_config.properties");
        AlbertConfig albertConfig = new AlbertConfig("/albert_config.properties");

        EmbeddingLayer embeddingLayer = new EmbeddingLayer(filePathConfig, albertConfig);
        MultiHeadSelfAttentionLayer multiHeadSelfAttentionLayer = new MultiHeadSelfAttentionLayer(filePathConfig, albertConfig);
        FeedForwardLayer feedForwardLayer = new FeedForwardLayer(filePathConfig, albertConfig);

        Matrix mappedEmbeddingsOutput = embeddingLayer.forward(inputIds, segmentIds, positionIds);
        Matrix attentionOutput = multiHeadSelfAttentionLayer.forward(mappedEmbeddingsOutput, inputMask);
        Matrix ffnOutput = feedForwardLayer.forward(attentionOutput);

        Matrix ffnIntermediate = feedForwardLayer.getFfnIntermediate();
        Matrix ffnOutputBeforeLayerNorm = feedForwardLayer.getFfnOutputBeforeLayerNorm();

        String outputDir = "src/test/resources/outputs/encoder_0/";
        File outputDirFile = new File(outputDir);
        if (!outputDirFile.exists()) {
            boolean create = outputDirFile.mkdirs();
        }
        writeOutputs(outputDir + "ffn_intermediate.txt", ffnIntermediate);
        writeOutputs(outputDir + "ffn_output_before_layer_norm.txt", ffnOutputBeforeLayerNorm);
        writeOutputs(outputDir + "ffn_output.txt", ffnOutput);
    }

    public static void main(String[] args) throws Exception {
        new DemoVerifyFeedForwardLayer().demo();
    }
}
