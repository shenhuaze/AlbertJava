package com.huaze.shen;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * @author shenhuaze
 * @date 2021-01-30
 *
 * 验证AlbertModel
 */
public class DemoVerifyAlbertModel {
    public Matrix readInputs(String file) throws Exception {
        InputStream inputStream = DemoVerifyAlbertModel.class.getResourceAsStream(file);
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

        AlbertModel albertModel = new AlbertModel(filePathConfig, albertConfig);
        AlbertInput albertInput = new AlbertInput(inputIds, segmentIds, positionIds, inputMask);
        AlbertOutput albertOutput = albertModel.forward(albertInput);
        Matrix embeddingOutput = albertOutput.getEmbeddingOutput();
        List<Matrix> allEncoderOutputs = albertOutput.getAllEncoderOutputs();
        Matrix3D sequenceOutput3D = albertOutput.getSequenceOutput();
        Matrix sequenceOutput =
                sequenceOutput3D.reshapeTo2D(sequenceOutput3D.getM() * sequenceOutput3D.getN(), sequenceOutput3D.getP());
        Matrix pooledOutput = albertOutput.getPooledOutput();

        String outputDir = "src/test/resources/outputs/output_layer/";
        File outputDirFile = new File(outputDir);
        if (!outputDirFile.exists()) {
            boolean create = outputDirFile.mkdirs();
        }

        writeOutputs( outputDir + "albert_sequence_output.txt", sequenceOutput);
        writeOutputs(outputDir + "albert_pooled_output.txt", pooledOutput);
    }

    public static void main(String[] args) throws Exception {
        new DemoVerifyAlbertModel().demo();
    }
}
