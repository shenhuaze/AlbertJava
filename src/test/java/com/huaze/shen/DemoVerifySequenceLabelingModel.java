package com.huaze.shen;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * @author shenhuaze
 * @date 2021-02-01
 *
 * 验证SequenceLabelingModel
 */
public class DemoVerifySequenceLabelingModel {
    public Matrix readInputs(String file) throws Exception {
        InputStream inputStream = DemoVerifySequenceLabelingModel.class.getResourceAsStream(file);
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

        SequenceLabelingModel sequenceLabelingModel = new SequenceLabelingModel(filePathConfig, albertConfig);

        AlbertInput albertInput = new AlbertInput(inputIds, segmentIds, positionIds, inputMask);

        Matrix predictions = sequenceLabelingModel.forward(albertInput);
        Matrix3D probabilities3D = sequenceLabelingModel.getProbabilities();
        Matrix probabilities = probabilities3D.reshapeTo2D(probabilities3D.getM() * probabilities3D.getN(), probabilities3D.getP());

        String outputDir = "src/test/resources/outputs/output_layer/";
        File outputDirFile = new File(outputDir);
        if (!outputDirFile.exists()) {
            boolean create = outputDirFile.mkdirs();
        }

        writeOutputs( outputDir + "sequence_predictions.txt", predictions);
        writeOutputs(outputDir + "sequence_probabilities.txt", probabilities);
    }

    public static void main(String[] args) throws Exception {
        new DemoVerifySequenceLabelingModel().demo();
    }
}
