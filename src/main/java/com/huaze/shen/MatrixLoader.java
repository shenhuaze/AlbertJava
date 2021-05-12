package com.huaze.shen;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * 从文件中加载模型权重到矩阵中
 */
public class MatrixLoader {
    public static Matrix loadTextFile(String matrixFile, int numRows, int numCols) {
        double[][] array = new double[numRows][numCols];
        try {
            InputStreamReader inputStreamReader = new InputStreamReader(MatrixLoader.class.getResourceAsStream(matrixFile));
            BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
            String line;
            line = bufferedReader.readLine();
            String[] lineSplit = line.split(" ");
            int numRows1 = Integer.parseInt(lineSplit[0]);
            int numCols1 = Integer.parseInt(lineSplit[1]);
            assert numRows == numRows1;
            assert numCols == numCols1;
            int row = 0;
            while ((line = bufferedReader.readLine()) != null) {
                lineSplit = line.split(" ");
                for (int i = 0; i < lineSplit.length; i++) {
                    array[row][i] = Double.parseDouble(lineSplit[i]);
                }
                row += 1;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new Matrix(array);
    }
}
