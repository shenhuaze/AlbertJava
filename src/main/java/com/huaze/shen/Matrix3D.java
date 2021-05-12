package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-27
 *
 * 三维矩阵
 */
public class Matrix3D {
    private double[][][] A;

    private int m, n, p;

    public Matrix3D (int m, int n, int p) {
        this.m = m;
        this.n = n;
        this.p = p;
        A = new double[m][n][p];
    }

    public Matrix3D (int m, int n, int p, double s) {
        this.m = m;
        this.n = n;
        this.p = p;
        A = new double[m][n][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    A[i][j][k] = s;
                }
            }
        }
    }

    public Matrix3D (double[][][] A) {
        m = A.length;
        n = A[0].length;
        p = A[0][0].length;
        for (int i = 0; i < m; i++) {
            if (A[i].length != n) {
                throw new IllegalArgumentException("All rows must have the same length.");
            }
            for (int j = 0; j < n; j++) {
                if (A[i][j].length != p) {
                    throw new IllegalArgumentException("All rows must have the same length.");
                }
            }
        }
        this.A = A;
    }

    public Matrix4D broadcastAxis1(int dim) {
        double[][][][] broadcastArray = new double[m][dim][n][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < dim; j++) {
                broadcastArray[i][j] = A[i];
            }
        }
        return new Matrix4D(broadcastArray);
    }

    public Matrix extractFirstAlongAxis1() {
        double[][] resultArray = new double[m][p];
        for (int i = 0; i < m; i++) {
            resultArray[i] = A[i][0];
        }
        return new Matrix(resultArray);
    }

    public Matrix3D (double[][][] A, int m, int n, int p) {
        this.A = A;
        this.m = m;
        this.n = n;
        this.p = p;
    }

    public double[][][] getArray() {
        return A;
    }

    public Matrix reshapeTo2D(int a, int b) {
        double[][] result = new double[a][b];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    int globalId = i * n * p + j * p + k;

                    int groupId1 = globalId / b;
                    int remainder1 = globalId % b;

                    result[groupId1][remainder1] = A[i][j][k];
                }
            }
        }
        return new Matrix(result);
    }

    public Matrix argmaxAxis2() {
        Matrix X = new Matrix(m, n);
        double[][] C = X.getArray();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int maxIndex = 0;
                double max = A[i][j][0];
                for (int k = 0; k < p; k++) {
                    if (A[i][j][k] > max) {
                        max = A[i][j][k];
                        maxIndex = k;
                    }
                }
                C[i][j] = maxIndex;
            }

        }
        return X;
    }

    public int getM() {
        return m;
    }

    public int getN() {
        return n;
    }

    public int getP() {
        return p;
    }
}
