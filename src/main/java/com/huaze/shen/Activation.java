package com.huaze.shen;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * 激活函数
 */
public class Activation {
    public static double gelu(double x) {
        return 0.5 * x * (1.0 + Math.tanh((Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))));
    }

    public static double relu(double x) {
        return Math.max(0, x);
    }

    public static double tanh(double x) {
        return Math.tanh(x);
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double[] softmax(double[] array) {
        int size = array.length;
        double[] expArray = new double[size];
        double sum = 0;
        for (int i = 0; i < size; i++) {
            double exp = Math.exp(array[i]);
            expArray[i] = exp;
            sum += exp;
        }
        double[] probs = new double[size];
        for (int i = 0; i < size; i++) {
            probs[i] = expArray[i] / sum;
        }
        return probs;
    }
}
