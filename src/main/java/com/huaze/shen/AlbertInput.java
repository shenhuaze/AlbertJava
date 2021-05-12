package com.huaze.shen;

import java.util.List;

/**
 * @author shenhuaze
 * @date 2021-02-02
 *
 * ALBERT的输入，包含四部分：inputIds, segmentIds, positionIds, inputMask
 */
public class AlbertInput {
    private Matrix inputIds;
    private Matrix segmentIds;
    private Matrix positionIds;
    private Matrix inputMask;

    private List<List<String>> normalizedTokensList;
    private List<List<String>> inputTokensList;

    public AlbertInput(Matrix inputIds, Matrix segmentIds, Matrix positionIds, Matrix inputMask) {
        this.inputIds = inputIds;
        this.segmentIds = segmentIds;
        this.positionIds = positionIds;
        this.inputMask = inputMask;
    }

    public AlbertInput(Matrix inputIds, Matrix segmentIds, Matrix positionIds, Matrix inputMask,
                       List<List<String>> normalizedTokensList, List<List<String>> inputTokensList) {
        this.inputIds = inputIds;
        this.segmentIds = segmentIds;
        this.positionIds = positionIds;
        this.inputMask = inputMask;
        this.normalizedTokensList = normalizedTokensList;
        this.inputTokensList = inputTokensList;
    }

    public Matrix getInputIds() {
        return inputIds;
    }

    public Matrix getSegmentIds() {
        return segmentIds;
    }

    public Matrix getPositionIds() {
        return positionIds;
    }

    public Matrix getInputMask() {
        return inputMask;
    }

    public List<List<String>> getNormalizedTokensList() {
        return normalizedTokensList;
    }

    public List<List<String>> getInputTokensList() {
        return inputTokensList;
    }
}
