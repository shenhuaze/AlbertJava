package com.huaze.shen;

public class Entity {
    private String type;
    private String value;
    private int start;
    private int end;
    private int normalizedStart;
    private int normalizedEnd;

    public Entity(String type, String value, int normalizedStart, int normalizedEnd) {
        this.type = type;
        this.value = value;
        this.normalizedStart = normalizedStart;
        this.normalizedEnd = normalizedEnd;
    }

    public Entity(String type, String value, int start, int end, int normalizedStart, int normalizedEnd) {
        this.type = type;
        this.value = value;
        this.start = start;
        this.end = end;
        this.normalizedStart = normalizedStart;
        this.normalizedEnd = normalizedEnd;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public int getStart() {
        return start;
    }

    public void setStart(int start) {
        this.start = start;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    public int getNormalizedStart() {
        return normalizedStart;
    }

    public void setNormalizedStart(int normalizedStart) {
        this.normalizedStart = normalizedStart;
    }

    public int getNormalizedEnd() {
        return normalizedEnd;
    }

    public void setNormalizedEnd(int normalizedEnd) {
        this.normalizedEnd = normalizedEnd;
    }

    @Override
    public String toString() {
        return "Entity{" +
                "type='" + type + '\'' +
                ", value='" + value + '\'' +
                ", start=" + start +
                ", end=" + end +
                ", normalizedStart=" + normalizedStart +
                ", normalizedEnd=" + normalizedEnd +
                '}';
    }
}
