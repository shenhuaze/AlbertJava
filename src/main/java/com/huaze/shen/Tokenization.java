package com.huaze.shen;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * @author shenhuaze
 * @date 2021-02-02
 *
 * 将原始输入文本转为ALBERT的输入AlbertInput(包含inputIds, segmentIds, positionIds, inputMask)
 */
public class Tokenization {
    private int seqLength = 256;
    private Pattern numberPattern = Pattern.compile("^[0-9]+$");
    private Pattern letterPattern = Pattern.compile("^[a-z]+$");
    private Pattern numberLetterPattern = Pattern.compile("^[0-9a-z]+$");
    private Map<String, Integer> token2id;
    private Map<Integer, String> id2token;

    public Tokenization(String vocabFile) {
        loadVocab(vocabFile);
    }

    public void loadVocab(String vocabFile) {
        token2id = new HashMap<>();
        id2token = new HashMap<>();
        try {
            InputStream inputStream = Tokenization.class.getResourceAsStream(vocabFile);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                if (token2id.containsKey(line)) {
                    continue;
                }
                int size = token2id.size();
                token2id.put(line, size);
                id2token.put(size, line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public AlbertInput tokenize(String text) {
        List<String> texts = new ArrayList<>();
        texts.add(text);
        return tokenize(texts);
    }

    public AlbertInput tokenize(List<String> textList) {
        List<List<String>> normalizedTokensList = new ArrayList<>();
        List<List<String>> inputTokensList = new ArrayList<>();
        for (String text: textList) {
            text = text.replace(" ", "☐");
            List<String> normalizedTokens = numberLetterNormalize(text);
            List<String> inputTokens = getInputTokens(normalizedTokens);
            normalizedTokensList.add(normalizedTokens);
            inputTokensList.add(inputTokens);
        }
        int size = inputTokensList.size();
        double[][] inputIds = new double[size][seqLength];
        double[][] segmentIds = new double[size][seqLength];
        double[][] positionIds = new double[size][seqLength];
        double[][] inputMask = new double[size][seqLength];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < seqLength; j++) {
                int actualLength = inputTokensList.get(i).size();
                if (j < actualLength) {
                    inputIds[i][j] = token2id.get(inputTokensList.get(i).get(j));
                    inputMask[i][j] = 1;
                } else {
                    inputIds[i][j] = 0;
                    inputMask[i][j] = 0;
                }
                segmentIds[i][j] = 0;
                positionIds[i][j] = j;
            }
        }
        return new AlbertInput(new Matrix(inputIds), new Matrix(segmentIds), new Matrix(positionIds),
                new Matrix(inputMask), normalizedTokensList, inputTokensList);
    }

    public List<String> getInputTokens(List<String> normalizedTokens) {
        List<String> inputTokens = new ArrayList<>();
        for (String token: normalizedTokens) {
            if (numberPattern.matcher(token).matches()) {
                token = "[NUMBER]";
            } else if (letterPattern.matcher(token).matches()) {
                token = "[LETTER]";
            } else if (numberLetterPattern.matcher(token).matches()) {
                token = "[MIX]";
            }
            if (!token2id.containsKey(token)) {
                token = "[UNK]";
            }
            inputTokens.add(token);
        }
        return inputTokens;
    }

    public List<String> numberLetterNormalize(String text) {
        text = text.toLowerCase();
        List<String> normalizedTokens = new ArrayList<>();
        normalizedTokens.add("[CLS]");
        List<String> numberLetters = new ArrayList<>();
        for (int i = 0; i < text.length(); i++) {
            String ch = text.substring(i, i + 1);
            if (numberLetterPattern.matcher(ch).find()) {
                numberLetters.add(ch);
            } else {
                if (numberLetters.size() > 0) {
                    normalizedTokens.add(String.join("", numberLetters));
                    numberLetters = new ArrayList<>();
                }
                normalizedTokens.add(ch);
            }
        }
        if (numberLetters.size() > 0) {
            normalizedTokens.add(String.join("", numberLetters));
        }
        if (normalizedTokens.size() > seqLength - 1) {
            normalizedTokens = normalizedTokens.subList(0, seqLength - 1);
        }
        normalizedTokens.add("[SEP]");
        return normalizedTokens;
    }

    public Map<String, Integer> getToken2id() {
        return token2id;
    }

    public Map<Integer, String> getId2token() {
        return id2token;
    }

    public static void main(String[] args) {
        Tokenization tokenization = new Tokenization("/vocab.txt");
        String text = "[携程网]已出票:订单5180493941[上海航空☐fm9206☐福州长乐机场-上海虹桥机场t2☐12月8日9:15-12月8日10:30☐黄迪健,票号781-8228096741;汪振财,票号781-8228096742]请提前2小时至机场值机.退改点击http://t.ctrip.cn/fc94iux☐[安全提醒:发生航变时以任何理由要求客户转账、汇款均为欺诈行为.如遇陌生号码发送机票异常信息,请通过携程微信和app查询,谨防受骗]☐";
        AlbertInput albertInput = tokenization.tokenize(text);
        System.out.println(albertInput.getNormalizedTokensList());
        System.out.println(albertInput.getInputTokensList());
    }
}
