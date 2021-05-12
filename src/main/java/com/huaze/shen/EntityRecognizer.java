package com.huaze.shen;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author shenhuaze
 * @date 2021-01-26
 *
 * 命名实体识别器
 */
public class EntityRecognizer {
    private Tokenization tokenization;
    private SequenceLabelingModel sequenceLabelingModel;
    private Map<String, Integer> tag2id;
    private Map<Integer, String> id2tag;

    public EntityRecognizer() {
        FilePathConfig filePathConfig = new FilePathConfig("/file_path_config.properties");
        AlbertConfig albertConfig = new AlbertConfig(filePathConfig.getAlbertConfigFile());
        this.tokenization = new Tokenization(filePathConfig.getVocabFile());
        this.sequenceLabelingModel = new SequenceLabelingModel(filePathConfig, albertConfig);
        loadTagFile(filePathConfig.getTagFile());
    }

    public void loadTagFile(String tagFile) {
        tag2id = new HashMap<>();
        id2tag = new HashMap<>();
        try {
            InputStream inputStream = EntityRecognizer.class.getResourceAsStream(tagFile);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                if (tag2id.containsKey(line)) {
                    continue;
                }
                int size = tag2id.size();
                tag2id.put(line, size);
                id2tag.put(size, line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public List<Entity> recognize(String text) {
        AlbertInput albertInput = tokenization.tokenize(text);
        Matrix predictTagIds = sequenceLabelingModel.forward(albertInput);
        List<List<String>> predictTagsList = convertIdToRealLabels(predictTagIds, id2tag);
        List<String> predictTags = predictTagsList.get(0);
        List<Entity> entities = combineTagToEntity(albertInput.getNormalizedTokensList().get(0), predictTags);
        recoverRealIndex(albertInput.getNormalizedTokensList().get(0), entities);
        //System.out.println(albertInput.getNormalizedTokensList());
        //System.out.println(albertInput.getInputTokensList());
        return entities;
    }

    public void recoverRealIndex(List<String> normalizedTokens, List<Entity> entities) {
        List<Integer> realStartIndexList = new ArrayList<>();
        int accumIndex = 0;
        for (int i = 0; i < normalizedTokens.size(); i++) {
            if (i == 0) {
                realStartIndexList.add(0);
            } else {
                realStartIndexList.add(accumIndex);
                accumIndex += normalizedTokens.get(i).length();
            }
        }
        for (Entity entity: entities) {
            int realStartIndex = realStartIndexList.get(entity.getNormalizedStart());
            int realEndIndex = realStartIndex + entity.getValue().length();
            entity.setStart(realStartIndex);
            entity.setEnd(realEndIndex);
        }
    }

    public List<Entity> combineTagToEntity(List<String> tokens, List<String> tags) {
        //System.out.println(tokens);
        //System.out.println(tags);
        List<Entity> entities = new ArrayList<>();
        int size = tokens.size();
        List<String> part = new ArrayList<>();
        int start = 0;
        String type = "";
        String currTag = "";
        String lastTag = "";
        for (int i = 0; i < size; i++) {
            if ("[PAD]".equals(tokens.get(i))) {
                break;
            }
            String token = tokens.get(i);
            currTag = tags.get(i);
            if (currTag.startsWith("B")) {
                if (part.size() > 0) {
                    String entityValue = String.join("", part);
                    if (entityValue.contains("☐")) {
                        entityValue = entityValue.replace("☐", " ");
                    }
                    entities.add(new Entity(type, entityValue, start, i));
                    part = new ArrayList<>();
                }
                start = i;
                part.add(token);
                type = currTag.substring(2);
            } else if (currTag.startsWith("I")) {
                if (part.size() > 0) {
                    part.add(token);
                }
            } else {
                if (part.size() > 0) {
                    String entityValue = String.join("", part);
                    if (entityValue.contains("☐")) {
                        entityValue = entityValue.replace("☐", " ");
                    }
                    entities.add(new Entity(type, entityValue, start, i));
                    part = new ArrayList<>();
                }
            }
            lastTag = currTag;
        }
        return entities;
    }

    public List<List<String>> convertIdToRealLabels(Matrix matrix, Map<Integer, String> id2label) {
        List<List<String>> results = new ArrayList<>();
        int m = matrix.getRowDimension();
        int n = matrix.getColumnDimension();
        for (int i = 0; i < m; i++) {
            List<String> result = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                result.add(id2label.get((int) matrix.get(i, j)));
            }
            results.add(result);
        }
        return results;
    }

    public static void main(String[] args) {
        EntityRecognizer entityRecognizer = new EntityRecognizer();
        String text = "[携程网]已出票:订单5180493941[上海航空☐fm9206☐福州长乐机场-上海虹桥机场t2☐12月8日9:15-12月8日10:30☐黄迪健,票号781-8228096741;汪振财,票号781-8228096742]请提前2小时至机场值机.退改点击http://t.ctrip.cn/fc94iux☐[安全提醒:发生航变时以任何理由要求客户转账、汇款均为欺诈行为.如遇陌生号码发送机票异常信息,请通过携程微信和app查询,谨防受骗]☐";
        //String text = "[吉祥航空]尊敬的林大伟旅客,您好!感谢您乘坐吉祥航空的#日期# #航班号#由贵阳(龙洞堡)到-丽江(三义)航班,这个世界太小,小到我们有幸陪伴您空中之旅,这个世界太大,大到我们珍惜与您的每次相遇,因为您让我们变得更好.让我们发现您的期待,了解您的需求,欢迎您参加我们的\"小吉知多少\"有奖问卷活动.此问卷仅用于公司内部调研,您的资料我们将予以保密.问卷内容请点击:=HYPERLINK(\"HTTPS://MMBIZURL.CN/S/ETOWQM7UC, 此链接在航班后七天以内有效(包括航班当日.回TD退订)";
        System.out.println(text);
        List<Entity> entities = entityRecognizer.recognize(text);
        for (Entity entity: entities) {
            System.out.println(entity);
        }
    }
}
