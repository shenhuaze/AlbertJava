package com.huaze.shen;

import java.util.List;

public class DemoEntityRecognizer {
    public static void main(String[] args) {
        EntityRecognizer entityRecognizer = new EntityRecognizer();
        //String text = "[携程网]已出票:订单5180493941[上海航空☐fm9206☐福州长乐机场-上海虹桥机场t2☐12月8日9:15-12月8日10:30☐黄迪健,票号781-8228096741;汪振财,票号781-8228096742]请提前2小时至机场值机.退改点击http://t.ctrip.cn/fc94iux☐[安全提醒:发生航变时以任何理由要求客户转账、汇款均为欺诈行为.如遇陌生号码发送机票异常信息,请通过携程微信和app查询,谨防受骗]☐";
        String text = "[携程网]已出票:订单5180493941[上海航空 fm9206 福州长乐机场-上海虹桥机场t2 12月8日9:15-12月8日10:30 黄迪健,票号781-8228096741;汪振财,票号781-8228096742]请提前2小时至机场值机.退改点击http://t.ctrip.cn/fc94iux [安全提醒:发生航变时以任何理由要求客户转账、汇款均为欺诈行为.如遇陌生号码发送机票异常信息,请通过携程微信和app查询,谨防受骗] ";
        //String text = "[吉祥航空]尊敬的林大伟旅客,您好!感谢您乘坐吉祥航空的#日期# #航班号#由贵阳(龙洞堡)到-丽江(三义)航班,这个世界太小,小到我们有幸陪伴您空中之旅,这个世界太大,大到我们珍惜与您的每次相遇,因为您让我们变得更好.让我们发现您的期待,了解您的需求,欢迎您参加我们的\"小吉知多少\"有奖问卷活动.此问卷仅用于公司内部调研,您的资料我们将予以保密.问卷内容请点击:=HYPERLINK(\"HTTPS://MMBIZURL.CN/S/ETOWQM7UC, 此链接在航班后七天以内有效(包括航班当日.回TD退订)";
        List<Entity> entities = entityRecognizer.recognize(text);
        System.out.println(text);
        for (Entity entity: entities) {
            System.out.println(entity);
        }
    }
}
