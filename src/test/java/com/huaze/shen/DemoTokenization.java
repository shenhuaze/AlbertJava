package com.huaze.shen;

public class DemoTokenization {
    public static void main(String[] args) {
        Tokenization tokenization = new Tokenization("/vocab.txt");
        String text = "[携程网]已出票:订单5180493941[上海航空☐fm9206☐福州长乐机场-上海虹桥机场t2☐12月8日9:15-12月8日10:30☐黄迪健,票号781-8228096741;汪振财,票号781-8228096742]请提前2小时至机场值机.退改点击http://t.ctrip.cn/fc94iux☐[安全提醒:发生航变时以任何理由要求客户转账、汇款均为欺诈行为.如遇陌生号码发送机票异常信息,请通过携程微信和app查询,谨防受骗]☐";
        AlbertInput albertInput = tokenization.tokenize(text);
        System.out.println(albertInput.getNormalizedTokensList());
        System.out.println(albertInput.getInputTokensList());
    }
}
