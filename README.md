# AlbertJava

基于纯Java实现ALBERT前向推理框架，不依赖任何第三方库

## 项目结构

```text
src
├── main
│   ├── java
│   │   ├── com.huaze.shen
│   │   │   ├── Activation: 激活函数
│   │   │   ├── AlbertConfig: Albert模型相关的配置参数
│   │   │   ├── AlbertInput: Albert模型的输入
│   │   │   ├── AlbertModel: Albert模型
│   │   │   ├── AlbertOutput: Albert模型的输出
│   │   │   ├── DenseLayer: 全连接层
│   │   │   ├── EmbeddingLayer: Embedding层
│   │   │   ├── Entity: 实体
│   │   │   ├── EntityRecognizer: 实体识别器
│   │   │   ├── FeedForwardLayer: transformer里的ffn层
│   │   │   ├── FilePathConfig: 文件路径相关的配置参数
│   │   │   ├── LayerNormalization: 层归一化
│   │   │   ├── LookupTable: lookup表
│   │   │   ├── Matrix: 二维矩阵相关方法
│   │   │   ├── Matrix3D: 三维矩阵相关方法
│   │   │   ├── Matrix4D: 四维矩阵相关方法
│   │   │   ├── MatrixLoader: 模型加载器
│   │   │   ├── MultiHeadSelfAttentionLayer: 多头自注意力层
│   │   │   ├── SequenceLabelingModel: 序列标注模型
│   │   │   ├── Tokenization: 分词器
│   │   │   └── TransformerEncoder: 一层TransformerEncoder
│   ├── resources: 资源文件
│   │   │   ├── weights: 模型权重文件
│   │   │   ├── albert_config.properties: albert模型相关的配置参数
│   │   │   ├── file_path_config.properties: 文件路径相关的配置参数
│   │   │   ├── tag.txt: 序列标注的tag集合
│   │   │   └── vocab.txt: 词表
├── test
│   ├── java
│   │   ├── com.huaze.shen
│   │   │   ├── DemoEntityRecognizer: 命名实体识别的demo
│   │   │   ├── DemoTokenization: 分词的demo
│   │   │   ├── DemoVerifyAlbertModel: 验证AlbertModel的参数
│   │   │   ├── DemoVerifyEmbeddingLayer: 验证EmbeddingLayer的参数
│   │   │   ├── DemoVerifyFeedForwardLayer: 验证FeedForwardLayer的参数
│   │   │   ├── DemoVerifyMultiHeadSelfAttentionLayer: 验证MultiHeadSelfAttentionLayer的参数
│   │   │   ├── DemoVerifySequenceLabelingModel: 验证SequenceLabelingModel的参数
│   │   │   └── DemoVerifyTransformerEncoder: 验证TransformerEncoder的参数
│   ├── resources: 资源文件
│   │   │   ├── inputs: 一个样本的输入
│   │   │   └── outputs: 一个样本经过AlbertModel的所有中间输出
│   │   │   │   ├── embedding_layer: embedding层的输出
│   │   │   │   ├── encoder_0: 第一层TransformerEncoder的输出
│   │   │   │   ├── encoder_1: 第二层TransformerEncoder的输出
│   │   │   │   ├── encoder_2: 第三层TransformerEncoder的输出
│   │   │   │   ├── encoder_3: 第四层TransformerEncoder的输出
│   │   │   │   ├── encoder_4: 第五层TransformerEncoder的输出
│   │   │   │   ├── encoder_5: 第六层TransformerEncoder的输出
│   │   │   │   └── output_layer: 输出层的输出