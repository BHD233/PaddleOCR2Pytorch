## OCR模型列表
**说明** ：**`.pth`模型下载链接：https://pan.baidu.com/s/1r1DELT8BlgxeOP2RqREJEg 提取码：6clx** 

PaddleOCR模型百度网盘链接：https://pan.baidu.com/s/1getAprT2l_JqwhjwML0g9g 提取码：lmv7 

- [一、文本检测模型](#文本检测模型)
- [二、文本识别模型](#文本识别模型)
    - [1. 中文识别模型](#中文识别模型)
    - [2. 英文识别模型](#英文识别模型)
    - [3. 多语言识别模型](#多语言识别模型)
- [三、文本方向分类模型](#文本方向分类模型)

PaddleOCR提供的可下载模型包括`推理模型`、`训练模型`、`预训练模型`、`slim模型`，<u>**PyTorch模型只支持非slim模型**</u>，模型区别说明如下：

|模型类型|模型格式|简介|
|--- | --- | --- |
|推理模型|inference.pdmodel、inference.pdiparams|用于python预测引擎推理，[详情](./inference.md)|
|训练模型、预训练模型|\*.pdparams、\*.pdopt、\*.states |训练过程中保存的模型的参数、优化器状态和训练中间信息，多用于模型指标评估和恢复训练|

<a name="文本检测模型"></a>
### 一、文本检测模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|ch_ppocr_mobile_v2.0_det|原始超轻量模型，支持中英文、多语种文本检测|3M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|
|ch_ppocr_server_v2.0_det|通用模型，支持中英文、多语种文本检测，比超轻量模型更大，但效果更好|47M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)|


<a name="文本识别模型"></a>
### 二、文本识别模型

<a name="中文识别模型"></a>
#### 1. 中文识别模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|ch_ppocr_mobile_v2.0_rec|原始超轻量模型，支持中英文、数字识别|3.71M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
|ch_ppocr_server_v2.0_rec|通用模型，支持中英文、数字识别|94.8M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |

**说明：** `训练模型`是基于预训练模型在真实数据与竖排合成文本数据上finetune得到的模型，在真实应用场景中有着更好的表现，`预训练模型`则是直接基于全量真实数据与合成数据训练得到，更适合用于在自己的数据集上finetune。

<a name="英文识别模型"></a>
#### 2. 英文识别模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|en_number_mobile_v2.0_rec|原始超轻量模型，支持英文、数字识别|2.56M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/en_number_mobile_v2.0_rec_train.tar) |

<a name="多语言识别模型"></a>
#### 3. 多语言识别模型（更多语言持续更新中...）

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
| french_mobile_v2.0_rec |法文识别|2.65M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/french_mobile_v2.0_rec_train.tar) |
| german_mobile_v2.0_rec |德文识别|2.65M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/german_mobile_v2.0_rec_train.tar) |
| korean_mobile_v2.0_rec |韩文识别|3.9M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/korean_mobile_v2.0_rec_train.tar) |
| japan_mobile_v2.0_rec |日文识别|4.23M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/japan_mobile_v2.0_rec_train.tar) |
| it_mobile_v2.0_rec |意大利文识别|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/it_mobile_v2.0_rec_train.tar) |
| es_mobile_v2.0_rec |西班牙文识别|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/es_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/es_mobile_v2.0_rec_train.tar) |
| pt_mobile_v2.0_rec |葡萄牙文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pt_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/pt_mobile_v2.0_rec_train.tar) |
| ru_mobile_v2.0_rec |俄罗斯文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ru_mobile_v2.0_rec_train.tar) |
| ar_mobile_v2.0_rec |阿拉伯文识别|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ar_mobile_v2.0_rec_train.tar) |
| hi_mobile_v2.0_rec |印地文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/hi_mobile_v2.0_rec_train.tar) |
| ch_tra_mobile_v2.0_rec |中文繁体识别|5.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ch_tra_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ch_tra_mobile_v2.0_rec_train.tar) |
| ug_mobile_v2.0_rec |维吾尔文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ug_mobile_v2.0_rec_train.tar) |
| fa_mobile_v2.0_rec |波斯文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/fa_mobile_v2.0_rec_train.tar) |
| ur_mobile_v2.0_rec |乌尔都文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ur_mobile_v2.0_rec_train.tar) |
| rs_latin_mobile_v2.0_rec |塞尔维亚文（latin）识别|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_latin_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_latin_mobile_v2.0_rec_train.tar) |
| oc_mobile_v2.0_rec |欧西坦文识别|2.53M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/oc_mobile_v2.0_rec_train.tar) |
| mr_mobile_v2.0_rec |马拉地文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/mr_mobile_v2.0_rec_train.tar) |
| ne_mobile_v2.0_rec |尼泊尔文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ne_mobile_v2.0_rec_train.tar) |
| rs_cyrillic_mobile_v2.0_rec |塞尔维亚文（cyrillic）识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_cyrillic_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/rs_cyrillic_mobile_v2.0_rec_train.tar) |
| bg_mobile_v2.0_rec |保加利亚文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/bg_mobile_v2.0_rec_train.tar) |
| uk_mobile_v2.0_rec |乌克兰文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/uk_mobile_v2.0_rec_train.tar) |
| be_mobile_v2.0_rec |白俄罗斯文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/be_mobile_v2.0_rec_train.tar) |
| te_mobile_v2.0_rec |泰卢固文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/te_mobile_v2.0_rec_train.tar) |
| kn_mobile_v2.0_rec |卡纳达文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/kn_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/kn_mobile_v2.0_rec_train.tar) |
| ta_mobile_v2.0_rec |泰米尔文识别|2.63M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/multilingual/ta_mobile_v2.0_rec_train.tar) |


<a name="文本方向分类模型"></a>
### 三、文本方向分类模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|ch_ppocr_mobile_v2.0_cls|原始模型|1.38M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |