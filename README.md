# 统一模型开发框架

## DataEnhance

数据增强模型，使用中文roberta进行完形填空填空，将原始语句中的部分词语进行MASK，然后
使用MLM任务进行预测掩盖的内容

输入：一个组合两块菜板，轻松生熟分开，深度凹槽设计，收集[MASK][MASK]汁水，整洁台面

输出：一个组合两块菜板，轻松生熟分开，深度凹槽设计，收集蔬菜汁水，整洁台面，

使用 ``` python RunDataEnhance.py```

## PreFixNlg

使用连续的非离散前缀进行可控文本生成，属于提示学习的一种