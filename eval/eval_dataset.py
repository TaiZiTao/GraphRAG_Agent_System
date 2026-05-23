"""意图识别评估集 - 100 条，覆盖 6 种意图 + 边界 case

意图分布:
  search     30  (最常见的找商品)
  consult    20  (咨询/求建议，无具体商品名)
  compare    12  (对比)
  add_cart   12  (加购/下单)
  after_sale 13  (售后)
  chitchat   13  (闲聊)
"""

INTENT_DATASET = [
    # ============================================================
    # search (30 条)：明确找商品（带商品名/品类/属性）
    # ============================================================
    {"id": "I001", "query": "我想买一双板鞋", "expected": "search"},
    {"id": "I002", "query": "300以内的潮牌T恤", "expected": "search"},
    {"id": "I003", "query": "约会穿什么裙子", "expected": "search"},
    {"id": "I004", "query": "通勤包推荐", "expected": "search"},
    {"id": "I005", "query": "夏天的连衣裙", "expected": "search"},
    {"id": "I006", "query": "要一件牛仔外套", "expected": "search"},
    {"id": "I007", "query": "运动鞋推荐", "expected": "search"},
    {"id": "I008", "query": "白色衬衫", "expected": "search"},
    {"id": "I009", "query": "黑色短裙", "expected": "search"},
    {"id": "I010", "query": "Nike 跑鞋", "expected": "search"},
    {"id": "I011", "query": "500块以内的卫衣", "expected": "search"},
    {"id": "I012", "query": "性价比高的T恤", "expected": "search"},
    {"id": "I013", "query": "高端商务皮鞋", "expected": "search"},
    {"id": "I014", "query": "便宜点的牛仔裤", "expected": "search"},
    {"id": "I015", "query": "适合海边度假的衣服", "expected": "search"},
    {"id": "I016", "query": "气质风女装", "expected": "search"},
    {"id": "I017", "query": "复古连衣裙", "expected": "search"},
    {"id": "I018", "query": "学院风外套", "expected": "search"},
    {"id": "I019", "query": "找一双白鞋", "expected": "search"},
    {"id": "I020", "query": "潮牌帽子", "expected": "search"},
    {"id": "I021", "query": "想要个双肩包", "expected": "search"},
    {"id": "I022", "query": "查一下连衣裙", "expected": "search"},
    {"id": "I023", "query": "搜一下小白鞋", "expected": "search"},
    {"id": "I024", "query": "Adidas 短袖", "expected": "search"},
    {"id": "I025", "query": "派对礼服裙", "expected": "search"},
    {"id": "I026", "query": "300到500的外套", "expected": "search"},
    {"id": "I027", "query": "适合面试穿的西装", "expected": "search"},
    {"id": "I028", "query": "出差用的行李箱", "expected": "search"},
    {"id": "I029", "query": "宽松一点的T恤", "expected": "search"},
    {"id": "I030", "query": "短款羽绒服", "expected": "search"},

    # ============================================================
    # consult (20 条)：咨询/求建议（无具体商品名）
    # ============================================================
    {"id": "I031", "query": "我要给我男朋友买礼物", "expected": "consult"},
    {"id": "I032", "query": "送女朋友什么好", "expected": "consult"},
    {"id": "I033", "query": "你们家什么最值得买", "expected": "consult"},
    {"id": "I034", "query": "我应该穿什么风格", "expected": "consult"},
    {"id": "I035", "query": "帮我搭配一下", "expected": "consult"},
    {"id": "I036", "query": "送爸爸什么生日礼物", "expected": "consult"},
    {"id": "I037", "query": "圣诞节送男朋友什么", "expected": "consult"},
    {"id": "I038", "query": "有什么推荐的", "expected": "consult"},
    {"id": "I039", "query": "新人入手什么", "expected": "consult"},
    {"id": "I040", "query": "现在流行什么", "expected": "consult"},
    {"id": "I041", "query": "夏天穿什么显瘦", "expected": "consult"},
    {"id": "I042", "query": "我这种身材适合穿什么", "expected": "consult"},
    {"id": "I043", "query": "怎么搭配显腿长", "expected": "consult"},
    {"id": "I044", "query": "帮我推荐一下", "expected": "consult"},
    {"id": "I045", "query": "你们觉得哪个好看", "expected": "consult"},
    {"id": "I046", "query": "送闺蜜礼物", "expected": "consult"},
    {"id": "I047", "query": "什么风格适合我", "expected": "consult"},
    {"id": "I048", "query": "需要点建议", "expected": "consult"},
    {"id": "I049", "query": "给点穿搭建议", "expected": "consult"},
    {"id": "I050", "query": "母亲节送什么好", "expected": "consult"},

    # ============================================================
    # compare (12 条)：对比商品
    # ============================================================
    {"id": "I051", "query": "这两个哪个好", "expected": "compare"},
    {"id": "I052", "query": "A 和 B 比", "expected": "compare"},
    {"id": "I053", "query": "对比一下这两件", "expected": "compare"},
    {"id": "I054", "query": "Nike 和 Adidas 哪个好", "expected": "compare"},
    {"id": "I055", "query": "这两双鞋选哪个", "expected": "compare"},
    {"id": "I056", "query": "比较一下", "expected": "compare"},
    {"id": "I057", "query": "哪个性价比更高", "expected": "compare"},
    {"id": "I058", "query": "你觉得哪个更适合我", "expected": "compare"},
    {"id": "I059", "query": "第一个和第二个对比", "expected": "compare"},
    {"id": "I060", "query": "这件和那件区别", "expected": "compare"},
    {"id": "I061", "query": "差别在哪", "expected": "compare"},
    {"id": "I062", "query": "选哪个更好", "expected": "compare"},

    # ============================================================
    # add_cart (12 条)：加购/下单
    # ============================================================
    {"id": "I063", "query": "加入购物车", "expected": "add_cart"},
    {"id": "I064", "query": "我要这个", "expected": "add_cart"},
    {"id": "I065", "query": "下单", "expected": "add_cart"},
    {"id": "I066", "query": "加购", "expected": "add_cart"},
    {"id": "I067", "query": "买它", "expected": "add_cart"},
    {"id": "I068", "query": "就要这个了", "expected": "add_cart"},
    {"id": "I069", "query": "拍下第一个", "expected": "add_cart"},
    {"id": "I070", "query": "加到购物车", "expected": "add_cart"},
    {"id": "I071", "query": "我要买这件", "expected": "add_cart"},
    {"id": "I072", "query": "立即购买", "expected": "add_cart"},
    {"id": "I073", "query": "把这个加进去", "expected": "add_cart"},
    {"id": "I074", "query": "确认下单", "expected": "add_cart"},

    # ============================================================
    # after_sale (13 条)：售后
    # ============================================================
    {"id": "I075", "query": "我要退货", "expected": "after_sale"},
    {"id": "I076", "query": "怎么申请退款", "expected": "after_sale"},
    {"id": "I077", "query": "物流到哪了", "expected": "after_sale"},
    {"id": "I078", "query": "什么时候发货", "expected": "after_sale"},
    {"id": "I079", "query": "我的订单到哪了", "expected": "after_sale"},
    {"id": "I080", "query": "想换个尺码", "expected": "after_sale"},
    {"id": "I081", "query": "怎么联系客服", "expected": "after_sale"},
    {"id": "I082", "query": "包裹丢了", "expected": "after_sale"},
    {"id": "I083", "query": "申请售后", "expected": "after_sale"},
    {"id": "I084", "query": "退款进度", "expected": "after_sale"},
    {"id": "I085", "query": "怎么换货", "expected": "after_sale"},
    {"id": "I086", "query": "发票怎么开", "expected": "after_sale"},
    {"id": "I087", "query": "签收的东西有问题", "expected": "after_sale"},

    # ============================================================
    # chitchat (13 条)：闲聊（无购物意图）
    # ============================================================
    {"id": "I088", "query": "在吗", "expected": "chitchat"},
    {"id": "I089", "query": "你好", "expected": "chitchat"},
    {"id": "I090", "query": "你是谁", "expected": "chitchat"},
    {"id": "I091", "query": "今天天气真好", "expected": "chitchat"},
    {"id": "I092", "query": "讲个笑话", "expected": "chitchat"},
    {"id": "I093", "query": "嗨", "expected": "chitchat"},
    {"id": "I094", "query": "早上好", "expected": "chitchat"},
    {"id": "I095", "query": "你叫什么名字", "expected": "chitchat"},
    {"id": "I096", "query": "你能干嘛", "expected": "chitchat"},
    {"id": "I097", "query": "谢谢", "expected": "chitchat"},
    {"id": "I098", "query": "再见", "expected": "chitchat"},
    {"id": "I099", "query": "好的", "expected": "chitchat"},
    {"id": "I100", "query": "哈哈", "expected": "chitchat"},
]
