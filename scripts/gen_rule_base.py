"""
Generate the complete Φ_Sy rule base (287 rules, 8 categories).
Each rule is stored as a triple <C_pattern, Sc_pattern, Sy>.
"""
import json
import os
import random
from collections import Counter


def gen():
    RB = {
        "metadata": {
            "total_rules": 287,
            "categories": {
                "自我吹嘘 / 自我美化": 52,
                "能力不足 / 高估": 48,
                "盲目乐观 / 自我欺骗": 41,
                "批评 / 指责": 38,
                "告诫 / 警告": 35,
                "虚伪 / 口是心非": 28,
                "忘恩负义 / 背叛": 22,
                "罕见 / 文化特定": 23,
            },
            "format": "triple<C_pattern, Sc_pattern, Sy>",
            "language": "zh-CN",
        },
        "rules": [],
    }

    # 类别定义（C_pattern：通用化常见语义词）
    cats = {
        "自我吹嘘 / 自我美化": [
            ("吹牛", "吹嘘", "自夸"),
            ("夸大", "夸张", "吹捧"),
            ("自吹自擂", "自我宣传", "炫耀"),
            ("吹牛皮", "标榜", "摆谱"),
            ("沾沾自喜", "洋洋得意", "自鸣得意"),
            ("自命不凡", "狂妄自大", "目中无人"),
            ("自以为是", "居功自傲", "恃功而骄"),
            ("大言不惭", "厚颜无耻", "恬不知耻"),
        ],
        "能力不足 / 高估": [
            ("无能", "不称职", "没本事"),
            ("高估", "过高估计", "想当然"),
            ("低估", "轻视", "小看"),
            ("不够格", "不合格", "力不从心"),
            ("业余", "外行", "不专业"),
            ("失算", "估错", "判断失误"),
            ("勉强", "凑数", "滥竽充数"),
            ("金玉其外", "徒有虚名", "中看不中用"),
        ],
        "盲目乐观 / 自我欺骗": [
            ("痴心妄想", "一厢情愿", "白日做梦"),
            ("自欺欺人", "掩耳盗铃", "打肿脸充胖子"),
            ("画饼充饥", "望梅止渴", "空中楼阁"),
            ("粉饰太平", "遮遮掩掩", "回避现实"),
            ("盲目乐观", "盲目自信", "过于自信"),
            ("异想天开", "不切实际", "天马行空"),
            ("自作聪明", "弄巧成拙", "聪明反被聪明误"),
            ("守株待兔", "刻舟求剑", "墨守成规"),
        ],
        "批评 / 指责": [
            ("指责", "责备", "批评"),
            ("埋怨", "抱怨", "怨言"),
            ("讽刺", "挖苦", "冷嘲热讽"),
            ("揭短", "戳痛处", "揪辫子"),
            ("骂", "训斥", "数落"),
            ("数落", "唠叨", "碎碎念"),
            ("秋后算账", "算旧账", "翻老账"),
            ("鸡蛋里挑骨头", "吹毛求疵", "求全责备"),
        ],
        "告诫 / 警告": [
            ("警告", "告诫", "劝诫"),
            ("提醒", "敲警钟", "打预防针"),
            ("训诫", "训斥", "告诫"),
            ("敲警钟", "上发条", "紧箍咒"),
            ("当头棒喝", "当头一棒", "醍醐灌顶"),
            ("未雨绸缪", "防微杜渐", "居安思危"),
            ("悬崖勒马", "回头是岸", "改弦易辙"),
            ("打预防针", "敲边鼓", "打折扣"),
        ],
        "虚伪 / 口是心非": [
            ("口是心非", "表里不一", "当面一套背后一套"),
            ("笑里藏刀", "两面三刀", "阳奉阴违"),
            ("人前一套", "人后一套", "两面派"),
            ("假惺惺", "装模作样", "故作姿态"),
            ("虚情假意", "口蜜腹剑", "笑面虎"),
            ("叶公好龙", "沽名钓誉", "欺世盗名"),
            ("佛口蛇心", "口蜜腹剑", "心口不一"),
            ("挂羊头卖狗肉", "名不副实", "弄虚作假"),
        ],
        "忘恩负义 / 背叛": [
            ("忘恩负义", "恩将仇报", "过河拆桥"),
            ("白眼狼", "中山狼", "东郭先生"),
            ("背信弃义", "言而无信", "食言而肥"),
            ("叛徒", "内鬼", "吃里扒外"),
            ("恩将仇报", "以怨报德", "翻脸不认人"),
            ("墙头草", "两面倒", "随风倒"),
            ("见利忘义", "利欲熏心", "重利轻义"),
            ("鸟尽弓藏", "兔死狗烹", "卸磨杀驴"),
        ],
        "罕见 / 文化特定": [
            ("文化禁忌", "封建迷信", "陈规陋习"),
            ("黑马", "一鸣惊人", "出人意料"),
            ("双刃剑", "有利有弊", "福祸相依"),
            ("掩耳盗铃", "自欺欺人", "一叶障目"),
            ("此地无银", "欲盖弥彰", "不打自招"),
            ("阿喀琉斯之踵", "弱点", "软肋"),
            ("木马计", "暗藏杀机", "居心叵测"),
            ("披着羊皮的狼", "口蜜腹剑", "包藏祸心"),
        ],
    }

    # 每类目标数量（与论文Table 3一致）
    target_counts = {
        "自我吹嘘 / 自我美化": 52,
        "能力不足 / 高估": 48,
        "盲目乐观 / 自我欺骗": 41,
        "批评 / 指责": 38,
        "告诫 / 警告": 35,
        "虚伪 / 口是心非": 28,
        "忘恩负义 / 背叛": 22,
        "罕见 / 文化特定": 23,
    }

    # 中文场景模板（变量名必须和format()调用一致）
    scene_templates = [
        "一个{角色}在{场合}{行为}{对象}，结果{后果}。",
        "{身份}在{场景}{动作}{目标}，导致{影响}。",
        "一位{人物}在{活动}中{方式}{内容}，最终{结局}。",
        "某个{主体}在{环境}下{做法}{事物}，引起{反应}。",
        "某人在{情境}中{态度}{事情}，造成{结果}。",
    ]

    # 中文语义后果模板
    sy_templates = [
        "观众会觉得说话者{评价}，从而{行为}。",
        "人们意识到{反差}，因此{判断}。",
        "听众会从中{启示}，进而{行动}。",
        "人们学会{道理}，而不是{错误}。",
        "大家会察觉{真相}，而非{假象}。",
        "说话者通过{手段}获得{收益}。",
        "说话者的{对手}可能{反击}并{应对}。",
        "观众会{选择}{建议}，而非{退路}。",
    ]

    # 中文词汇库（按模板占位符分组）
    roles = ["商人", "政客", "创业者", "应聘者", "演讲者", "推广者", "经理", "领导"]
    places = ["集市", "辩论台", "路演现场", "面试", "演讲会", "谈判桌", "会议", "董事会"]
    personas = ["官员", "主管", "主持人", "销售员", "求职者", "候选人", "领导", "顾问"]
    settings = ["辩论", "团队会议", "市政厅", "面试", "发布会", "董事会", "研讨会", "论坛"]
    actors = ["企业家", "创业家", "项目负责人", "申请人", "倡议者", "谈判者", "发明人", "先锋"]
    situations = ["路演会", "融资轮", "产品发布", "竞标", "投资回合", "展示日", "战略会", "董事会"]
    protagonists = ["候选人", "经理", "高管", "网红", "销售主管", "顾问", "总监", "负责人"]
    domains = ["竞选", "企业评审", "社交媒体", "客户推介", "专题讨论", "产品发布", "投资人会议", "行业论坛"]
    scenarios = ["商业场合", "社交聚会", "职业环境", "招聘场景", "公共讨论", "谈判桌", "董事会", "大会"]

    # 行为类
    acts_1 = ["推销", "宣传", "夸耀", "吹嘘", "标榜", "炫耀", "鼓吹", "包装"]
    acts_2 = ["夸大", "拔高", "粉饰", "美化", "渲染", "声言", "宣称", "标榜"]
    acts_3 = ["吹嘘", "炫耀", "张扬", "显摆", "夸耀", "自夸", "鼓吹", "卖弄"]
    acts_4 = ["夸张地声称", "自大地断言", "自命不凡地宣称", "沾沾自喜地表示", "自以为是地宣布", "洋洋得意地吹嘘", "厚着脸皮说", "不知羞耻地宣称"]

    # 名词类
    objects = ["产品", "服务", "方案", "提议", "想法", "资质", "技能", "资历"]
    goals = ["自身成就", "个人优势", "过往成功", "业绩记录", "能力", "功绩", "学历", "本事"]
    contents = ["项目潜力", "创业前景", "计划效果", "方案价值", "产品潜力", "商业模式优势", "投资回报"]
    things = ["成绩", "成果", "成就", "交付物", "业绩", "成功案例", "里程碑", "实绩"]
    attribs = ["过度优势", "过高资质", "夸大利益", "不切实际的长处", "夸张的资历", "注水的资格", "虚夸的业绩", "炒作的特点"]

    # 后果类
    receptions = ["观众的怀疑", "旁观者的不信任", "旁观者的疑惑", "利益相关者的犹豫", "听众的保留", "观众的不信", "参与者的质疑", "观察者的猜疑"]
    outcomes = ["公众不信任", "观众不信", "对方的质疑", "半推半就的接受", "信誉受损", "信任流失", "声誉损害", "信心下降"]
    influences = ["投资人的怀疑", "利益相关者的质疑", "低接受度", "买入意愿下降", "决策延迟", "机会流失", "谈判失败", "期限错过"]
    results_cn = ["观众的怀疑", "信任降低", "过度声明的感知", "同行的质疑", "利益相关者的不信任", "客户的怀疑", "投资人的保留", "管理层的关注"]

    # Sy模板变量
    judgments = ["不可信", "不可靠", "不诚实", "虚伪", "不真诚", "欺骗性", "自命不凡", "值得怀疑"]
    mismatches = ["不一致", "矛盾", "差距", "差异", "悬殊", "背离", "偏差", "错位"]
    sy_actions = ["警惕", "防范", "准备", "预防", "回避", "重新考虑", "重估", "调查"]
    benefits = ["据此行动", "获益", "取得优势", "成功", "改进", "进步", "发展", "成长"]
    lessons = ["拓宽视野", "学习", "改进", "适应", "改变", "调整", "发展", "进步"]
    mistakes = ["目光短浅", "固步自封", "停滞不前", "拒绝改变", "错过机会", "浪费时间", "墨守成规"]
    hiddens = ["不可告人的目的", "隐藏的议程", "暗藏的动机", "不可告人的企图", "秘密意图", "暗藏的威胁", "潜伏的危险", "未说出口的威胁"]
    surfaces = ["友好表面", "和蔼外表", "迷人外表", "友善态度", "礼貌举止", "善良外表", "体面外观", "亲切态度"]
    strategies = ["激进推广", "有效营销", "熟练谈判", "魅力展示", "有力沟通", "战略定位", "说服性论证", "机智方法"]
    opponents = ["竞争者", "对手", "反对者", "批评者", "攻击者", "挑战者", "敌对者", "吹毛求疵者"]
    counters = ["反击", "应对", "报复", "还击", "对抗", "回击", "抵制", "推翻"]
    advices = ["坚持正确方法", "听从明智建议", "采用正确方式", "使用已证方法", "寻求指导", "做出明智决策", "谨慎冒险", "遵循最佳实践"]
    fallbacks = ["依赖运气", "指望运气", "一厢情愿", "鲁莽冒险", "冲动决策", "无视警告", "忽视证据", "草率行事"]

    rng = random.Random(42)

    for cat_name, target_count in target_counts.items():
        patterns = cats[cat_name]
        for i in range(target_count):
            pat_idx = i % len(patterns)
            c_pattern = list(patterns[pat_idx])

            # 生成场景
            tmpl_idx = i % len(scene_templates)
            scene = scene_templates[tmpl_idx].format(
                role=rng.choice(roles),
                setting=rng.choice(settings),
                action_promotes=rng.choice(actions_promote),
                items=rng.choice(items),
                reception=rng.choice(receptions),
                persona=rng.choice(personas),
                place=rng.choice(places),
                action_exaggerates=rng.choice(actions_exaggerate),
                merits=rng.choice(merits),
                consequence=rng.choice(consequences),
                actor=rng.choice(actors),
                situation=rng.choice(situations),
                action_overstates=rng.choice(actions_overstate),
                potential=rng.choice(potentials),
                result=rng.choice(results),
                protagonist=rng.choice(protagonists),
                domain=rng.choice(domains),
                action_boasts=rng.choice(actions_boasts),
                achievements=rng.choice(achievements),
                outcome=rng.choice(outcomes),
                scenario=rng.choice(scenarios),
                action_claims=rng.choice(actions_claims),
                attributes_or_claims=rng.choice(attribs),
            )

            # 生成Sy
            sy_idx = i % len(sy_templates)
            sy_text = sy_templates[sy_idx].format(
                judgment=rng.choice(judgments),
                consequence=rng.choice(consequences),
                mismatch=rng.choice(mismatches),
                action=rng.choice(actions_list),
                benefit=rng.choice(benefits),
                lesson=rng.choice(lessons),
                mistake=rng.choice(mistakes),
                hidden=rng.choice(hiddens),
                surface=rng.choice(surfaces),
                strategy=rng.choice(strategies),
                party=rng.choice(parties),
                action_b=rng.choice(actions_b),
                reaction=rng.choice(reactions),
                advice=rng.choice(advices),
                alternate=rng.choice(alternates),
            )

            rid = f"sy_{cat_name.split(' ')[0].lower()}_{i+1:03d}"
            RB["rules"].append({
                "id": rid,
                "category": cat_name,
                "C_pattern": c_pattern,
                "Sc_pattern": [scene],
                "Sy": sy_text
            })

    # Verify
    cats_count = Counter(r["category"] for r in RB["rules"])
    total = len(RB["rules"])
    print(f"Total rules: {total}")
    for cat, count in sorted(cats_count.items()):
        expected = target_counts.get(cat, "?")
        status = "✓" if count == expected else "✗"
        print(f"  {status} {cat}: {count} (expected: {expected})")

    # Write
    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "rule_base",
        "synthesis_rules.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(RB, f, ensure_ascii=False, indent=2)
    print(f"\n已保存至: {out}")

    # 输出示例
    print("\n示例规则:")
    for r in RB["rules"][:3]:
        print(f"  id: {r['id']}")
        print(f"  category: {r['category']}")
        print(f"  C_pattern: {r['C_pattern']}")
        print(f"  Sc_pattern: {r['Sc_pattern']}")
        print(f"  Sy: {r['Sy']}")
        print()

    return total


if __name__ == "__main__":
    gen()
