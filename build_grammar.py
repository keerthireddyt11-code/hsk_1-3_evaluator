"""Build data/hsk_grammar.json (HSK 2.0, Levels 1-3) - the RAG knowledge base.

SOURCE for the LIST of grammar points + patterns: Chinese Grammar Wiki, AllSet Learning,
  'HSK 1 / HSK 2 / HSK 3 grammar points' pages (resources.allsetlearning.com/chinese/grammar/HSK_N_grammar_points),
  accessed 2026-10-03. Wiki content is (c) AllSet Learning, NON-COMMERCIAL use with attribution (Creative Commons).
EXPLANATIONS and EXAMPLES below are written by the project author in original wording; the grammar point names and
patterns are factual and follow the wiki. HSK levels are the wiki's estimates (HSK 2.0 had no official grammar list).

Run: python build_grammar.py     (re-checks every example against the HSK 2.0 vocab guardrail)
"""
import json, sys
from src.vocab_check import check
import re, unicodedata
from src.vocab_check import VOCAB
from pypinyin import lazy_pinyin, Style, load_single_dict, load_phrases_dict

# Pinyin = pypinyin (context-aware, so 了/还/都 are read correctly), grouped into words (longest match on the HSK word list).
# Neutral tones (妈妈 māma, 朋友 péngyou ...) are taken from the vocab file's own Pinyin when it marks a syllable as neutral.
load_single_dict({ord("得"): "de"})                                          # complement 得 is 'de' (none of our rules use 'děi')
load_phrases_dict({"不了": [["bù"], ["le"]], "东西": [["dōng"], ["xi"]]})     # 不了 here is 不 + 了(le); 东西 = dōngxi
_PUNCT = {"。": ".", "？": "?", "！": "!", "，": ",", "、": ","}
_PROPER = {"北京", "中国"}                                                     # capitalised in Pinyin
_VPY = {w["hanzi"]: w["pinyin"].split() for w in json.load(open("data/hsk_vocab.json", encoding="utf-8"))}
_GROUP = set(VOCAB) | {"没有", "这里", "那里", "吃饭", "越来越", "很多", "一边", "一下", "一点", "有点"}   # only used to group syllables
_MAXW = max(len(w) for w in _GROUP)
_HAN = re.compile(r"([\u4e00-\u9fff]+)")

def _base(syl):   # strip tone marks: 'xī' -> 'xi'
    return "".join(c for c in unicodedata.normalize("NFD", syl) if not unicodedata.combining(c)).lower()

def _is_neutral(syl):
    return syl.isalpha() and syl == "".join(c for c in unicodedata.normalize("NFD", syl) if not unicodedata.combining(c))

def _words(run):  # longest-match grouping on the HSK 2.0 vocab
    out, i = [], 0
    while i < len(run):
        for n in range(min(_MAXW, len(run) - i), 0, -1):
            if run[i:i + n] in _GROUP or n == 1:
                out.append(run[i:i + n]); i += n; break
    return out

def _run_pinyin(run):
    syl = lazy_pinyin(run, style=Style.TONE)
    if len(syl) != len(run):                       # safety net: fall back to one syllable per character
        syl = [lazy_pinyin(c, style=Style.TONE)[0] for c in run]
    syl = ["yī" if c == "一" else "bù" if c == "不" else sy for c, sy in zip(run, syl)]   # base tones: no 一/不 sandhi
    out, i = [], 0
    for w in _words(run):
        part, i = syl[i:i + len(w)], i + len(w)
        v = _VPY.get(w)
        if v and len(v) == len(part):
            part = [vs if _is_neutral(vs) and _base(vs) == _base(ps) else ps for vs, ps in zip(v, part)]
        word = "".join(part)
        out.append(word[:1].upper() + word[1:] if w in _PROPER else word)
    return out

def to_pinyin(text):
    """Tone-marked Pinyin grouped by word. Base tones everywhere: no 一/不/third-tone sandhi ('yī gè', 'bù shì', 'nǐ hǎo')."""
    out = []
    for part in _HAN.split(text):
        if not part: continue
        if _HAN.fullmatch(part):
            out.extend(_run_pinyin(part))
        else:
            for ch in part:
                if ch in _PUNCT:
                    if out: out[-1] += _PUNCT[ch]
                elif not ch.isspace():
                    out.append(ch)
    s = " ".join(out)
    return s[:1].upper() + s[1:]

# (id, level, category, title, pattern, explanation, ok_zh, ok_en, bad_zh, bad_en, why, keywords, wiki_point)
R = [
# ---------------- HSK 1 ----------------
("G1-01",1,"word order","Basic sentence order","Subj. + Verb + Obj.",
 "The verb comes before its object, as in English. Do not put the object in front of the verb.",
 "我喝茶。","I drink tea.","我茶喝。","(I tea drink.)","The object is placed before the verb.",
 "subject verb object SVO word order","Basic sentence order"),
("G1-02",1,"adjectives","Adjectives do not need 是","Subj. + 很 + Adj.",
 "An adjective can be the predicate by itself, so do not add 是 before it. Put 很 (or another adverb) before the adjective.",
 "我很忙。","I am busy.","我是很忙。","(I am very busy, with a wrong 是.)","是 should not be used before an adjective.",
 "to be shi adjective hen very busy","Simple 'noun + adjective' sentences"),
("G1-03",1,"adverbs","都 and 也 go before the verb","Subj. + 都/也 + Verb",
 "都 (all) and 也 (also) come after the subject and before the verb. They do not go after the verb.",
 "我们都是学生。","We are all students.","我们是都学生。","(We are all students, with 都 after 是.)","都 must come before 是, not after it.",
 "dou ye all also adverb position","The 'all' adverb 'dou'; The 'also' adverb 'ye'"),
("G1-04",1,"questions","Yes/no questions with 吗","Statement + 吗？",
 "Add 吗 to the end of a statement to make a yes/no question. Do not put it in the middle.",
 "你喜欢喝茶吗？","Do you like drinking tea?","你吗喜欢喝茶？","(Do you like drinking tea? with 吗 in the wrong place.)","吗 belongs at the end of the sentence.",
 "ma question particle yes no","Yes-no questions with 'ma'"),
("G1-05",1,"questions","Follow-up questions with 呢","Statement，Noun/Pronoun + 呢？",
 "Put 呢 after a noun or pronoun to ask 'and you?' or 'what about...?'. The word comes first, then 呢.",
 "我很好，你呢？","I am fine, and you?","我很好，呢你？","(I am fine, and you? with 呢 first.)","呢 comes after the word, not before it.",
 "ne how about and you","Questions with 'ne'"),
("G1-06",1,"questions","Question words stay in place","Subj. + Verb + 什么/谁/哪儿",
 "A Chinese question word takes the place of the answer. It does not move to the front of the sentence like in English.",
 "你想吃什么？","What do you want to eat?","你什么想吃？","(What you want eat?)","什么 should come after the verb, where the answer would be.",
 "what who where shenme question word order","Placement of question words"),
("G1-07",1,"questions","No 吗 with question words","Question-word question (no 吗)",
 "A question that already has 什么, 谁, 哪儿, 几 or 怎么 does not need 吗.",
 "你叫什么名字？","What is your name?","你叫什么名字吗？","(What is your name? with an extra 吗.)","吗 should not be added to a question that has a question word.",
 "ma shenme redundant question word","Yes-no questions with 'ma'; Placement of question words"),
("G1-08",1,"measure words","几 needs a measure word","几 + Measure Word + Noun？",
 "To ask 'how many', put a measure word between 几 and the noun.",
 "你有几个朋友？","How many friends do you have?","你有几朋友？","(How many friends? with no measure word.)","The measure word 个 is missing.",
 "ji how many measure word ge","Measure words in quantity questions"),
("G1-09",1,"measure words","Numbers need a measure word","Number + 个 + Noun",
 "Between a number and a noun you need a measure word. 个 is the most common one.",
 "我有三个苹果。","I have three apples.","我有三苹果。","(I have three apples, no measure word.)","The measure word is missing.",
 "ge measure word counting classifier","Measure word 'ge'"),
("G1-10",1,"word order","Time words go before the verb","Subj. + Time + Verb  /  Time + Subj. + Verb",
 "Time words such as 明天 and 昨天 go before the verb: after the subject or at the start of the sentence. They do not go at the end.",
 "我明天去学校。","I am going to school tomorrow.","我去学校明天。","(I go to school tomorrow, with the time word last.)","The time word 明天 is placed at the end.",
 "time words tomorrow yesterday word order","Time words and word order"),
("G1-11",1,"word order","Place goes before the verb with 在","Subj. + 在 + Place + Verb",
 "To say where an action happens, put 在 + place before the verb.",
 "我在家看书。","I read at home.","我看书在家。","(I read at home, with the place last.)","在 + place must come before the verb.",
 "zai at location where action happens","Indicating location with 'zai' before verbs"),
("G1-12",1,"verbs","在 means 'be at', 有 means 'there is'","Thing + 在 + Place  /  Place + 有 + Thing",
 "Use 在 to say where something is. Use 有 to say that something exists in a place; the place comes first.",
 "学校里有很多学生。","There are many students in the school.","学校里在很多学生。","(In the school are many students, with 在.)","有, not 在, says that something exists.",
 "zai you exist there is location","Expressing existence in a place with 'zai'; Expressing existence with 'you'"),
("G1-13",1,"negation","Negate 有 with 没有","Subj. + 没(有) + Obj.",
 "The negative of 有 is 没有 (or just 没). Never say 不有.",
 "我没有钱。","I do not have any money.","我不有钱。","(I not have money.)","不 cannot be used to negate 有.",
 "mei you not have negation","Negation of 'you' with 'mei'"),
("G1-14",1,"negation","不 vs 没 (now/habit vs past)","不 + Verb (now, future, habit)  /  没(有) + Verb (past)",
 "Use 不 for the present, the future and habits. Use 没 for something that did not happen in the past.",
 "昨天我没去学校。","I did not go to school yesterday.","昨天我不去学校。","(Yesterday I do not go to school.)","A past action needs 没, not 不.",
 "bu mei negation past present habit","Comparing 'bu' and 'mei'; Negation of past actions with 'meiyou'"),
("G1-15",1,"negation","Use 不, not 没, with adjectives","不 + Adj.",
 "To say an adjective is not true, use 不. 没 is not used with adjectives.",
 "今天不冷。","It is not cold today.","今天没冷。","(Today not cold, with 没.)","没 cannot negate an adjective.",
 "bu negate adjective cold","Standard negation with 'bu'"),
("G1-16",1,"particles","了 after the verb for finished actions","Subj. + Verb + 了 + Obj.",
 "Put 了 right after the verb to show the action is finished. It does not go before the verb.",
 "我吃了一个苹果。","I ate an apple.","我了吃一个苹果。","(I le eat an apple, with 了 before the verb.)","了 is placed before the verb.",
 "le completed action finished","Expressing completion with 'le'"),
("G1-17",1,"particles","了 at the end for a new situation","... + 了",
 "Put 了 at the end of the sentence to show a change or a new situation. In 'not any more' sentences 了 also goes at the end.",
 "我不想吃了。","I do not want to eat any more.","我不了想吃。","(I not le want eat.)","了 must be at the end of the sentence.",
 "le sentence final change not anymore","Expressing 'not anymore' with 'le'; Expressing 'now' with 'le'"),
("G1-18",1,"particles","的 for possession: owner first","Noun/Pronoun 1 + 的 + Noun 2",
 "The owner comes first, then 的, then the thing that is owned.",
 "这是我的书。","This is my book.","这是书我的。","(This is book my.)","The owner and the thing are in the wrong order.",
 "de possession my your","Expressing possession with 'de'"),
("G1-19",1,"auxiliary verbs","能 and 会 need a verb","Subj. + 能/会 + Verb",
 "能 and 会 are followed by a verb. 会 can sometimes stand alone, but 能 cannot.",
 "我能说汉语。","I can speak Chinese.","我能汉语。","(I can Chinese.)","能 needs a verb after it.",
 "neng hui can ability skill","Expressing ability or possibility with 'neng'; Expressing a learned skill with 'hui'"),
("G1-20",1,"auxiliary verbs","想 goes before the verb","Subj. + 想 + Verb",
 "想 (would like to) comes before the main verb.",
 "我想喝茶。","I would like to drink tea.","我喝想茶。","(I drink would-like tea.)","想 is placed after the verb.",
 "xiang want would like","Expressing 'would like to' with 'xiang'"),
("G1-21",1,"verb phrases","去 + place first, then the purpose","Subj. + 去 + Place + Verb",
 "When you go somewhere to do something, say 去 + place first and the action after it.",
 "我去商店买东西。","I go to the shop to buy things.","我买东西去商店。","(I buy things go to the shop.)","The going and the action are in the wrong order.",
 "qu go purpose serial verbs","Directional verbs 'lai' and 'qu'; Using the verb 'qu'"),
("G1-22",1,"adverbs","太 ... 了 means 'too / so'","太 + Adj. + 了",
 "太 + adjective + 了 means 'too ...' or 'so ...'. Do not add 很 to it.",
 "今天太冷了。","It is too cold today.","今天很太冷。","(Today very too cold.)","很 and 太 should not be used together.",
 "tai too so very","Expressing 'excessively' with 'tai'"),
("G1-23",1,"numbers","Dates go from big to small","Month + 月 + Day + 号",
 "Say the month first, then the day. The bigger unit comes before the smaller one.",
 "今天是十月五号。","Today is October 5th.","今天是五号十月。","(Today is the 5th October, wrong order.)","The day is placed before the month.",
 "date month day yue hao","Structure of dates"),
("G1-24",1,"numbers","Age uses 岁, not 有","Subj. + Number + 岁",
 "Say someone's age with number + 岁. Do not use 有 as in English 'have'.",
 "我女儿五岁。","My daughter is five years old.","我女儿有五岁。","(My daughter has five years.)","有 should not be used to give an age.",
 "sui age years old","Age with 'sui'"),
("G1-25",1,"sentence patterns","是……的 stresses how, when or where","是 + Detail + 的",
 "是……的 stresses the time, the place or the way of something that has already happened. 的 goes at the end.",
 "你是怎么来的？","How did you come here?","你是怎么来？","(How did you come, missing 的.)","The final 的 is missing.",
 "shi de emphasis how when where","The 'shi... de' construction for emphasizing details"),
# ---------------- HSK 2 ----------------
("G2-01",2,"numbers","二 vs 两 before measure words","两 + Measure Word + Noun",
 "Before a measure word use 两, not 二. 二 is for counting and for numbers.",
 "我有两个朋友。","I have two friends.","我有二个朋友。","(I have two friends, with 二.)","二 is used before a measure word.",
 "er liang two measure word","Comparing 'er' and 'liang'"),
("G2-02",2,"prepositions","离 for distance","Place 1 + 离 + Place 2 + 很 + 近/远",
 "To say how far two places are, put 离 between them. The first place is the starting point.",
 "我家离学校很近。","My home is close to the school.","我家学校离很近。","(My home school from close.)","离 must be placed between the two places.",
 "li distance near far from","Expressing distance with 'li'"),
("G2-03",2,"comparison","比 sentences do not use 很","A + 比 + B + Adj.",
 "To compare, say A + 比 + B + adjective. Do not put 很 before the adjective. Use 更 for 'even more'.",
 "他比我高。","He is taller than me.","他比我很高。","(He is than me very tall.)","很 cannot be used in a 比 sentence.",
 "bi compare than taller","Basic comparisons with 'bi'"),
("G2-04",2,"adverbs","更 and 最 replace 很","更/最 + Adj.",
 "更 (even more) and 最 (most) are used instead of 很. Never use them together with 很.",
 "她最漂亮。","She is the prettiest.","她最很漂亮。","(She most very pretty.)","最 and 很 should not be used together.",
 "geng zui more most superlative","Superlative 'zui'; Expressing 'even more' with 'geng'"),
("G2-05",2,"prepositions","给 + person goes before the verb","Subj. + 给 + Person + Verb",
 "To do something for someone, put 给 + the person before the verb.",
 "我给妈妈打电话。","I phone my mother.","我打电话妈妈。","(I phone mother, missing 给.)","给 is missing before the person.",
 "gei for to someone","Expressing 'for' with 'gei'"),
("G2-06",2,"verb phrases","一起 goes before the verb","Subj. + 一起 + Verb",
 "一起 (together) comes before the verb.",
 "我们一起去商店吧。","Let us go to the shop together.","我们去一起商店吧。","(We go together shop.)","一起 is placed after the verb.",
 "yiqi together","Expressing 'together' with 'yiqi'"),
("G2-07",2,"verb phrases","一下 comes after the verb","Verb + 一下",
 "Put 一下 after the verb to make the action short and light.",
 "你看一下。","Take a look.","你一下看。","(You a-bit look.)","一下 is placed before the verb.",
 "yixia briefly a bit","Verbing briefly with 'yixia'"),
("G2-08",2,"particles","过 for past experience","Subj. + Verb + 过 + Obj.",
 "过 comes right after the verb and means you have done it before.",
 "我去过中国。","I have been to China.","我去中国过。","(I go China ever.)","过 must come right after the verb.",
 "guo experience have ever","Expressing experiences with 'guo'"),
("G2-09",2,"complements","Degree complement with 得","Subj. + Obj. + Verb + 得 + Adj.",
 "To say how well someone does something, put the object first, then Verb + 得 + adjective.",
 "他汉语说得很好。","He speaks Chinese very well.","他说得汉语很好。","(He speaks well Chinese good.)","The object is in the wrong place.",
 "de degree complement how well","Degree complement"),
("G2-10",2,"complements","Result complement 完","Subj. + Verb + 完 + 了",
 "完 means 'finished' and goes right after the verb.",
 "我吃完了。","I have finished eating.","我完吃了。","(I finish eat.)","完 is placed before the verb.",
 "wan finish result complement","Result complement '-wan' for finishing"),
("G2-11",2,"questions","Verb-not-verb questions: no 吗","Verb + 不 + Verb？",
 "Say the verb twice with 不 in the middle to ask a question. Do not add 吗.",
 "你去不去？","Are you going?","你去不去吗？","(Are you going or not? with 吗.)","吗 should not be added to a verb-not-verb question.",
 "affirmative negative question bu ma","Affirmative-negative question"),
("G2-12",2,"adverbs","快……了 means 'about to'","快 + Verb/Adj. + 了",
 "快 ... 了 means something is about to happen. 了 is needed at the end.",
 "快下雨了。","It is about to rain.","快下雨。","(About to rain, missing 了.)","The final 了 is missing.",
 "kuai about to soon le","Expressing 'about to happen' with 'le'"),
("G2-13",2,"adverbs","已经……了 means 'already'","已经 + Verb + 了",
 "已经 ... 了 means 'already'. Keep 了 at the end.",
 "他已经走了。","He has already left.","他已经走。","(He already leave.)","The final 了 is missing.",
 "yijing already le","Expressing 'already' with 'yijing'"),
("G2-14",2,"adverbs","有点 vs 一点","有点 + Adj.  /  Verb + Adj. + 一点",
 "有点 goes before an adjective and often means 'a bit too'. 一点 goes after the adjective, for example in polite requests.",
 "请说慢一点。","Please speak a bit slower.","请说一点慢。","(Please speak a bit slow.)","一点 must come after the adjective.",
 "youdian yidian a bit little","Comparing 'youdian' and 'yidian'"),
("G2-15",2,"particles","Describing phrase + 的 + noun","Phrase + 的 + Noun",
 "A phrase that describes a noun goes before the noun and is joined to it with 的.",
 "这是去学校的出租车。","This is the taxi to school.","这是出租车去学校的。","(This is the taxi to school, with the description after the noun.)","The describing phrase is placed after the noun.",
 "de modifier phrase before noun","Modifying nouns with phrase + 'de'"),
("G2-16",2,"measure words","这 and 那 need a measure word","这/那 + Measure Word + Noun",
 "After 这 or 那 you need a measure word before the noun.",
 "这本书很好。","This book is good.","这书很好。","(This book is good, no measure word.)","The measure word 本 is missing.",
 "this that zhe na measure word","Measure words with 'this' and 'that'"),
("G2-17",2,"auxiliary verbs","可以 for permission","Subj. + 可以 + Verb",
 "可以 comes before the verb to say that you may do something.",
 "这里可以坐吗？","May I sit here?","这里坐可以吗？","(Here sit may?)","可以 must come before the verb.",
 "keyi may permission allowed","Expressing permission with 'keyi'"),
("G2-18",2,"adverbs","正在 / 在 for actions in progress","Subj. + 正在/在 + Verb",
 "Put 正在 or 在 before the verb to say that the action is happening now.",
 "他正在看电视。","He is watching TV.","他看电视正在。","(He watches TV now-in-progress.)","正在 is placed after the verb.",
 "zhengzai zai progressive happening now","Expressing actions in progress with 'zai'"),
# ---------------- HSK 3 ----------------
("G3-01",3,"sentence patterns","把 sentences","Subj. + 把 + Obj. + Verb Phrase",
 "把 puts the object before the verb. The verb needs something after it, such as 完 or 了.",
 "我把书看完了。","I finished reading the book.","我看完把书了。","(I finished-read ba the book.)","把 + object must come before the verb.",
 "ba disposal sentence object before verb","Using 'ba' sentences"),
("G3-02",3,"verb phrases","了 + length of time","Verb + 了 + Duration + Obj.",
 "To say how long you did something, put the length of time after 了 and before the object.",
 "我学习了两年汉语。","I studied Chinese for two years.","我学习汉语了两年。","(I studied Chinese le two years.)","The length of time must come right after 了.",
 "le duration how long for years","Expressing duration with 'le'"),
("G3-03",3,"adverbs","再 (future) vs 又 (past)","再 + Verb (future)  /  又 + Verb + 了 (past)",
 "再 repeats an action in the future. 又 repeats an action that has already happened, usually with 了.",
 "明天我再来。","I will come again tomorrow.","明天我又来。","(Tomorrow I again come, with 又.)","又 is used for a future action.",
 "zai you again repeat","Comparing 'zai' and 'you'"),
("G3-04",3,"sentence patterns","一边……一边……","一边 + Verb 1，一边 + Verb 2",
 "Put 一边 before both verbs to say two actions happen at the same time.",
 "他一边吃饭，一边看电视。","He watches TV while eating.","他一边吃饭，看电视。","(He while eats, watches TV.)","The second 一边 is missing.",
 "yibian while at the same time","Simultaneous tasks with 'yibian'"),
("G3-05",3,"sentence patterns","如果 clause comes first","如果 + Condition，(就) + Result",
 "The 'if' part comes first and the result comes second. 就 often comes before the result.",
 "如果明天下雨，我就不去。","If it rains tomorrow, I will not go.","我就不去，如果明天下雨。","(I will not go, if it rains tomorrow.)","The 如果 part must come first.",
 "ruguo if then jiu condition","Expressing 'if... then...' with 'ruguo... jiu...'"),
("G3-06",3,"adverbs","越来越 means 'more and more'","越来越 + Adj. + 了",
 "越来越 means 'more and more'. Do not add 很 to it.",
 "天气越来越冷了。","The weather is getting colder and colder.","天气越来越很冷。","(Weather more-and-more very cold.)","很 should not be used with 越来越.",
 "yuelaiyue more and more getting","Expressing 'more and more' with 'yuelaiyue'"),
("G3-07",3,"verbs","让 + person + verb","Subj. + 让 + Person + Verb",
 "让 means 'let' or 'ask someone to'. The person comes right after 让, then the verb.",
 "妈妈让我去商店。","Mom asks me to go to the shop.","妈妈我让去商店。","(Mom me asks go shop.)","The person must come after 让.",
 "rang let make ask causative","Causative verbs"),
("G3-08",3,"conjunctions","还是 in questions, 或者 in statements","A + 还是 + B？",
 "Use 还是 to offer a choice in a question. 或者 is used in statements.",
 "你喝茶还是咖啡？","Do you want tea or coffee?","你喝茶或者咖啡？","(You drink tea or coffee, with 或者 in a question.)","还是, not 或者, is used in a choice question.",
 "haishi huozhe or choice","Offering choices with 'haishi'; Comparing 'haishi' and 'huozhe'"),
("G3-09",3,"particles","Adjective + 的 + noun","(很) + Adj. + 的 + Noun",
 "When an adjective with 很 describes a noun, put 的 between the adjective and the noun.",
 "我有一个很漂亮的朋友。","I have a very pretty friend.","我有一个很漂亮朋友。","(I have a very pretty friend, missing 的.)","The 的 after the adjective is missing.",
 "de adjective before noun modify","Modifying nouns with adjective + 'de'"),
("G3-10",3,"adverbs","就 (early) vs 才 (late)","Time + 就 + Verb + 了  /  Time + 才 + Verb",
 "就 shows something happened early. 才 shows it happened late, and 才 usually does not take 了.",
 "他六点就起床了。","He got up as early as six.","他六点才起床了。","(He only got up at six, with 了.)","才 usually does not go with 了.",
 "jiu cai early late only","Comparing 'cai' and 'jiu'; Expressing earliness with 'jiu'"),
]

WIKI = "https://resources.allsetlearning.com/chinese/grammar/HSK_{}_grammar_points"

# ---------------------------------------------------------------------------------------------
# REVIEW ROUND 1 (changes after a reviewer's notes; each change was checked, some were modified/rejected).
# error_strength of the wrong example:
#   clear          = ungrammatical in any register  -> safe to use as a definite error in the answer key
#   non-canonical  = understandable / grammatical but marked, or meaning-changing -> do NOT use as a definite error in the answer key
#   standard-written = wrong in standard written Chinese, common in informal writing
# kind: "error-pattern" (default) or "do-not-flag" (a pattern the AI must NOT mark as wrong)
# ---------------------------------------------------------------------------------------------
NONCANON = {"G1-01", "G1-11", "G1-21", "G1-24", "G1-25", "G2-12", "G2-13", "G3-08"}

def ex(zh, en, why=None):
    d = {"zh": zh, "en": en}
    if why: d["why"] = why
    return d

OVERRIDES = {
 "G1-01": dict(explanation="In neutral word order the verb comes before its object. Putting the object first (我茶喝) is only used for contrast or a topic, so beginners should use Subj. + Verb + Obj."),
 "G1-02": dict(title="Adjective sentences: 很 comes before the adjective", pattern="Subj. + 很 + Adj.",
   explanation="An adjective can be the predicate by itself, so a plain statement does not need 是 (我很忙). 是 + adjective does exist, but only for emphasis or contrast ('I AM busy, but...'), so do not mark it wrong. The degree word 很 goes before the adjective, not after it.",
   wrong_example=ex("我忙很。", "(I busy very.)", "很 must come before the adjective, not after it.")),
 "G1-09": dict(explanation="Between a number and a noun you need a measure word. 个 is very common, but many nouns have their own measure word (一本书, 一杯水)."),
 "G1-10": dict(title="Time words usually go before the verb",
   explanation="In neutral word order a time expression goes before the main verb: after the subject or at the start of the sentence (我明天去学校). Putting it after the verb or at the end is non-standard; in speech it can be added as an afterthought, but beginners should avoid it.",
   wrong_example=ex("我去明天学校。", "(I go tomorrow school.)", "The time word is placed after the verb; it should come before it.")),
 "G1-11": dict(explanation="To say where an action happens, the normal order is 在 + place before the verb (我在家看书). Putting the place after the verb and object (我看书在家) is non-canonical in neutral speech, so teach the standard order."),
 "G1-14": dict(title="不 vs 没(有)", pattern="不 + Verb  /  没(有) + Verb (+ 过)",
   explanation="不 commonly negates habitual, general, present/future or willing (volitional) situations. 没(有) commonly negates an event that did not happen or has not happened yet. 'Never done it' uses 没(有) + Verb + 过.",
   correct_example=ex("我没去过北京。", "I have never been to Beijing."),
   wrong_example=ex("我不去过北京。", "(I not have-been to Beijing.)", "不 cannot be used with 过; use 没(有) for 'have never'.")),
 "G1-15": dict(title="不 (太) before adjectives", pattern="不 (太) + Adj.",
   explanation="不 is the normal way to negate an adjective (不冷, 不太冷). 没 is for events, and it does not combine with a degree word such as 很.",
   correct_example=ex("今天不太冷。", "It is not very cold today."),
   wrong_example=ex("今天没很冷。", "(Today not-have very cold.)", "没 cannot be used together with 很 before an adjective; use 不太.")),
 "G1-16": dict(title="Verbal 了 goes right after the verb",
   explanation="了 right after a verb marks a bounded, completed event, often with a quantity (我吃了一个苹果). It is not simply a past-tense marker, and it never goes before the verb. Sentence-final 了 is different (see the 了 comparison rule)."),
 "G1-21": dict(explanation="To say you go somewhere in order to do something, put 去 + place first and the purpose after it. The reverse order describes two separate actions one after the other (buy things, then go to the shop).",
   wrong_example=ex("我买东西去商店。", "(I buy things, go to the shop.)", "With this order the shop is no longer the place where you buy things.")),
 "G1-24": dict(explanation="Say someone's age with number + 岁. 有 + age (有五岁了) appears in speech only for emphasis, so beginners should not use 有 here."),
 "G1-25": dict(explanation="是……的 is an emphasis construction for the time, place or manner of an event you already know about. It is not needed for every past event, and 是 can be dropped in positive sentences. In the usual form 的 closes the sentence."),
 "G2-03": dict(title="The adjective after 比 has no 很",
   explanation="In A + 比 + B + adjective, the adjective stands without 很. To add degree, put 更 or 还 before it, or 一点 or 多了 after it (他比我高一点)."),
 "G2-04": dict(title="更 and 最 are degree words before adjectives",
   explanation="更 (even more) and 最 (most) go directly before an adjective, in the same place as 很. An adjective takes only one degree word, so 最很 or 很更 is wrong."),
 "G2-09": dict(title="得 comes right after the verb",
   pattern="Subj. + Verb + 得 + Adj.  /  Subj. + Obj. + Verb + 得 + Adj.  /  Subj. + Verb + Obj. + Verb + 得 + Adj.",
   explanation="To say how well someone does something, 得 sits right after the verb. With an object, either repeat the verb (他说汉语说得很好) or move the object before the verb (他汉语说得很好). 得 never goes after the object.",
   wrong_example=ex("他说汉语得很好。", "(He speaks Chinese de very well.)", "得 is placed after the object; it must follow the verb.")),
 "G2-14": dict(explanation="一点 means 'a little'. It can follow an adjective (慢一点) or a verb (吃一点), or come before a noun (一点水). 有点 goes before an adjective and often adds a slightly negative feeling (有点贵)."),
 "G2-16": dict(title="这/那 + (number) + measure word", pattern="这/那 + (Number) + Measure Word + Noun",
   explanation="In standard Chinese, 这 and 那 are followed by a measure word before the noun (这本书). Some colloquial phrases drop it, but after a number it is always needed (这两本书).",
   correct_example=ex("这两本书很好。", "These two books are good."),
   wrong_example=ex("这两书很好。", "(These two book good.)", "The measure word 本 is missing after the number.")),
 "G2-17": dict(title="可以 goes before the verb; do not stack modals", pattern="Subj. + 可以 + Verb",
   explanation="可以 (may / can) comes before the verb. Do not put it together with 能, 会 or 想 in front of the same verb.",
   correct_example=ex("我可以坐这里吗？", "May I sit here?"),
   wrong_example=ex("我能可以坐这里吗？", "(I can may sit here?)", "Two modal verbs (能 and 可以) are stacked; use only one.")),
 "G3-02": dict(title="How long: 了 + duration",
   pattern="Verb + 了 + Duration + (的) + Obj.  /  Verb + Obj. + Verb + 了 + Duration",
   explanation="There are several ways to say how long: put the duration after 了 and before the object (学习了两年汉语), or repeat the verb (学习汉语学习了两年). The duration cannot simply follow the object."),
 "G3-03": dict(title="再 vs 又 (repeating)", pattern="再 + Verb  /  又 + Verb + 了",
   explanation="再 is for a repeat that has not happened yet: plans, requests and wishes (请再说一次). 又 is for a repeat that has already happened (他又迟到了), and it can also mark a regular recurrence.",
   correct_example=ex("请再说一次。", "Please say it once more."),
   wrong_example=ex("请又说一次。", "(Please again say it once more.)", "又 is not used in requests; use 再.")),
 "G3-04": dict(title="Two actions at once: 一边……一边……",
   explanation="Put 一边 before each verb to say two actions happen at the same time. 和 joins nouns (我和他), not two full actions.",
   wrong_example=ex("他吃饭和看电视。", "(He eats and watches TV, with 和.)", "和 does not join two actions; use 一边……一边…….")),
 "G3-05": dict(title="如果 sets the condition; the result uses 就",
   explanation="如果 starts the condition and usually comes first (in speech it can follow as an afterthought). The result part often uses 就, not 所以 or 但是.",
   wrong_example=ex("如果明天下雨，所以我不去。", "(If it rains tomorrow, so I will not go.)", "所以 belongs with 因为, not 如果; use 就 in the result.")),
 "G3-08": dict(explanation="还是 offers a choice in a question and is the normal form there. 或者 is used in statements ('either ... or')."),
 "G3-10": dict(title="就 (early) vs 才 (late) before the verb", pattern="Time + 就 + Verb  /  Time + 才 + Verb",
   explanation="就 and 才 go after the time expression and before the verb. 就 suggests early or quick; 才 suggests late or slow, and in the 'not until' sense it is normally not used with sentence-final 了.",
   wrong_example=ex("他才起床六点。", "(He only got-up six o'clock.)", "才 and the time are in the wrong order; the time comes first.")),
}

NEW = [
 dict(rule_id="G1-26", level=1, kind="do-not-flag", category="word order", title="The subject can be left out",
   pattern="(Subj.) + Verb ... (subject understood from context)",
   explanation="When the subject is clear from context, Chinese often leaves it out (昨天去了商店). Do not mark such a sentence as wrong just because it has no subject.",
   correct_example=ex("昨天去了商店。", "(I) went to the shop yesterday."), wrong_example=None,
   source_point="Added from reviewer feedback (not a wiki list item)", keywords="omit subject dropped pronoun context"),
 dict(rule_id="G2-19", level=2, category="particles", title="了: after the verb vs at the end",
   pattern="Verb + 了 + Obj.  /  ... + 了",
   explanation="Verbal 了 follows the verb and marks a completed event (我吃了两个苹果). Sentence-final 了 marks a change or new situation (我吃苹果了). Both together mean 'so far' (我吃了两个苹果了). 了 is not a past-tense marker: habits and states do not take it.",
   correct_example=ex("我吃了两个苹果了。", "I have eaten two apples so far."),
   wrong_example=ex("我每天去了学校。", "(Every day I went-le to school.)", "了 is not used for a habitual action."),
   source_point="Added from reviewer feedback; relates to 'Expressing completion with le' and 'Expressing now with le'", keywords="le verbal sentence final habitual not past tense"),
 dict(rule_id="G2-20", level=2, category="measure words", title="The measure word must fit the noun",
   pattern="Number + Measure Word + Noun",
   explanation="Many nouns need their own measure word: 一杯水, 一本书, 一个人. 个 is very common but is not a replacement for all of them.",
   correct_example=ex("我要一杯水。", "I want a glass of water."),
   wrong_example=ex("我要一个水。", "(I want one 个 water.)", "水 is counted with 杯, not 个."),
   source_point="Added from reviewer feedback; relates to 'Measure words for counting'", keywords="measure word classifier ge bei ben match noun"),
 dict(rule_id="G2-21", level=2, category="complements", title="Potential complement: 不 goes inside",
   pattern="Verb + 不 + Complement (cannot)  /  Verb + 得 + Complement (can)",
   explanation="To say you cannot manage to do something, put 不 between the verb and its result: 听不懂 (cannot understand). Do not put 不 before the verb.",
   correct_example=ex("我听不懂。", "I cannot understand (what I hear)."),
   wrong_example=ex("我不听懂。", "(I not listen-understand.)", "不 must go between the verb and the complement."),
   source_point="Potential complement '-bu dong' for not understanding", keywords="bu dong cannot potential complement understand"),
 dict(rule_id="G3-11", level=3, category="particles", title="的, 地 and 得",
   pattern="Modifier + 的 + Noun  /  Adverb + 地 + Verb  /  Verb + 得 + Complement",
   explanation="的 links a describing word to a noun (红色的车). 地 links an adverb to a verb (慢慢地走). 得 links a verb to its complement (跑得很快). All three sound like 'de', so they cannot be told apart in Pinyin.",
   correct_example=ex("他跑得很快。", "He runs very fast."),
   wrong_example=ex("他跑的很快。", "(He runs de very fast.)", "After a verb the complement marker is 得, not 的."),
   error_strength="standard-written", source_point="Structural particle 'de'", keywords="de di de particles structural complement"),
 dict(rule_id="G3-12", level=3, category="complements", title="Direction words 来 and 去 after verbs",
   pattern="Verb + Place + 来/去",
   explanation="来 (toward the speaker) or 去 (away) can follow a verb. If the verb has a place object, the place goes between them: 回家去, not 回去家.",
   correct_example=ex("他回家去了。", "He went back home."),
   wrong_example=ex("他回去家了。", "(He went-back-away home.)", "The place should come between 回 and 去."),
   source_point="Direction complement", keywords="lai qu direction complement place object"),
]

def finish(r, kw=None):
    r.setdefault("standard", "HSK2.0"); r.setdefault("kind", "error-pattern")
    for key in ("correct_example", "wrong_example"):
        e = r.get(key)
        if e:   # put pinyin right after the Chinese
            r[key] = {"zh": e["zh"], "pinyin": to_pinyin(e["zh"]), **{k: v for k, v in e.items() if k != "zh"}}
    if r["wrong_example"] is None:
        r["error_strength"] = "n/a"
    else:
        r.setdefault("error_strength", "non-canonical" if r["rule_id"] in NONCANON else "clear")
    r.setdefault("source", f"Chinese Grammar Wiki (AllSet Learning), HSK {r['level']} grammar points, {WIKI.format(r['level'])}, accessed 2026-10-03")
    if r["kind"] == "do-not-flag":
        r["source"] = "Author addition from reviewer feedback (general grammar)"
        mistake = "Do NOT mark this pattern as wrong."
    else:
        mistake = f"Common mistake: {r['wrong_example']['why']}"
    kw = kw or r.pop("keywords", "")
    r["embed_text"] = f"{r['title']}. {r['explanation']} {mistake} Keywords: {kw}. Pattern: {r['pattern']}"
    return r

rules, problems = [], []
for (rid, lvl, cat, title, pat, expl, okz, oke, badz, bade, why, kw, wp) in R:
    r = {"rule_id": rid, "level": lvl, "category": cat, "title": title, "pattern": pat, "explanation": expl,
         "correct_example": ex(okz, oke), "wrong_example": ex(badz, bade, why), "source_point": wp}
    r.update(OVERRIDES.get(rid, {}))
    rules.append(finish(r, kw))
for n in NEW:
    rules.append(finish(dict(n)))

order = lambda r: (r["level"], r["rule_id"])
rules.sort(key=order)
for r in rules:
    texts = [("correct", r["correct_example"]["zh"]), ("pattern", r["pattern"]), ("title", r["title"]), ("explanation", r["explanation"])]
    if r["wrong_example"]: texts.append(("wrong", r["wrong_example"]["zh"]))
    for label, text in texts:
        flagged = check(text)["out_of_scope_lenient"]
        if flagged: problems.append((r["rule_id"], label, text[:30], flagged))

ids = [r["rule_id"] for r in rules]
assert len(ids) == len(set(ids)), "duplicate rule ids"
json.dump({"meta": {
    "standard": "HSK 2.0, Levels 1-3",
    "grammar_source": "Chinese Grammar Wiki (AllSet Learning) HSK 1/2/3 grammar point lists, accessed 2026-10-03; plus reviewer-suggested additions",
    "license_note": "Wiki: non-commercial use with attribution (Creative Commons) - confirm exact terms on the wiki. Explanations/examples here are original wording.",
    "caveat": "HSK 2.0 had no official grammar standard; levels are the wiki's estimates.",
    "error_strength_key": "clear = safe definite error for the answer key; non-canonical = marked/meaning-changing, do not use as a definite error; standard-written = wrong in formal writing only",
    "rule_count": len(rules)}, "rules": rules},
    open("data/hsk_grammar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

from collections import Counter
print(f"{len(rules)} rules written | per level: {dict(Counter(r['level'] for r in rules))}")
print("error_strength:", dict(Counter(r['error_strength'] for r in rules)), "| kinds:", dict(Counter(r['kind'] for r in rules)))
print("OUT-OF-SCOPE words in examples/patterns/text:" if problems else "All examples & patterns pass the HSK 2.0 vocab check.")
for p in problems: print("  ", p)