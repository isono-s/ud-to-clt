# obtain CLT and DLT costs from a CoNLL-U file

from dataclasses import dataclass
import re
from nltk.tree import Tree as nltkTree
from nltk.tree.prettyprinter import TreePrettyPrinter
import argparse

def main(input_path: str, output_path: str | None = None, is_headfinal: bool = False, save_trees: bool = False):
    sents = read_ud(input_path)
    sents = [attach_unattached(sent) for sent in sents]
    sents = [calc_dlt(sent) for sent in sents]
    sents = [fix_crossed(sent) for sent in sents]
    trees = [ud2tree(sent, is_headfinal) for sent in sents]
    if save_trees:
        for i, tree in enumerate(trees):
            save_tree(tree, f"parses/tree_{i}_enf.txt")
    trees = [calc_clt_stor(calc_clt(enf2inc(tree))) for tree in trees]
    if save_trees:
        for i, tree in enumerate(trees):
            save_tree(tree, f"parses/tree_{i}_inc.txt")
    save_clt(trees, output_path)

@dataclass
class UDToken:
    token_id: str
    word: str
    pos: str
    head: str
    deptype: str
    preds: dict[str, int]
    misc: str

@dataclass
class Sent:
    sent_id: str | None
    tokens: list[UDToken]

def read_ud(path: str) -> list[Sent]:
    # read a UD annotation file and return the list of sentences:
    sents = []
    with open(path) as f:
        pgs = f.read().strip().split("\n\n")
    for pg in pgs:
        sent = Sent(None, [])
        lines = [line.strip() for line in pg.split("\n")]
        for line in lines:
            m = re.match("# (.*)", line)
            if m:
                sent.sent_id = m.group(1)
                continue
            if line[0] == "#":
                continue
            cols = line.split("\t")
            sent.tokens.append(UDToken(str(cols[0]), cols[1], cols[3], str(cols[6]), cols[7],
                                        {"dlt_integ_nv":0, "dlt_integ_nvjr":0, "dlt_integ_char":0, "dlt_stor":0, "dlt_count":0}, cols[9]))
        sents.append(sent)
    return sents

def attach_unattached(sent: Sent) -> Sent:
    # attach tokens to the preceding one if its head is unspecified
    for i, token in enumerate(sent.tokens):
        if token.head == "_":
            token.head = sent.tokens[i-1].token_id
    return sent

def calc_dlt(sent: Sent) -> Sent:
    # calculate DLT-related predictors
    def dist_pos(tokens: list[UDToken], pos: list[str]) -> int:
        return len([token for token in tokens if token.pos in pos])
    
    def dist_char(tokens: list[UDToken]) -> int:
        return sum([len(token.word) for token in tokens])

    for token in sent.tokens:
        deps = [t for t in sent.tokens if precede(t.token_id, token.token_id) and (t.head == token.token_id or t.token_id == token.head)]
        token.preds["dlt_count"] = len(deps)
        token.preds["dlt_integ_nv"] = sum([dist_pos(tokens_in_range(sent.tokens, t.token_id, token.token_id)[1:], ["NOUN","PROPN","VERB"]) for t in deps])
        token.preds["dlt_integ_nvjr"] = sum([dist_pos(tokens_in_range(sent.tokens, t.token_id, token.token_id)[1:], ["NOUN","PROPN","VERB","ADJ","ADV"]) for t in deps])
        token.preds["dlt_integ_char"] = sum([dist_char(tokens_in_range(sent.tokens, t.token_id, token.token_id)[1:]) for t in deps])
        token.preds["dlt_stor"] = len([t for t in sent.tokens if\
                                       (precede(t.token_id, token.token_id) and precede(token.token_id, t.head)) or\
                                       (precede("0", t.head) and precede(t.head, token.token_id) and precede(token.token_id, t.token_id))])
    return sent

def precede(a: str, b: str) -> bool:
    # compare two token IDs and returns if a precedes b
    try:
        a = float(a.split("-")[0]) if "-" in a else float(a)
        b = float(b.split("-")[0]) if "-" in b else float(b)
        return a < b
    except ValueError:
        raise ValueError(f"Non-parsable token ID: {a} or {b}")

def sort_ids(ids: tuple[str, str]) -> tuple[str, str]:
    # sort two IDs in order
    if precede(ids[0], ids[1]):
        return ids
    return (ids[1], ids[0])

def max_id(a: str, b:str) -> str:
    return sort_ids((a,b))[1]

def min_id(a: str, b:str) -> str:
    return sort_ids((a,b))[0]
    
def tokens_in_range(tokens: list[UDToken], id_from: str, id_to: str) -> list[UDToken]:
    # return the list of tokens in the specified range
    i = 0
    for j, token in enumerate(tokens):
        if token.token_id == id_from:
            i = j
        elif token.token_id == id_to:
            return tokens[i:j]

def fix_crossed(sent: Sent) -> Sent:
    # fix crossed dependencies in UD
    # if two dependencies h1-d1 and h2-d2 cross, and A is higher than B, then resolve it by attaching d2 to h1
    for d1 in sent.tokens:
        if d1.head == "0":
            for d2 in descendants(sent, d1.head):
                left, right = sort_ids((d2.token_id, d2.head))
                if precede(left, d1.token_id) and precede(d1.token_id, right):
                    d2.head = d1.token_id
        else:
            for d2 in descendants(sent, d1.head):
                left1, right1 = sort_ids((d1.token_id, d1.head))
                left2, right2 = sort_ids((d2.token_id, d2.head))
                if (((precede(left1, left2) and precede(right1, right2)) or
                    (precede(left2, left1) and precede(right2, right1))) and
                    precede(max_id(left1, left2), min_id(right1, right2))):
                    d2.head = d1.head
    return sent

def descendants(sent: Sent, head: int) -> list[UDToken]:
    # return tokens that are descendants of a given head
    clds = [token for token in sent.tokens if token.head == head]
    return clds + sum([descendants(sent, cld.token_id) for cld in clds], [])

@dataclass
class Tree:
    sent_id: str | None
    rule: str
    clds: list["Tree"]
    token_id: str | None = None
    word: str | None = None
    pos: str | None = None
    preds: dict[str, int] | None = None
    misc: str | None = None

# returns true if the head-dependency relation has to be reversed 
# this applies to modifiers and functional categories
def is_reverse(deptype: str) -> bool:
    lst = ["obl","vocative","dislocated","advcl","advmod","discourse","aux","cop","mark",
           "nmod","appos","nummod","acl","amod","det","clf","case","compound","punct","dep"]
    return (deptype.startswith(e) for e in lst)

def ud2tree(sent: Sent, is_headfinal: bool, head_id: int = None) -> Tree:
    # convert a UD sentence to a CCG tree
    if head_id is None:
        head = next((token for token in sent.tokens if token.head == "0"), sent.tokens[0])
        head_id = head.token_id
    else:
        head = next((token for token in sent.tokens if token.token_id == head_id), sent.tokens[0])
    tree = Tree(sent.sent_id, "lex", [], head.token_id, head.word, head.pos,
                head.preds | {"clt_integ_nv":0, "clt_integ_nvjr":0, "clt_integ_char":0, "clt_count":0}, head.misc)
    clds = [token for token in sent.tokens if token.head == head_id]
    left_clds = [cld for cld in clds if precede(cld.token_id, head_id)]
    right_clds = [cld for cld in clds if precede(head_id, cld.token_id)]
    if not is_headfinal:
        for cld in right_clds:
            rule = "<" if is_reverse(cld.deptype) else ">"
            tree = Tree(sent.sent_id, rule, [tree, ud2tree(sent, is_headfinal, cld.token_id)])
        for cld in reversed(left_clds):
            rule = ">" if is_reverse(cld.deptype) else "<"
            tree = Tree(sent.sent_id, rule, [ud2tree(sent, is_headfinal, cld.token_id), tree])
    else:
        for cld in reversed(left_clds):
            rule = ">" if is_reverse(cld.deptype) else "<"
            tree = Tree(sent.sent_id, rule, [ud2tree(sent, is_headfinal, cld.token_id), tree])
        for cld in right_clds:
            rule = "<" if is_reverse(cld.deptype) else ">"
            tree = Tree(sent.sent_id, rule, [tree, ud2tree(sent, is_headfinal, cld.token_id)])
    return tree

def save_tree(tree: Tree, path: str) -> None:
    # save the visualization of a CCG tree as a text file
    def to_str(tree: Tree) -> str:
        if tree.rule == "lex":
            return f"[{tree.pos} {tree.word}@{tree.preds['clt_integ_nv']}]"
        return f"[{tree.rule} {' '.join([to_str(cld) for cld in tree.clds])}]"
    
    t = nltkTree.fromstring(to_str(tree), brackets = "[]")
    with open(path, "w") as f:
        f.write(TreePrettyPrinter(t, None, ()).text())

# patterns of conversion (ENF top, ENF sub, INC top, INC sub, unary for children)
CONV_PATTERN = [(">",">",">",">B",None,None,None),
                (">","<",">",">B",None,">T",None),
                ("<",">",">",">B",">T",None,None),
                ("<","<",">",">B",">T",">T",None),
                (">B",">B",">B",">B",None,None,None)]


def enf2inc(tree: Tree) -> Tree:
    # convert an ENF CCG tree to an incremental one
    for pattern in CONV_PATTERN:
        if tree.rule == pattern[0] and tree.clds[1].rule == pattern[1]:
            cld_0 = tree.clds[0] if pattern[4] is None else Tree(tree.sent_id, pattern[4], [tree.clds[0]])
            cld_1 = tree.clds[1].clds[0] if pattern[5] is None else Tree(tree.sent_id, pattern[5], [tree.clds[1].clds[0]])
            cld_2 = tree.clds[1].clds[1] if pattern[6] is None else Tree(tree.sent_id, pattern[6], [tree.clds[1].clds[1]])
            subtree = Tree(tree.sent_id, pattern[3], [cld_0, cld_1])
            return enf2inc(Tree(tree.sent_id, pattern[2], [subtree, cld_2]))
    tree.clds = [enf2inc(cld) for cld in tree.clds]
    return tree

def calc_clt(tree: Tree) -> Tree:
    # calculate CLT-related predictors
    def rightmost_term(t: Tree) -> Tree:
        if not t.clds:
            return t
        return rightmost_term(t.clds[-1])

    if not tree.clds:
        return tree
    if len(tree.clds) == 1:
        tree.clds = [calc_clt(tree.clds[0])]
        return tree
    r = rightmost_term(tree)
    r.preds[f"clt_count"] += 1
    lr = rightmost_term(tree.clds[0])
    intervening_terminals = [t for t in terminals(tree) if precede(lr.token_id, t.token_id) and precede(t.token_id, r.token_id)]
    r.preds[f"clt_integ_nv"] += len([t for t in intervening_terminals if t.pos in ["NOUN","PROPN","VERB"]])
    r.preds[f"clt_integ_nvjr"] += len([t for t in intervening_terminals if t.pos in ["NOUN","PROPN","VERB","ADJ","ADV"]])
    r.preds[f"clt_integ_char"] += sum([len(t.word) for t in intervening_terminals])
    tree.clds = [calc_clt(cld) for cld in tree.clds]
    return tree

def calc_clt_stor(tree: Tree) -> Tree:
    clt_stor = 0
    for term in terminals(tree):
        clt_stor += 1
        clt_stor -= term.preds[f"clt_count"]
        term.preds[f"clt_stor"] = clt_stor
    return tree
    
def save_clt(trees: list[Tree], path: str) -> None:
    # save CLT costs as a tsv file
    pred_names = terminals(trees[0])[0].preds.keys()
    text = "sent_id\ttoken_id\tword\t" + "\t".join(pred_names) + "\tmisc"
    for tree in trees:
        for t in terminals(tree):
            text += f"\n{t.sent_id}\t{t.token_id}\t{t.word}"
            for pred_name in pred_names:
                text += f"\t{t.preds[pred_name]}"
            text += f"\t{t.misc}"
    with open(path, "w") as f:
        f.write(text)

def terminals(tree: Tree) -> list[Tree]:
    if not tree.clds:
        return [tree]
    return sum([terminals(cld) for cld in tree.clds], [])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path of the input UD file")
    parser.add_argument("--output", help="Path of the output CLT file", required=False)
    parser.add_argument("--headfinal", action="store_true", help="Set if the language is head-final", required=False)
    parser.add_argument("--savetrees", action="store_true", help="Set to visually check generated CCG trees", required=False)
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output if args.output else "clt.tsv"
    is_headfinal = args.headfinal
    save_trees = args.savetrees
    main(input_path, output_path, is_headfinal, save_trees)