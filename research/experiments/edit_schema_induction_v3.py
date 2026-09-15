from __future__ import annotations
import json, hashlib, time
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, deque
import numpy as np
import pandas as pd

REQ = {
    'arc-agi_training_challenges.json',
    'arc-agi_training_solutions.json',
    'arc-agi_evaluation_challenges.json',
    'arc-agi_evaluation_solutions.json',
    'arc-agi_test_challenges.json',
}


def data_dir():
    roots = [Path('/kaggle/input/competitions'), Path('/kaggle/input'), Path('.')]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob('arc-agi_test_challenges.json'):
            names = {p.name for p in f.parent.iterdir()}
            if REQ.issubset(names):
                hits.append(f.parent)
    if not hits:
        raise FileNotFoundError("Attach 'ARC Prize 2026 - ARC-AGI-2' as Kaggle input, then rerun.")
    return sorted(hits, key=lambda p: (0 if 'competitions' in str(p) else 1, len(str(p))))[0]


def load_arc():
    d = data_dir()
    load = lambda n: json.load(open(d / n))
    return d, load('arc-agi_training_challenges.json'), load('arc-agi_training_solutions.json')


def A(g):
    return np.asarray(g, dtype=np.int8)


def K(g):
    return tuple(map(tuple, A(g).tolist()))


def background(x):
    x = A(x)
    vals, cnt = np.unique(x, return_counts=True)
    return int(vals[np.argmax(cnt)])


@dataclass(frozen=True)
class Obj:
    pts: tuple
    color: int
    bbox: tuple
    size: int
    h: int
    w: int
    holes: int
    mask_key: tuple


def _holes(mask):
    h, w = mask.shape
    outside = np.zeros_like(mask, bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = 1; q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = 1; q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = 1; q.append((rr, cc))
    unseen = (~mask) & (~outside)
    seen = np.zeros_like(mask, bool)
    n = 0
    for r in range(h):
        for c in range(w):
            if unseen[r, c] and not seen[r, c]:
                n += 1
                qq = [(r, c)]; seen[r, c] = 1
                for rr, cc in qq:
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and unseen[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = 1; qq.append((nr, nc))
    return n


def objects(x):
    x = A(x); b = background(x); h, w = x.shape
    seen = np.zeros((h, w), bool); out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(x[r, c]) == b:
                continue
            col = int(x[r, c]); seen[r, c] = 1; q = [(r, c)]; pts = []
            for rr, cc in q:
                pts.append((rr, cc))
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(x[nr, nc]) == col:
                        seen[nr, nc] = 1; q.append((nr, nc))
            rs = [p[0] for p in pts]; cs = [p[1] for p in pts]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            mask = np.zeros((r1-r0+1, c1-c0+1), bool)
            for rr, cc in pts: mask[rr-r0, cc-c0] = 1
            out.append(Obj(tuple(sorted(pts)), col, (r0,c0,r1,c1), len(pts),
                           mask.shape[0], mask.shape[1], _holes(mask), K(mask.astype(np.int8))))
    return out


GEOMS = ('id','r90','r180','r270','flr','fud','tr','atr')


def geom(a, name):
    a = np.asarray(a)
    if name == 'id': return a.copy()
    if name == 'r90': return np.rot90(a, 1)
    if name == 'r180': return np.rot90(a, 2)
    if name == 'r270': return np.rot90(a, 3)
    if name == 'flr': return np.fliplr(a)
    if name == 'fud': return np.flipud(a)
    if name == 'tr': return a.T.copy()
    if name == 'atr': return np.rot90(a.T, 2)
    raise ValueError(name)


def obj_mask(o):
    return A(o.mask_key).astype(bool)


def selector_map(x):
    os = objects(x); out = {}
    if not os: return out
    def add(label, idxs):
        if len(idxs) == 1: out[label] = idxs[0]
    sizes = [o.size for o in os]; colors = [o.color for o in os]; holes = [o.holes for o in os]
    add('largest', [i for i,o in enumerate(os) if o.size == max(sizes)])
    add('smallest', [i for i,o in enumerate(os) if o.size == min(sizes)])
    add('top', [i for i,o in enumerate(os) if o.bbox[0] == min(z.bbox[0] for z in os)])
    add('bottom', [i for i,o in enumerate(os) if o.bbox[2] == max(z.bbox[2] for z in os)])
    add('left', [i for i,o in enumerate(os) if o.bbox[1] == min(z.bbox[1] for z in os)])
    add('right', [i for i,o in enumerate(os) if o.bbox[3] == max(z.bbox[3] for z in os)])
    add('most_holes', [i for i,o in enumerate(os) if o.holes == max(holes)])
    for s,c in Counter(sizes).items():
        if c == 1: add(f'unique_size:{s}', [i for i,o in enumerate(os) if o.size == s])
    for col,c in Counter(colors).items():
        if c == 1: add(f'unique_color:{col}', [i for i,o in enumerate(os) if o.color == col])
    return out


def selectors_for_index(x, idx):
    return tuple(sorted(k for k,v in selector_map(x).items() if v == idx))


def select_obj(x, selector):
    os = objects(x); idx = selector_map(x).get(selector)
    return None if idx is None or idx >= len(os) else os[idx]


def render_crop(x, o, out_color=None, g='id'):
    b = background(x); m = geom(obj_mask(o), g)
    y = np.full(m.shape, b, dtype=np.int8)
    y[m] = o.color if out_color is None else int(out_color)
    return y


def render_isolate(x, o, out_color=None):
    x = A(x); y = np.full_like(x, background(x))
    col = o.color if out_color is None else int(out_color)
    for r,c in o.pts: y[r,c] = col
    return y


def color_rules(x, src_idx, target_color):
    os = objects(x); src = os[src_idx]
    rules = {f'const:{int(target_color)}'}
    if int(target_color) == src.color: rules.add('same')
    for sel,j in selector_map(x).items():
        if j != src_idx and os[j].color == int(target_color): rules.add('refcolor:' + sel)
    return tuple(sorted(rules))


def resolve_color(x, src, rule):
    if rule == 'same': return src.color
    if rule.startswith('const:'): return int(rule.split(':',1)[1])
    if rule.startswith('refcolor:'):
        ref = select_obj(x, rule.split(':',1)[1])
        return None if ref is None else ref.color
    return None


def destination_rules(x, src_idx, target_top, target_hw):
    os = objects(x); src = os[src_idx]; sr, sc, _, _ = src.bbox
    tr, tc = target_top; th, tw = target_hw
    rules = {f'delta:{tr-sr}:{tc-sc}'}
    if (tr,tc) == (sr,sc): rules.add('same_pos')
    for sel,j in selector_map(x).items():
        if j == src_idx: continue
        r0,c0,r1,c1 = os[j].bbox
        if tr == r0:
            rules.add(f'right_of|{sel}|{tc-(c1+1)}|top')
            rules.add(f'left_of|{sel}|{c0-(tc+tw)}|top')
        if tc == c0:
            rules.add(f'below|{sel}|{tr-(r1+1)}|left')
            rules.add(f'above|{sel}|{r0-(tr+th)}|left')
    return tuple(sorted(rules))


def resolve_destination(x, src, rule, target_hw):
    sr, sc, _, _ = src.bbox; th, tw = target_hw
    if rule == 'same_pos': return sr, sc
    if rule.startswith('delta:'):
        _, dr, dc = rule.split(':'); return sr + int(dr), sc + int(dc)
    p = rule.split('|')
    if len(p) != 4: return None
    rel, sel, gap, align = p; ref = select_obj(x, sel)
    if ref is None: return None
    gap = int(gap); r0,c0,r1,c1 = ref.bbox
    if rel == 'right_of' and align == 'top': return r0, c1 + 1 + gap
    if rel == 'left_of' and align == 'top': return r0, c0 - tw - gap
    if rel == 'below' and align == 'left': return r1 + 1 + gap, c0
    if rel == 'above' and align == 'left': return r0 - th - gap, c0
    return None


def paint_shape(canvas, mask, top, color):
    y = canvas.copy(); r0,c0 = top; H,W = y.shape
    for rr,cc in np.argwhere(mask):
        r,c = r0 + int(rr), c0 + int(cc)
        if not (0 <= r < H and 0 <= c < W): return None
        y[r,c] = int(color)
    return y


def infer_cmap(src, dst):
    src, dst = A(src), A(dst)
    if src.shape != dst.shape: return None
    mp = {}
    for a,b in zip(src.flat, dst.flat):
        a,b = int(a),int(b)
        if a in mp and mp[a] != b: return None
        mp[a] = b
    return tuple(sorted(mp.items()))


@dataclass(frozen=True)
class Schema:
    kind: str
    selector: str = ''
    geom: str = 'id'
    color_rule: str = 'same'
    dest_rule: str = ''
    cmap: tuple = ()
    cost: float = 0.0
    def key(self): return (self.kind,self.selector,self.geom,self.color_rule,self.dest_rule,self.cmap)


def _cost(s):
    b = {'global':1.0,'crop':1.5,'isolate':1.8,'recolor':2.0,'move':2.5,'copy':2.7,'fill_holes':1.6}.get(s.kind,4.0)
    return b + .15*(s.geom!='id') + .25*s.color_rule.startswith('const:') + .2*s.dest_rule.startswith('delta:') + .15*len(s.cmap)


def apply_schema(s, x):
    x = A(x)
    if s.kind == 'global':
        y = geom(x, s.geom)
        if s.cmap:
            z = y.copy()
            for a,b in s.cmap: z[y == int(a)] = int(b)
            y = z
        return y
    if s.kind == 'fill_holes':
        y = x.copy()
        for o in objects(x):
            r0,c0,_,_ = o.bbox; m = obj_mask(o); h,w = m.shape
            outside = np.zeros_like(m,bool); q=deque()
            for r in range(h):
                for c in (0,w-1):
                    if not m[r,c] and not outside[r,c]: outside[r,c]=1;q.append((r,c))
            for c in range(w):
                for r in (0,h-1):
                    if not m[r,c] and not outside[r,c]: outside[r,c]=1;q.append((r,c))
            while q:
                r,c=q.popleft()
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    rr,cc=r+dr,c+dc
                    if 0<=rr<h and 0<=cc<w and not m[rr,cc] and not outside[rr,cc]: outside[rr,cc]=1;q.append((rr,cc))
            hole=(~m)&(~outside)
            if hole.any():
                rr,cc=np.where(hole); y[rr+r0,cc+c0]=o.color
        return y
    src = select_obj(x, s.selector)
    if src is None: return None
    col = resolve_color(x, src, s.color_rule)
    if col is None: return None
    if s.kind == 'crop': return render_crop(x, src, col, s.geom)
    if s.kind == 'isolate': return render_isolate(x, src, col) if s.geom == 'id' else None
    if s.kind == 'recolor':
        if s.geom != 'id': return None
        y=x.copy()
        for r,c in src.pts: y[r,c]=col
        return y
    if s.kind in ('move','copy'):
        mask=geom(obj_mask(src),s.geom); top=resolve_destination(x,src,s.dest_rule,mask.shape)
        if top is None: return None
        y=x.copy()
        if s.kind=='move':
            b=background(x)
            for r,c in src.pts: y[r,c]=b
        return paint_shape(y,mask,top,col)
    return None


def derive_demo_schemas(inp, out):
    x,y=A(inp),A(out); cand={}
    def add(s):
        try:
            p=apply_schema(s,x)
            if p is not None and np.array_equal(p,y):
                ss=Schema(s.kind,s.selector,s.geom,s.color_rule,s.dest_rule,s.cmap,_cost(s)); cand[ss.key()]=ss
        except Exception: pass
    for g in GEOMS:
        gx=geom(x,g)
        if gx.shape!=y.shape: continue
        if np.array_equal(gx,y): add(Schema('global',geom=g))
        mp=infer_cmap(gx,y)
        if mp is not None: add(Schema('global',geom=g,cmap=mp))
    add(Schema('fill_holes'))
    osx,osy=objects(x),objects(y)
    for i,src in enumerate(osx):
        sels=selectors_for_index(x,i)
        if not sels: continue
        for g in GEOMS:
            m=geom(obj_mask(src),g)
            if m.shape==y.shape:
                for tc in map(int, np.unique(y)):
                    for sel in sels:
                        for cr in color_rules(x,i,tc): add(Schema('crop',sel,g,cr))
        if y.shape==x.shape:
            srcmask=np.zeros_like(x,bool)
            for r,c in src.pts: srcmask[r,c]=1
            fg=y!=background(y)
            if np.array_equal(fg,srcmask):
                cols=np.unique(y[fg])
                if len(cols)==1:
                    for sel in sels:
                        for cr in color_rules(x,i,int(cols[0])): add(Schema('isolate',sel,'id',cr))
            diff=x!=y
            if diff.any() and np.all(~diff | srcmask):
                vals=np.unique(y[srcmask])
                if len(vals)==1:
                    for sel in sels:
                        for cr in color_rules(x,i,int(vals[0])): add(Schema('recolor',sel,'id',cr))
            for tgt in osy:
                for g in GEOMS:
                    sm=geom(obj_mask(src),g); tm=obj_mask(tgt)
                    if sm.shape!=tm.shape or not np.array_equal(sm,tm): continue
                    tr,tc,_,_=tgt.bbox
                    for sel in sels:
                        for cr in color_rules(x,i,tgt.color):
                            for dr in destination_rules(x,i,(tr,tc),sm.shape):
                                add(Schema('move',sel,g,cr,dr)); add(Schema('copy',sel,g,cr,dr))
    return sorted(cand.values(),key=lambda s:(s.cost,s.key()))


def induce_schemas(train):
    per=[derive_demo_schemas(e['input'],e['output']) for e in train]
    if not per: return [],per
    maps=[{s.key():s for s in xs} for xs in per]
    common=set(maps[0])
    for m in maps[1:]: common &= set(m)
    valid=[]
    for k in common:
        s=maps[0][k]
        if all((p:=apply_schema(s,e['input'])) is not None and np.array_equal(p,A(e['output'])) for e in train): valid.append(s)
    valid.sort(key=lambda s:(s.cost,s.key()))
    return valid,per


def predict(task,max_predictions=2):
    schemas,per=induce_schemas(task['train']); preds=[]; seen=set()
    for s in schemas:
        outs=[]; ok=True
        for te in task['test']:
            p=apply_schema(s,te['input'])
            if p is None: ok=False; break
            outs.append(p)
        if not ok: continue
        sig=tuple(K(p) for p in outs)
        if sig in seen: continue
        seen.add(sig); preds.append((s,outs))
        if len(preds)>=max_predictions: break
    return schemas,per,preds


def evaluate(TR,TS,ids,limit=50):
    rows=[]; chosen=ids if limit is None else ids[:limit]; t0=time.time()
    for n,tid in enumerate(chosen,1):
        task=TR[tid]; start=time.time(); schemas,per,preds=predict(task,2); truth=[A(g) for g in TS[tid]]
        exact=[]
        for _,outs in preds: exact.append(len(outs)==len(truth) and all(np.array_equal(a,b) for a,b in zip(outs,truth)))
        cnt=[len(x) for x in per]
        rows.append({'task':tid,'train_examples':len(task['train']),'test_outputs':len(task['test']),
                     'mean_demo_explanations':float(np.mean(cnt)) if cnt else 0.0,'anti_unified_schema_count':len(schemas),
                     'demo_coverage':bool(schemas),'attempts':len(preds),'test_exact_pass1':bool(exact[0]) if exact else False,
                     'test_exact_pass2':bool(any(exact[:2])),'best_schema':repr(preds[0][0]) if preds else '',
                     'seconds':time.time()-start})
        if n%10==0: print(f"{n}/{len(chosen)} coverage={sum(r['demo_coverage'] for r in rows)}/{n} pass2={sum(r['test_exact_pass2'] for r in rows)}/{n}")
    print('elapsed',time.time()-t0)
    return pd.DataFrame(rows)


def run(limit=50,split='heldout',out_dir='/kaggle/working/edit_schema_v3'):
    D,TR,TS=load_arc(); ids=sorted(TR)
    dev=[t for t in ids if int(hashlib.sha256(t.encode()).hexdigest(),16)%5 != 0]
    held=[t for t in ids if t not in set(dev)]; chosen=held if split=='heldout' else dev
    print('ARC data:',D); print('dev/held:',len(dev),len(held),'split=',split,'limit=',limit)
    df=evaluate(TR,TS,chosen,limit); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    path=out/f'edit_schema_v3_{split}.csv'; df.to_csv(path,index=False)
    summary={'split':split,'limit':limit,'tasks':int(len(df)),'demo_coverage_tasks':int(df.demo_coverage.sum()) if len(df) else 0,
             'pass1_tasks':int(df.test_exact_pass1.sum()) if len(df) else 0,'pass2_tasks':int(df.test_exact_pass2.sum()) if len(df) else 0,
             'median_anti_unified_schemas':float(df.anti_unified_schema_count.median()) if len(df) else 0,
             'median_seconds':float(df.seconds.median()) if len(df) else 0,'output_csv':str(path)}
    (out/f'edit_schema_v3_{split}_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    return df,summary
