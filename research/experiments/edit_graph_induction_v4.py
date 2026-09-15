from __future__ import annotations

import json, hashlib, time, itertools
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, deque, defaultdict
import numpy as np
import pandas as pd

REQ = {
    "arc-agi_training_challenges.json",
    "arc-agi_training_solutions.json",
    "arc-agi_evaluation_challenges.json",
    "arc-agi_evaluation_solutions.json",
    "arc-agi_test_challenges.json",
}

# ---------------- data ----------------
def data_dir():
    roots = [Path("/kaggle/input/competitions"), Path("/kaggle/input"), Path(".")]
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("arc-agi_test_challenges.json"):
            names = {p.name for p in f.parent.iterdir()}
            if REQ.issubset(names):
                hits.append(f.parent)
    if not hits:
        raise FileNotFoundError("Attach 'ARC Prize 2026 - ARC-AGI-2' as Kaggle input, then rerun.")
    return sorted(hits, key=lambda p: (0 if "competitions" in str(p) else 1, len(str(p))))[0]

def load_arc():
    d = data_dir()
    load = lambda n: json.load(open(d / n))
    return d, load("arc-agi_training_challenges.json"), load("arc-agi_training_solutions.json")

def A(g):
    return np.asarray(g, dtype=np.int8)

def K(g):
    return tuple(map(tuple, A(g).tolist()))

def background(x):
    x = A(x)
    vals, cnt = np.unique(x, return_counts=True)
    return int(vals[np.argmax(cnt)])

# ---------------- objects ----------------
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
                outside[r, c] = 1
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = 1
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr,cc] and not outside[rr,cc]:
                outside[rr,cc] = 1
                q.append((rr,cc))
    unseen = (~mask) & (~outside)
    seen = np.zeros_like(mask, bool)
    n = 0
    for r in range(h):
        for c in range(w):
            if unseen[r,c] and not seen[r,c]:
                n += 1
                qq = [(r,c)]
                seen[r,c] = 1
                for rr,cc in qq:
                    for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr,nc = rr+dr,cc+dc
                        if 0 <= nr < h and 0 <= nc < w and unseen[nr,nc] and not seen[nr,nc]:
                            seen[nr,nc] = 1
                            qq.append((nr,nc))
    return n

def objects(x, diag=False):
    x = A(x)
    b = background(x)
    h,w = x.shape
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    if diag:
        dirs += [(1,1),(1,-1),(-1,1),(-1,-1)]
    seen = np.zeros((h,w), bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r,c] or int(x[r,c]) == b:
                continue
            col = int(x[r,c])
            seen[r,c] = 1
            q = [(r,c)]
            pts = []
            for rr,cc in q:
                pts.append((rr,cc))
                for dr,dc in dirs:
                    nr,nc = rr+dr,cc+dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr,nc] and int(x[nr,nc]) == col:
                        seen[nr,nc] = 1
                        q.append((nr,nc))
            rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
            r0,r1,c0,c1=min(rs),max(rs),min(cs),max(cs)
            mask=np.zeros((r1-r0+1,c1-c0+1),bool)
            for rr,cc in pts:
                mask[rr-r0,cc-c0]=1
            out.append(Obj(tuple(sorted(pts)),col,(r0,c0,r1,c1),len(pts),
                           mask.shape[0],mask.shape[1],_holes(mask),K(mask.astype(np.int8))))
    return out

def obj_mask(o):
    return A(o.mask_key).astype(bool)

GEOMS = ("id","r90","r180","r270","flr","fud","tr","atr")

def geom(a, name):
    a=np.asarray(a)
    if name=="id": return a.copy()
    if name=="r90": return np.rot90(a,1)
    if name=="r180": return np.rot90(a,2)
    if name=="r270": return np.rot90(a,3)
    if name=="flr": return np.fliplr(a)
    if name=="fud": return np.flipud(a)
    if name=="tr": return a.T.copy()
    if name=="atr": return np.rot90(a.T,2)
    raise ValueError(name)

def center(o):
    r0,c0,r1,c1=o.bbox
    return ((r0+r1)/2.0,(c0+c1)/2.0)

def touches_border(o, shape):
    h,w=shape; r0,c0,r1,c1=o.bbox
    return r0==0 or c0==0 or r1==h-1 or c1==w-1

def role_map(x):
    """Role predicates deliberately avoid literal size/color values."""
    os=objects(x)
    roles=defaultdict(set)
    if not os:
        return roles
    sizes=[o.size for o in os]; colors=[o.color for o in os]; holes=[o.holes for o in os]
    shape_counts=Counter(o.mask_key for o in os)
    def mark(role, idxs):
        if len(idxs)==1:
            roles[idxs[0]].add(role)
    mark("largest",[i for i,o in enumerate(os) if o.size==max(sizes)])
    mark("smallest",[i for i,o in enumerate(os) if o.size==min(sizes)])
    mark("top",[i for i,o in enumerate(os) if o.bbox[0]==min(z.bbox[0] for z in os)])
    mark("bottom",[i for i,o in enumerate(os) if o.bbox[2]==max(z.bbox[2] for z in os)])
    mark("left",[i for i,o in enumerate(os) if o.bbox[1]==min(z.bbox[1] for z in os)])
    mark("right",[i for i,o in enumerate(os) if o.bbox[3]==max(z.bbox[3] for z in os)])
    mark("most_holes",[i for i,o in enumerate(os) if o.holes==max(holes)])
    uniq_size=[i for i,o in enumerate(os) if Counter(sizes)[o.size]==1]
    if len(uniq_size)==1: roles[uniq_size[0]].add("unique_size")
    uniq_color=[i for i,o in enumerate(os) if Counter(colors)[o.color]==1]
    if len(uniq_color)==1: roles[uniq_color[0]].add("unique_color")
    uniq_shape=[i for i,o in enumerate(os) if shape_counts[o.mask_key]==1]
    if len(uniq_shape)==1: roles[uniq_shape[0]].add("unique_shape")
    border=[i for i,o in enumerate(os) if touches_border(o,A(x).shape)]
    if len(border)==1: roles[border[0]].add("unique_border_touching")
    holey=[i for i,o in enumerate(os) if o.holes>0]
    if len(holey)==1: roles[holey[0]].add("unique_holey")
    if len(os)==1:
        roles[0].add("only_object")
    return roles

def select_role(x, role):
    rm=role_map(x)
    hits=[i for i,rs in rm.items() if role in rs]
    if len(hits)!=1:
        return None
    os=objects(x)
    return os[hits[0]]

# ---------------- correspondences ----------------
@dataclass(frozen=True)
class Corr:
    src_idx: int
    out_idx: int
    geom: str
    score: float

def correspondence_edges(inp,out):
    """Shape-compatible object correspondences. Lower score is better."""
    xi,yo=objects(inp),objects(out)
    edges=[]
    for i,a in enumerate(xi):
        am=obj_mask(a)
        ar,ac=center(a)
        for j,b in enumerate(yo):
            bm=obj_mask(b)
            br,bc=center(b)
            for g in GEOMS:
                gm=geom(am,g)
                if gm.shape==bm.shape and np.array_equal(gm,bm):
                    score=0.0
                    score += 0.0 if a.color==b.color else 0.35
                    score += 0.005*(abs(ar-br)+abs(ac-bc))
                    edges.append(Corr(i,j,g,float(score)))
    return sorted(edges,key=lambda e:(e.score,e.src_idx,e.out_idx,e.geom))

# ---------------- concrete edit explanations ----------------
@dataclass(frozen=True)
class Edit:
    kind: str
    src_idx: int = -1
    out_idx: int = -1
    geom: str = "id"
    color_mode: str = "none"
    color_value: int = -1
    color_ref_idx: int = -1
    dest_mode: str = "none"
    delta: tuple = ()
    rel: str = ""
    rel_ref_idx: int = -1
    gap: int = 0
    align: str = ""
    cmap: tuple = ()

def paint_shape(canvas,mask,top,color):
    y=canvas.copy(); r0,c0=top; H,W=y.shape
    for rr,cc in np.argwhere(mask):
        r,c=r0+int(rr),c0+int(cc)
        if not (0<=r<H and 0<=c<W):
            return None
        y[r,c]=int(color)
    return y

def apply_concrete(e,x):
    x=A(x); os=objects(x)
    if e.kind=="global":
        y=geom(x,e.geom)
        if e.cmap:
            z=y.copy()
            for a,b in e.cmap: z[y==int(a)]=int(b)
            y=z
        return y
    if e.kind=="fill_holes":
        y=x.copy()
        for o in os:
            r0,c0,_,_=o.bbox; m=obj_mask(o); h,w=m.shape
            outside=np.zeros_like(m,bool); q=deque()
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
                    if 0<=rr<h and 0<=cc<w and not m[rr,cc] and not outside[rr,cc]:
                        outside[rr,cc]=1;q.append((rr,cc))
            hole=(~m)&(~outside)
            if hole.any():
                rr,cc=np.where(hole); y[rr+r0,cc+c0]=o.color
        return y
    if e.src_idx<0 or e.src_idx>=len(os): return None
    src=os[e.src_idx]
    if e.color_mode=="same": col=src.color
    elif e.color_mode=="const": col=e.color_value
    elif e.color_mode=="refcolor":
        if e.color_ref_idx<0 or e.color_ref_idx>=len(os): return None
        col=os[e.color_ref_idx].color
    else: col=src.color
    if e.kind=="crop":
        m=geom(obj_mask(src),e.geom)
        y=np.full(m.shape,background(x),dtype=np.int8); y[m]=col; return y
    if e.kind=="isolate":
        if e.geom!="id": return None
        y=np.full_like(x,background(x))
        for r,c in src.pts: y[r,c]=col
        return y
    if e.kind=="recolor":
        y=x.copy()
        for r,c in src.pts: y[r,c]=col
        return y
    if e.kind=="delete":
        y=x.copy()
        for r,c in src.pts: y[r,c]=background(x)
        return y
    if e.kind in ("move","copy"):
        m=geom(obj_mask(src),e.geom)
        sr,sc,_,_=src.bbox
        if e.dest_mode=="same": top=(sr,sc)
        elif e.dest_mode=="delta": top=(sr+e.delta[0],sc+e.delta[1])
        elif e.dest_mode=="relative":
            if e.rel_ref_idx<0 or e.rel_ref_idx>=len(os): return None
            ref=os[e.rel_ref_idx]; r0,c0,r1,c1=ref.bbox; th,tw=m.shape
            if e.rel=="right_of" and e.align=="top": top=(r0,c1+1+e.gap)
            elif e.rel=="left_of" and e.align=="top": top=(r0,c0-tw-e.gap)
            elif e.rel=="below" and e.align=="left": top=(r1+1+e.gap,c0)
            elif e.rel=="above" and e.align=="left": top=(r0-th-e.gap,c0)
            else: return None
        else: return None
        y=x.copy()
        if e.kind=="move":
            for r,c in src.pts: y[r,c]=background(x)
        return paint_shape(y,m,top,col)
    return None

def infer_cmap(src,dst):
    src,dst=A(src),A(dst)
    if src.shape!=dst.shape: return None
    mp={}
    for a,b in zip(src.flat,dst.flat):
        a,b=int(a),int(b)
        if a in mp and mp[a]!=b: return None
        mp[a]=b
    return tuple(sorted(mp.items()))

def color_modes(inp,src_idx,target_color):
    os=objects(inp); src=os[src_idx]
    out=[]
    if src.color==int(target_color):
        out.append(("same",-1,-1))
    out.append(("const",int(target_color),-1))
    for j,o in enumerate(os):
        if j!=src_idx and o.color==int(target_color):
            out.append(("refcolor",-1,j))
    return list(dict.fromkeys(out))

def destination_modes(inp,src_idx,target_top,target_hw):
    os=objects(inp); src=os[src_idx]; sr,sc,_,_=src.bbox
    tr,tc=target_top; th,tw=target_hw
    out=[]
    if (tr,tc)==(sr,sc): out.append(("same",(), "",-1,0,""))
    out.append(("delta",(tr-sr,tc-sc),"",-1,0,""))
    for j,ref in enumerate(os):
        if j==src_idx: continue
        r0,c0,r1,c1=ref.bbox
        if tr==r0:
            out.append(("relative",(),"right_of",j,tc-(c1+1),"top"))
            out.append(("relative",(),"left_of",j,c0-(tc+tw),"top"))
        if tc==c0:
            out.append(("relative",(),"below",j,tr-(r1+1),"left"))
            out.append(("relative",(),"above",j,r0-(tr+th),"left"))
    return list(dict.fromkeys(out))

def derive_edit_graphs(inp,out,max_edits=300):
    """Derive concrete edit explanations from the observed pair, then verify them exactly."""
    x,y=A(inp),A(out); cand={}
    def add(e):
        if len(cand)>=max_edits: return
        try:
            p=apply_concrete(e,x)
            if p is not None and np.array_equal(p,y):
                cand[e]=e
        except Exception:
            pass
    for g in GEOMS:
        gx=geom(x,g)
        if gx.shape!=y.shape: continue
        if np.array_equal(gx,y): add(Edit("global",geom=g))
        mp=infer_cmap(gx,y)
        if mp is not None: add(Edit("global",geom=g,cmap=mp))
    add(Edit("fill_holes"))
    osx,osy=objects(x),objects(y)
    edges=correspondence_edges(x,y)
    for edge in edges:
        i,j,g=edge.src_idx,edge.out_idx,edge.geom
        src,tgt=osx[i],osy[j]
        for cm,cv,cr in color_modes(x,i,tgt.color):
            if geom(obj_mask(src),g).shape==y.shape:
                add(Edit("crop",i,j,g,cm,cv,cr))
            if y.shape==x.shape:
                srcmask=np.zeros_like(x,bool)
                for r,c in src.pts: srcmask[r,c]=1
                fg=y!=background(y)
                if np.array_equal(fg,srcmask) and g=="id":
                    add(Edit("isolate",i,j,g,cm,cv,cr))
                diff=x!=y
                if diff.any() and np.all(~diff|srcmask) and g=="id":
                    add(Edit("recolor",i,j,g,cm,cv,cr))
                tr,tc,_,_=tgt.bbox; sm=geom(obj_mask(src),g)
                for dm,delta,rel,rj,gap,align in destination_modes(x,i,(tr,tc),sm.shape):
                    add(Edit("move",i,j,g,cm,cv,cr,dm,delta,rel,rj,gap,align))
                    add(Edit("copy",i,j,g,cm,cv,cr,dm,delta,rel,rj,gap,align))
    if y.shape==x.shape:
        for i,src in enumerate(osx):
            add(Edit("delete",i))
    return list(cand.values()), edges

# ---------------- anti-unification ----------------
@dataclass(frozen=True)
class Schema:
    kind: str
    src_role: str = ""
    geom: str = "id"
    color_mode: str = "none"
    color_value: int = -1
    color_ref_role: str = ""
    dest_mode: str = "none"
    delta: tuple = ()
    rel: str = ""
    rel_ref_role: str = ""
    gap: int = 0
    align: str = ""
    cmap: tuple = ()

def edit_skeleton(e):
    return (e.kind,e.geom,e.color_mode,e.dest_mode,e.rel,e.gap,e.align)

def source_roles(inp,e):
    if e.src_idx<0: return {""}
    return set(role_map(inp).get(e.src_idx,set()))

def color_ref_roles(inp,e):
    if e.color_mode!="refcolor": return {""}
    return set(role_map(inp).get(e.color_ref_idx,set()))

def dest_ref_roles(inp,e):
    if e.dest_mode!="relative": return {""}
    return set(role_map(inp).get(e.rel_ref_idx,set()))

def _compatible_constants(edits):
    if edits[0].color_mode=="const" and len({e.color_value for e in edits})!=1: return False
    if edits[0].dest_mode=="delta" and len({e.delta for e in edits})!=1: return False
    if edits[0].kind=="global" and len({e.cmap for e in edits})!=1: return False
    return True

def anti_unify_combo(train, edits):
    """Least-general executable schemas for one concrete explanation per demonstration."""
    if len(edits)!=len(train): return []
    if len({edit_skeleton(e) for e in edits})!=1: return []
    if not _compatible_constants(edits): return []
    e0=edits[0]
    if e0.kind in ("global","fill_holes"):
        return [Schema(e0.kind,geom=e0.geom,cmap=e0.cmap)]
    src_common=None
    for ex,e in zip(train,edits):
        rs=source_roles(ex["input"],e)
        src_common=rs if src_common is None else src_common & rs
    if not src_common: return []
    color_ref_common={""}
    if e0.color_mode=="refcolor":
        color_ref_common=None
        for ex,e in zip(train,edits):
            rs=color_ref_roles(ex["input"],e)
            color_ref_common=rs if color_ref_common is None else color_ref_common & rs
        if not color_ref_common: return []
    dest_ref_common={""}
    if e0.dest_mode=="relative":
        dest_ref_common=None
        for ex,e in zip(train,edits):
            rs=dest_ref_roles(ex["input"],e)
            dest_ref_common=rs if dest_ref_common is None else dest_ref_common & rs
        if not dest_ref_common: return []
    out=[]
    for sr in sorted(src_common):
        for cr in sorted(color_ref_common):
            for dr in sorted(dest_ref_common):
                out.append(Schema(
                    kind=e0.kind,src_role=sr,geom=e0.geom,
                    color_mode=e0.color_mode,
                    color_value=e0.color_value if e0.color_mode=="const" else -1,
                    color_ref_role=cr if e0.color_mode=="refcolor" else "",
                    dest_mode=e0.dest_mode,
                    delta=e0.delta if e0.dest_mode=="delta" else (),
                    rel=e0.rel if e0.dest_mode=="relative" else "",
                    rel_ref_role=dr if e0.dest_mode=="relative" else "",
                    gap=e0.gap if e0.dest_mode=="relative" else 0,
                    align=e0.align if e0.dest_mode=="relative" else "",
                ))
    return out

def apply_schema(s,x):
    x=A(x)
    if s.kind=="global":
        y=geom(x,s.geom)
        if s.cmap:
            z=y.copy()
            for a,b in s.cmap: z[y==int(a)]=int(b)
            y=z
        return y
    if s.kind=="fill_holes":
        return apply_concrete(Edit("fill_holes"),x)
    src=select_role(x,s.src_role)
    if src is None: return None
    if s.color_mode=="same": col=src.color
    elif s.color_mode=="const": col=s.color_value
    elif s.color_mode=="refcolor":
        ref=select_role(x,s.color_ref_role)
        if ref is None: return None
        col=ref.color
    else: col=src.color
    if s.kind=="crop":
        m=geom(obj_mask(src),s.geom); y=np.full(m.shape,background(x),dtype=np.int8); y[m]=col; return y
    if s.kind=="isolate":
        y=np.full_like(x,background(x))
        for r,c in src.pts: y[r,c]=col
        return y
    if s.kind=="recolor":
        y=x.copy()
        for r,c in src.pts: y[r,c]=col
        return y
    if s.kind=="delete":
        y=x.copy()
        for r,c in src.pts: y[r,c]=background(x)
        return y
    if s.kind in ("move","copy"):
        m=geom(obj_mask(src),s.geom); sr,sc,_,_=src.bbox
        if s.dest_mode=="same": top=(sr,sc)
        elif s.dest_mode=="delta": top=(sr+s.delta[0],sc+s.delta[1])
        elif s.dest_mode=="relative":
            ref=select_role(x,s.rel_ref_role)
            if ref is None: return None
            r0,c0,r1,c1=ref.bbox; th,tw=m.shape
            if s.rel=="right_of" and s.align=="top": top=(r0,c1+1+s.gap)
            elif s.rel=="left_of" and s.align=="top": top=(r0,c0-tw-s.gap)
            elif s.rel=="below" and s.align=="left": top=(r1+1+s.gap,c0)
            elif s.rel=="above" and s.align=="left": top=(r0-th-s.gap,c0)
            else: return None
        else: return None
        y=x.copy()
        if s.kind=="move":
            for r,c in src.pts: y[r,c]=background(x)
        return paint_shape(y,m,top,col)
    return None

def anti_unify_task(train, per_demo, combo_cap=30000):
    """Search explanation combinations by structural skeleton, then anti-unify object identities into common roles."""
    if not train or any(len(xs)==0 for xs in per_demo):
        return []
    by_demo=[]
    for xs in per_demo:
        d=defaultdict(list)
        for e in xs: d[edit_skeleton(e)].append(e)
        by_demo.append(d)
    common=set(by_demo[0])
    for d in by_demo[1:]: common &= set(d)
    schemas={}; combos_seen=0
    for sk in sorted(common,key=str):
        lists=[d[sk] for d in by_demo]
        for combo in itertools.product(*lists):
            combos_seen += 1
            if combos_seen>combo_cap:
                break
            for s in anti_unify_combo(train,combo):
                ok=True
                for ex in train:
                    p=apply_schema(s,ex["input"])
                    if p is None or not np.array_equal(p,A(ex["output"])):
                        ok=False; break
                if ok: schemas[s]=s
        if combos_seen>combo_cap:
            break
    return list(schemas.values())

# ---------------- evaluation ----------------
def predict(task,max_predictions=2):
    per=[]; corr_counts=[]
    for ex in task["train"]:
        edits,edges=derive_edit_graphs(ex["input"],ex["output"])
        per.append(edits); corr_counts.append(len(edges))
    schemas=anti_unify_task(task["train"],per)
    preds=[]; seen=set()
    for s in schemas:
        outs=[]; ok=True
        for te in task["test"]:
            p=apply_schema(s,te["input"])
            if p is None: ok=False; break
            outs.append(p)
        if not ok: continue
        sig=tuple(K(p) for p in outs)
        if sig in seen: continue
        seen.add(sig); preds.append((s,outs))
        if len(preds)>=max_predictions: break
    return per,corr_counts,schemas,preds

def evaluate(TR,TS,ids,limit=20):
    rows=[]; pair_rows=[]
    chosen=ids if limit is None else ids[:limit]
    for n,tid in enumerate(chosen,1):
        start=time.time(); task=TR[tid]
        per,corr_counts,schemas,preds=predict(task,2)
        truth=[A(g) for g in TS[tid]]
        exact=[]
        for _,outs in preds:
            exact.append(len(outs)==len(truth) and all(np.array_equal(a,b) for a,b in zip(outs,truth)))
        pair_ok=[len(xs)>0 for xs in per]
        for k,(xs,cc) in enumerate(zip(per,corr_counts)):
            pair_rows.append({
                "task":tid,"train_example":k,"correspondence_edges":cc,
                "edit_explanations":len(xs),"pair_explainable":bool(xs),
                "edit_kinds":"|".join(sorted(set(e.kind for e in xs))),
            })
        rows.append({
            "task":tid,
            "train_examples":len(task["train"]),
            "pair_explainable_count":int(sum(pair_ok)),
            "pair_explanation_rate":float(np.mean(pair_ok)) if pair_ok else 0.0,
            "mean_edit_explanations":float(np.mean([len(xs) for xs in per])) if per else 0.0,
            "mean_correspondence_edges":float(np.mean(corr_counts)) if corr_counts else 0.0,
            "anti_unified_schema_count":len(schemas),
            "train_reconstruction":bool(schemas),
            "attempts":len(preds),
            "test_exact_pass1":bool(exact[0]) if exact else False,
            "test_exact_pass2":bool(any(exact[:2])),
            "best_schema":repr(preds[0][0]) if preds else "",
            "seconds":time.time()-start,
        })
        if n%5==0:
            print(f"{n}/{len(chosen)} pair-explain={sum(r['pair_explainable_count'] for r in rows)}/"
                  f"{sum(r['train_examples'] for r in rows)} train-recon={sum(r['train_reconstruction'] for r in rows)}/{n} "
                  f"pass2={sum(r['test_exact_pass2'] for r in rows)}/{n}")
    return pd.DataFrame(rows),pd.DataFrame(pair_rows)

def run(limit=20,split="dev",out_dir="/kaggle/working/edit_graph_v4"):
    D,TR,TS=load_arc(); ids=sorted(TR)
    dev=[t for t in ids if int(hashlib.sha256(t.encode()).hexdigest(),16)%5 != 0]
    held=[t for t in ids if t not in set(dev)]
    chosen=dev if split=="dev" else held
    print("ARC data:",D)
    print("dev/held:",len(dev),len(held),"split=",split,"limit=",limit)
    df,pairs=evaluate(TR,TS,chosen,limit)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    task_path=out/f"edit_graph_v4_{split}.csv"
    pair_path=out/f"edit_graph_v4_{split}_pairs.csv"
    df.to_csv(task_path,index=False); pairs.to_csv(pair_path,index=False)

    total_pairs=int(len(pairs))
    pair_explain=int(pairs.pair_explainable.sum()) if total_pairs else 0
    pair_rate=float(pair_explain/total_pairs) if total_pairs else 0.0
    train_recon=int(df.train_reconstruction.sum()) if len(df) else 0
    train_rate=float(train_recon/len(df)) if len(df) else 0.0

    gate_pair = pair_rate >= 0.30
    gate_task = train_rate >= 0.20
    decision = (
        "CONTINUE_TO_CONTROLLED_ABLATION" if gate_pair and gate_task
        else "STOP_SYMBOLIC_BRANCH_PAIR_EXPLANATION_TOO_LOW" if not gate_pair
        else "REVISE_ANTI_UNIFICATION"
    )
    summary={
        "version":4,
        "split":split,
        "limit":limit,
        "tasks":int(len(df)),
        "training_pairs":total_pairs,
        "pair_explainable":pair_explain,
        "pair_explanation_rate":pair_rate,
        "tasks_with_train_reconstructing_schema":train_recon,
        "task_train_reconstruction_rate":train_rate,
        "pass1_tasks":int(df.test_exact_pass1.sum()) if len(df) else 0,
        "pass2_tasks":int(df.test_exact_pass2.sum()) if len(df) else 0,
        "median_anti_unified_schemas":float(df.anti_unified_schema_count.median()) if len(df) else 0.0,
        "median_seconds_per_task":float(df.seconds.median()) if len(df) else 0.0,
        "kill_criteria":{
            "pair_explanation_rate_min":0.30,
            "task_train_reconstruction_rate_min":0.20,
        },
        "decision":decision,
        "task_csv":str(task_path),
        "pair_csv":str(pair_path),
    }
    summary_path=out/f"edit_graph_v4_{split}_summary.json"
    summary_path.write_text(json.dumps(summary,indent=2))
    print("\n=== V4 STAGED RESULT ===")
    print(json.dumps(summary,indent=2))
    return df,pairs,summary

if __name__=="__main__":
    run()
