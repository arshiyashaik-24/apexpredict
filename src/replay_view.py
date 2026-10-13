"""Animated race replay (SVG + vanilla JS) built from the simulator's recorded races.

The browser animates one *simulated* race using the lap-by-lap times the simulator
produced (positions between lap boundaries are linearly interpolated). It is never
a live or real race. Animation runs entirely client-side: no Streamlit reruns per frame.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

TEAM_COLOURS = {"red bull": "#3671C6", "racing bulls": "#6692FF", "rb": "#6692FF", "mercedes": "#27F4D2",
                "ferrari": "#E8002D", "mclaren": "#FF8000", "aston": "#229971", "alpine": "#FF87BC",
                "williams": "#64C4FF", "haas": "#B6BABD", "audi": "#B01E4F", "sauber": "#52E252",
                "cadillac": "#C8A24B"}
FALLBACK = ["#E10600", "#4DA3FF", "#F5C518", "#7BE495", "#C77DFF", "#FF9F68", "#9AA0AA", "#FF6FB5"]
COMPOUNDS = {0: ("SOFT", "#FF3B3B"), 1: ("MEDIUM", "#FFD22E"), 2: ("HARD", "#EDEDED")}


def team_colour(team: str, idx: int = 0) -> str:
    t = str(team).lower()
    for k, v in TEAM_COLOURS.items():
        if k in t:
            return v
    return FALLBACK[idx % len(FALLBACK)]


def replay_options(res) -> list[str]:
    """Labels for the recorded races, e.g. 'Race 3 - winner VER'."""
    if not res.replay:
        return []
    pos = res.replay["positions"]
    return [f"Race {k + 1} - winner {res.drivers[int(np.argmin(pos[k]))]}" for k in range(len(pos))]


def stint_table(res, k: int) -> pd.DataFrame:
    """Tyre stints (driver, start lap, laps, compound) for recorded race k, ordered by finishing position."""
    rp = res.replay
    comp, retire, pos = rp["comp"][k], rp["retire"][k], rp["positions"][k]
    rows = []
    for i, d in enumerate(res.drivers):
        last = int(retire[i]) - 1 if retire[i] else comp.shape[0]
        start = 0
        for l in range(1, last + 1):
            if l == last or comp[l, i] != comp[l - 1, i]:
                rows.append((d, start + 1, l - start, COMPOUNDS[int(comp[start, i])][0], int(pos[i])))
                start = l
    return pd.DataFrame(rows, columns=["driver", "start", "laps", "compound", "finish"]).sort_values(["finish", "start"])


def build_payload(res, inputs: pd.DataFrame, k: int) -> dict:
    rp = res.replay
    summ = res.summary.set_index("driver")
    drivers = res.drivers
    seen, cols, alt = {}, [], []
    for i, tm in enumerate(inputs["team"]):
        seen[tm] = seen.get(tm, 0) + 1
        cols.append(team_colour(tm, len(seen)))
        alt.append(seen[tm] > 1)
    pos = rp["positions"][k]
    return {
        "drivers": drivers, "teams": inputs["team"].tolist(), "colours": cols, "alt": alt, "L": int(rp["t"].shape[1] - 1),
        "probs": {d: [float(summ.loc[d, "win_p"]), float(summ.loc[d, "podium_p"]), float(summ.loc[d, "exp_pos"])] for d in drivers},
        "race": {"t": np.round(rp["t"][k], 2).tolist(), "comp": rp["comp"][k].tolist(),
                 "pit": rp["pit"][k].astype(int).tolist(), "status": rp["status"][k].tolist(),
                 "retire": rp["retire"][k].tolist(), "grid": rp["grid"][k].tolist(), "wet": bool(rp["wet"][k]),
                 "final": np.argsort(pos).tolist(), "label": replay_options(res)[k]},
    }


def build_html(res, inputs: pd.DataFrame, k: int, title: str = "") -> str:
    payload = build_payload(res, inputs, k)
    fin = payload["race"]["final"]
    rows = "".join(f"<tr><td>{n + 1}</td><td>{payload['drivers'][i]}</td><td>{payload['teams'][i]}</td></tr>"
                   for n, i in enumerate(fin))
    html = TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":"))).replace("__FALLBACK_ROWS__", rows)
    return html.replace("__TITLE__", title)


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><style>
:root{--bg:#0A0B0D;--panel:#14161A;--panel2:#1A1D22;--line:#262A31;--red:#E10600;--txt:#F4F5F7;--mut:#8B919C;--ok:#3DDC84;--amb:#FFB020}
*{box-sizing:border-box}html,body{margin:0;background:transparent;color:var(--txt);font-family:"Titillium Web","Segoe UI",system-ui,sans-serif;overflow:hidden}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:14px;height:790px}
@media(max-width:900px){.wrap{grid-template-columns:1fr;height:auto}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;position:relative;overflow:hidden}
#trackcard{display:flex;flex-direction:column;min-height:560px}
.hud{position:absolute;left:0;right:0;top:0;display:flex;justify-content:space-between;align-items:flex-start;padding:14px 18px;pointer-events:none;z-index:2}
.lap{font-size:44px;font-weight:700;line-height:1;letter-spacing:.02em}.lap small{font-size:18px;color:var(--mut);font-weight:600}
.lap em{display:block;font-style:normal;font-size:11px;letter-spacing:.25em;color:var(--red);margin-bottom:4px}
.badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.b{font-size:11px;letter-spacing:.14em;padding:5px 10px;border:1px solid var(--line);border-radius:3px;background:rgba(10,11,13,.75);font-weight:600;display:flex;align-items:center;gap:7px}
.b i{width:8px;height:8px;border-radius:50%;background:var(--ok);display:inline-block;animation:pulse 1.6s infinite}
.b.sim{border-color:var(--red);color:#fff}.b.sim i{background:var(--red)}
.b.sc i{background:var(--amb);animation-duration:.6s}.b.sc{border-color:var(--amb);color:var(--amb)}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(61,220,132,.5)}50%{opacity:.5;box-shadow:0 0 0 5px rgba(61,220,132,0)}}
#trk{width:100%;flex:1;min-height:0;display:block}
#feed{position:absolute;left:18px;bottom:78px;font-size:12px;color:var(--mut);pointer-events:none;z-index:2}
#feed div{margin-top:3px;animation:fade 6s forwards}@keyframes fade{0%{opacity:0;transform:translateX(-8px)}8%{opacity:1;transform:none}80%{opacity:1}100%{opacity:0}}
#feed b{color:var(--txt)}
.ctrl{display:flex;align-items:center;gap:10px;padding:12px 16px;border-top:1px solid var(--line);background:var(--panel2);flex-wrap:wrap}
button{background:#1f2329;color:var(--txt);border:1px solid var(--line);border-radius:5px;padding:7px 13px;font:600 12px inherit;letter-spacing:.08em;cursor:pointer;transition:.15s}
button:hover{border-color:var(--red);box-shadow:0 0 12px rgba(225,6,0,.3)}button.on{background:var(--red);border-color:var(--red)}
#play{min-width:78px}label{font-size:12px;color:var(--mut);display:flex;align-items:center;gap:5px}
#bar{flex:1;min-width:180px;height:10px;background:#0d0e11;border:1px solid var(--line);border-radius:5px;position:relative;cursor:pointer;overflow:hidden}
#fill{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--red);box-shadow:0 0 10px rgba(225,6,0,.6)}
#bar .seg{position:absolute;top:0;bottom:0;background:rgba(255,176,32,.55)}#bar .seg.v{background:rgba(255,176,32,.25)}
#clock{font-size:12px;color:var(--mut);min-width:64px;text-align:right;font-variant-numeric:tabular-nums}
.side{display:flex;flex-direction:column;gap:14px;min-height:0}
.thead{display:grid;grid-template-columns:32px 6px 1fr 34px 72px;gap:8px;padding:9px 12px;font-size:10px;letter-spacing:.18em;color:var(--mut);border-bottom:1px solid var(--line)}
#rows{position:relative;margin:4px 0}
.row{position:absolute;left:0;right:0;height:24px;display:grid;grid-template-columns:32px 6px 1fr 34px 72px;gap:8px;align-items:center;padding:0 12px;font-size:13px;font-weight:600;cursor:pointer;transition:transform .45s cubic-bezier(.3,.8,.3,1),background .2s;font-variant-numeric:tabular-nums}
.row:hover{background:#20242b}.row.sel{background:#2a1517;box-shadow:inset 3px 0 0 var(--red)}
.row .p{color:var(--mut);text-align:right}.row .bar{height:16px;border-radius:2px}.row .g{text-align:right;color:var(--mut);font-size:12px}
.row .t{width:18px;height:18px;border-radius:50%;border:2px solid;font-size:9px;display:grid;place-items:center;color:#fff;justify-self:center}
.row.out{opacity:.4}.row.up{animation:up .9s}.row.dn{animation:dn .9s}
@keyframes up{0%,60%{background:rgba(61,220,132,.35)}100%{background:transparent}}@keyframes dn{0%,60%{background:rgba(225,6,0,.25)}100%{background:transparent}}
#detail{padding:14px 16px;flex:1;font-size:13px}
#detail h3{margin:0;font-size:26px;letter-spacing:.04em;display:flex;align-items:center;gap:10px}#detail h3 span{width:10px;height:26px;border-radius:2px}
#detail .sub{color:var(--mut);font-size:12px;margin-bottom:10px}
.kv{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px}.kv div{border-top:1px solid var(--line);padding-top:6px}
.kv small{display:block;color:var(--mut);font-size:10px;letter-spacing:.14em}.kv b{font-size:16px}
.ring{fill:none;stroke:#fff;stroke-width:2;transform-box:fill-box;transform-origin:center;animation:ring .9s ease-out forwards}
@keyframes ring{from{transform:scale(1);opacity:.9}to{transform:scale(4);opacity:0}}
.car{cursor:pointer;transition:opacity .3s}.car:hover circle.m{stroke:#fff;stroke-width:2}
#fallback{display:none;padding:20px}#fallback table{border-collapse:collapse;width:100%}#fallback td{padding:4px 8px;border-bottom:1px solid var(--line)}
</style></head><body>
<noscript><div style="padding:20px">JavaScript is disabled: the animated replay needs it. Final simulated classification is below.</div></noscript>
<div id="fallback"><b>Animation unavailable in this browser - final classification of the simulated race:</b><table>__FALLBACK_ROWS__</table></div>
<div class="wrap" id="app">
 <div class="card" id="trackcard">
  <div class="hud"><div class="lap"><em>LAP</em><span id="lapNow">0</span> <small>/ <span id="lapTot"></span></small></div>
   <div class="badges"><span class="b sim"><i></i>SIMULATED REPLAY - NOT A REAL RACE</span><span class="b" id="flag"><i></i>GREEN</span><span class="b" id="wx">DRY</span></div></div>
  <svg id="trk" viewBox="0 0 1000 620" preserveAspectRatio="xMidYMid meet">
   <defs><filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6"/></filter></defs>
   <path id="glowp" fill="none" stroke="#E10600" stroke-opacity=".28" stroke-width="30" filter="url(#glow)"/>
   <path id="edge" fill="none" stroke="#2c3038" stroke-width="30" stroke-linejoin="round"/>
   <path id="trkpath" fill="none" stroke="#14161a" stroke-width="22" stroke-linejoin="round"/>
   <path id="centre" fill="none" stroke="#fff" stroke-opacity=".12" stroke-width="1.5" stroke-dasharray="3 9"/>
   <line id="sf" stroke="#fff" stroke-width="3"/>
   <text x="985" y="605" fill="#555b66" font-size="11" text-anchor="end">Stylised schematic of Circuit of The Americas - not to scale</text>
   <g id="trails"></g><g id="rings"></g><g id="cars"></g>
  </svg>
  <div id="feed"></div>
  <div class="ctrl"><button id="play">PAUSE</button><button id="restart">RESTART</button>
   <span id="speeds"></span><label><input type="checkbox" id="trailChk" checked> Trails</label>
   <div id="bar" title="Click to seek"><div id="fill"></div></div><span id="clock">0:00</span></div>
 </div>
 <div class="side">
  <div class="card"><div class="thead"><span>POS</span><span></span><span>DRIVER</span><span>TYRE</span><span style="text-align:right">GAP</span></div><div id="rows"></div></div>
  <div class="card" id="detail"></div>
 </div>
</div>
<script>
const D=__DATA__;
(function(){try{
const N=D.drivers.length,L=D.L,R=D.race,T=R.t,CMP=R.comp,PIT=R.pit,ST=R.status,RET=R.retire,RH=24;
const COMP=R.wet?[{l:'I',c:'#3FA9F5',n:'INTER'}]:[{l:'S',c:'#FF3B3B',n:'SOFT'},{l:'M',c:'#FFD22E',n:'MEDIUM'},{l:'H',c:'#EDEDED',n:'HARD'}];
const cmpOf=(l,i)=>COMP[R.wet?0:CMP[l][i]];
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const mk=(tag,attrs,par)=>{const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(par)par.appendChild(e);return e};
// ---- stylised COTA outline (closed Catmull-Rom spline) ----
const PTS=[[850,480],[700,492],[540,490],[400,472],[305,430],[245,372],[262,312],[330,290],[382,332],[442,300],[470,242],[540,222],[602,252],[652,200],[705,150],[820,112],[925,100],[962,150],[930,202],[880,192],[852,242],[900,292],[932,352],[902,412],[852,432]];
function spline(p){const n=p.length;let d='M'+p[0];for(let i=0;i<n;i++){const a=p[(i-1+n)%n],b=p[i],c=p[(i+1)%n],e=p[(i+2)%n];
 d+=' C'+[b[0]+(c[0]-a[0])/6,b[1]+(c[1]-a[1])/6,c[0]-(e[0]-b[0])/6,c[1]-(e[1]-b[1])/6,c[0],c[1]].map(v=>v.toFixed(1)).join(',')}return d+'Z'}
const DPATH=spline(PTS);['glowp','edge','trkpath','centre'].forEach(id=>$(id).setAttribute('d',DPATH));
const path=$('trkpath'),TL=path.getTotalLength();
const pt=f=>path.getPointAtLength((((f%1)+1)%1)*TL);
{const a=pt(0),b=pt(0.004),dx=b.x-a.x,dy=b.y-a.y,l=Math.hypot(dx,dy)||1;$('sf').setAttribute('x1',a.x-dy/l*15);$('sf').setAttribute('y1',a.y+dx/l*15);$('sf').setAttribute('x2',a.x+dy/l*15);$('sf').setAttribute('y2',a.y-dx/l*15)}
// ---- precompute ----
const AGE=[],STOPS=[];for(let l=0;l<L;l++){AGE.push([]);STOPS.push([]);for(let i=0;i<N;i++){AGE[l].push(l&&!PIT[l-1][i]?AGE[l-1][i]+1:1);STOPS[l].push((l?STOPS[l-1][i]:0)+(l&&PIT[l-1][i]?1:0))}}
const maxL=i=>RET[i]?RET[i]-1:L;
let tEnd=0;for(let i=0;i<N;i++)if(!RET[i])tEnd=Math.max(tEnd,T[L][i]);
function prog(i,t){const m=maxL(i);if(t<=T[0][i])return -(T[0][i]-t)/Math.max(1e-6,T[1][i]-T[0][i]);if(t>=T[m][i])return m;
 let lo=0,hi=m;while(hi-lo>1){const q=(lo+hi)>>1;if(T[q][i]<=t)lo=q;else hi=q}return lo+(t-T[lo][i])/Math.max(1e-6,T[hi][i]-T[lo][i])}
function timeAt(j,p){const m=maxL(j);if(p<=0)return T[0][j]+p*(T[1][j]-T[0][j]);if(p>=m)return T[m][j];const l=Math.floor(p);return T[l][j]+(p-l)*(T[l+1][j]-T[l][j])}
const lapStart=l=>{let m=1e9;for(let i=0;i<N;i++)m=Math.min(m,T[l][i]);return m};
// ---- build DOM ----
$('lapTot').textContent=L;
const rowsEl=$('rows');rowsEl.style.height=(N*RH)+'px';
const rows=[],cars=[],trailC=[],TR=8;let sel=0;
for(let i=0;i<N;i++){const r=document.createElement('div');r.className='row';
 r.innerHTML='<span class="p"></span><span class="bar" style="background:'+D.colours[i]+'"></span><span class="n">'+D.drivers[i]+'</span><span class="t"></span><span class="g"></span>';
 r.onclick=()=>select(i);rowsEl.appendChild(r);rows.push({e:r,p:r.children[0],n:r.children[2],t:r.children[3],g:r.children[4]});}
const trg=$('trails');for(let i=0;i<N;i++){trailC.push([]);for(let k=0;k<TR;k++)trailC[i].push(mk('circle',{r:3.2,fill:D.colours[i],opacity:0,cx:-9,cy:-9},trg))}
const cg=$('cars');for(let i=0;i<N;i++){const g=mk('g',{class:'car'},cg);const m=mk('circle',{class:'m',r:7,fill:D.colours[i],stroke:'#0A0B0D','stroke-width':1.5},g);
 if(D.alt[i])mk('circle',{r:2.4,fill:'#fff'},g);const tx=mk('text',{y:-12,'text-anchor':'middle','font-size':11,'font-weight':700,fill:'#fff'},g);tx.textContent=D.drivers[i];
 const tt=mk('title',{},g);tt.textContent=D.drivers[i]+' - '+D.teams[i];g.onclick=()=>select(i);cars.push({g,m,tx})}
// speeds & controls
let mul=1,playing=true,t=0,last=null,showTrail=true;const SP=[0.5,1,2,4,8];
SP.forEach(s=>{const b=document.createElement('button');b.textContent=s+'x';b.dataset.s=s;if(s==1)b.className='on';b.onclick=()=>{mul=s;[...$('speeds').children].forEach(x=>x.className=x.dataset.s==s?'on':'')};$('speeds').appendChild(b)});
$('play').onclick=()=>{if(t>=tEnd){restart();return}playing=!playing;$('play').textContent=playing?'PAUSE':'PLAY'};
function restart(){t=0;playing=true;last=null;prevRank=null;$('play').textContent='PAUSE';trails.length=0}
$('restart').onclick=restart;$('trailChk').onchange=e=>{showTrail=e.target.checked;trg.style.display=showTrail?'':'none'};
$('bar').onclick=e=>{const r=$('bar').getBoundingClientRect();t=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*tEnd;prevRank=null;trails.length=0;last=null};
for(let l=0;l<L;l++)if(ST[l]>0){const s=document.createElement('div');s.className='seg'+(ST[l]==1?' v':'');s.style.left=(lapStart(l)/tEnd*100)+'%';s.style.width=((lapStart(l+1)-lapStart(l))/tEnd*100)+'%';$('bar').insertBefore(s,$('fill'))}
function select(i){sel=i;cars[i].g.parentNode.appendChild(cars[i].g);rows.forEach((r,k)=>r.e.classList.toggle('sel',k==i))}select(R.final[0]);
// ---- state ----
let prevRank=null,overtakes=0,trails=[],lastTxt=0,lastTrail=0;
const fmt=g=>g<10?g.toFixed(3):g.toFixed(1);
function compute(t){const P=[],out=[],fin=[];for(let i=0;i<N;i++){P[i]=prog(i,t);out[i]=!!RET[i]&&t>=T[maxL(i)][i];fin[i]=!RET[i]&&t>=T[L][i]}
 let order=[...Array(N).keys()].sort((a,b)=>out[a]!=out[b]?(out[a]?1:-1):fin[a]&&fin[b]?T[L][a]-T[L][b]:fin[a]!=fin[b]?(fin[a]?-1:1):P[b]-P[a]);
 if(t>=tEnd)order=R.final.slice();const rank=[];order.forEach((i,r)=>rank[i]=r);return{P,out,fin,order,rank}}
function gaps(S,t){const g=[],iv=[],ld=S.order[0];S.order.forEach((i,r)=>{if(S.out[i]){g[i]='OUT';iv[i]='OUT';return}
 if(r==0){g[i]='LEADER';iv[i]='-';return}const lead=S.P[ld]-S.P[i];
 if(lead>=1&&!S.fin[i]){g[i]='+'+Math.floor(lead)+'L';}else g[i]=S.fin[i]?'+'+fmt(T[L][i]-T[L][ld]):'+'+fmt(Math.max(0,t-timeAt(ld,S.P[i])));
 const ah=S.order[r-1];iv[i]=S.fin[i]&&S.fin[ah]?'+'+fmt(T[L][i]-T[L][ah]):'+'+fmt(Math.max(0,t-timeAt(ah,Math.min(S.P[i],maxL(ah)))))});return{g,iv}}
function feed(h){const f=$('feed');const d=document.createElement('div');d.innerHTML=h;f.appendChild(d);while(f.children.length>4)f.removeChild(f.firstChild);setTimeout(()=>d.remove(),6000)}
function frame(now){if(last===null)last=now;let dt=Math.min(0.1,(now-last)/1000);last=now;
 if(playing){t=Math.min(tEnd,t+dt*(tEnd/90)*mul);if(t>=tEnd){playing=false;$('play').textContent='REPLAY'}}
 const S=compute(t),ld=S.order[0],lapIdx=Math.min(L-1,Math.max(0,Math.floor(S.P[ld])));
 // markers
 cars.forEach((c,i)=>{let f=S.P[i];if(S.fin[i])f=1-0.012*(S.rank[i]+1);const p=pt(f);c.g.setAttribute('transform','translate('+p.x.toFixed(1)+','+p.y.toFixed(1)+')');
  c.g.style.opacity=S.out[i]?.3:S.fin[i]?.55:1;c.m.setAttribute('r',i==sel?10:7);c.tx.style.display=(i==sel||S.rank[i]<3)?'':'none'});
 // overtakes
 const perf=now;
 if(prevRank&&playing){S.order.forEach(i=>{if(S.rank[i]<prevRank[i]&&!S.out[i]&&!S.fin[i]){const passed=[];for(let j=0;j<N;j++)if(j!=i&&prevRank[j]<prevRank[i]&&S.rank[j]>S.rank[i]&&!S.out[j])passed.push(j);
   const pi=PIT[lapIdx]&&(PIT[lapIdx][i]||passed.some(j=>PIT[lapIdx][j]));
   if(passed.length&&!pi){overtakes++;rows[i].e.classList.remove('up');void rows[i].e.offsetWidth;rows[i].e.classList.add('up');
    const p=pt(S.P[i]);const rg=mk('circle',{class:'ring',cx:p.x,cy:p.y,r:7,stroke:D.colours[i]},$('rings'));setTimeout(()=>rg.remove(),900);
    feed('<b>'+D.drivers[i]+'</b> &#9650; passes '+D.drivers[passed[0]]+' for P'+(S.rank[i]+1))}}
  })}
 prevRank=S.rank.slice();
 // tower positions (cheap) + text (throttled)
 S.order.forEach((i,r)=>{rows[i].e.style.transform='translateY('+(r*RH)+'px)';rows[i].e.classList.toggle('out',S.out[i])});
 if(now-lastTxt>120){lastTxt=now;const G=gaps(S,t);
  for(let i=0;i<N;i++){const l=Math.min(L-1,Math.max(0,Math.floor(Math.max(0,S.P[i]))));const cp=cmpOf(l,i);const pit=PIT[l][i]&&!S.out[i]&&!S.fin[i]&&(S.P[i]-l)>0.8;
   rows[i].p.textContent=S.rank[i]+1;rows[i].g.textContent=pit?'PIT':G.g[i];const te=rows[i].t;te.textContent=cp.l;te.style.borderColor=cp.c;}
  const lapN=Math.min(L,Math.floor(Math.max(0,S.P[ld]))+1);$('lapNow').textContent=t>=tEnd?L:lapN;
  const st=ST[lapIdx];const fl=$('flag');fl.className='b'+(st?' sc':'');fl.innerHTML='<i></i>'+(t>=tEnd?'CHEQUERED FLAG':st==2?'SAFETY CAR':st==1?'VIRTUAL SC':'GREEN FLAG');
  $('wx').textContent=R.wet?'WET RACE':'DRY RACE';
  const m=Math.floor(t/60),s=Math.floor(t%60);$('clock').textContent=m+':'+(s<10?'0':'')+s;
  const l2=Math.min(L-1,Math.max(0,Math.floor(Math.max(0,S.P[sel])))),cp=cmpOf(l2,sel),pr=D.probs[D.drivers[sel]],G2=G;
  $('detail').innerHTML='<h3><span style="background:'+D.colours[sel]+'"></span>'+D.drivers[sel]+'<small style="font-size:13px;color:var(--mut);margin-left:auto">P'+(S.rank[sel]+1)+'</small></h3><div class="sub">'+D.teams[sel]+' - started P'+R.grid[sel]+'</div><div class="kv">'+
   '<div><small>GAP TO LEADER</small><b>'+G2.g[sel]+'</b></div><div><small>INTERVAL AHEAD</small><b>'+G2.iv[sel]+'</b></div>'+
   '<div><small>TYRE</small><b style="color:'+cp.c+'">'+cp.n+' - '+AGE[l2][sel]+' laps</b></div><div><small>PIT STOPS</small><b>'+STOPS[l2][sel]+'</b></div>'+
   '<div><small>MODEL WIN / PODIUM</small><b>'+(pr[0]*100).toFixed(1)+'% / '+(pr[1]*100).toFixed(0)+'%</b></div><div><small>MODEL EXP. FINISH</small><b>P'+pr[2].toFixed(1)+'</b></div></div>'+
   '<div class="sub" style="margin-top:10px">Status: '+(S.out[sel]?'<b style="color:var(--red)">RETIRED (lap '+RET[sel]+')</b>':S.fin[sel]?'Finished':'Running')+' - overtakes this replay: '+overtakes+'</div>'}
 // trails
 if(playing&&showTrail&&now-lastTrail>90){lastTrail=now;trails.unshift(cars.map((c,i)=>{const p=pt(S.fin[i]?0.99:S.P[i]);return[p.x,p.y]}));if(trails.length>TR)trails.pop()}
 if(showTrail)for(let i=0;i<N;i++)for(let k=0;k<TR;k++){const c=trailC[i][k],q=trails[k];if(q&&!S.out[i]&&!S.fin[i]){c.setAttribute('cx',q[i][0]);c.setAttribute('cy',q[i][1]);c.setAttribute('opacity',(0.32*(1-k/TR)).toFixed(2))}else c.setAttribute('opacity',0)}
 $('fill').style.width=(t/tEnd*100)+'%';
 requestAnimationFrame(frame)}
window.__apex={seek:x=>{t=x;prevRank=null},pause:()=>{playing=false},compute,rows,N,tEnd,R};
requestAnimationFrame(frame);
}catch(e){console.error(e);document.getElementById('app').style.display='none';document.getElementById('fallback').style.display='block'}})();
</script></body></html>"""
