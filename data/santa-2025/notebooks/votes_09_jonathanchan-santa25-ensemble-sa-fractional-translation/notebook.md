# [Santa25] Ensemble + SA + Fractional Translation

- **Author:** Jonathan Chan
- **Votes:** 192
- **Ref:** jonathanchan/santa25-ensemble-sa-fractional-translation
- **URL:** https://www.kaggle.com/code/jonathanchan/santa25-ensemble-sa-fractional-translation
- **Last run:** 2025-12-08 03:01:33.057000

---

# 1. Ensemble 

Based on https://www.kaggle.com/code/seshurajup/74-65-santa-2025-ensemble-with-jit

NOTE: Check for update of each dataset and notebook before ensembling

Repository:
1. https://www.kaggle.com/datasets/jazivxt/bucket-of-chump
2. https://raw.githubusercontent.com/SmartManoj/Santa-Scoreboard/main/submission.csv
3. https://www.kaggle.com/datasets/seowoohyeon/santa-2025-try3 and adding back https://www.kaggle.com/code/seowoohyeon/santa2025-ver2 (removed inactive)
4. https://www.kaggle.com/datasets/jonathanchan/santa25-public
5. https://www.kaggle.com/datasets/ethanrivera1/123456 (removed inactive)
6. https://www.kaggle.com/datasets/asalhi/telegram-public-shared-solution-for-santa-2025 (kindly extracted and shared by @asalhi at this author's request)

Notebooks:
1. https://www.kaggle.com/code/chistyakov/santa-2025-simple-optimization-new-slow-version
2. https://www.kaggle.com/code/egortrushin/santa25-simulated-annealing-with-translations (removed inactive)
3. https://www.kaggle.com/code/jonathanchan/santa25-ensemble-sa-fractional-translation
4. https://www.kaggle.com/code/seshurajup/74-15-santa-2025-ensemble-with-jit (renamed to various versions)
5. https://www.kaggle.com/code/smartmanoj/santa-claude
6. https://www.kaggle.com/code/chistyakov/santa-2025-quick-and-simple-optimization (removed inactive)
7. /kaggle/input/fork-of-santa-2025-simple-optimization (obsolete and replaced by a fork by the author) https://www.kaggle.com/code/guntasdhanjal/santa-2025-simple-optimization (removed inactive) /kaggle/input/santa-2025-simple-optimization-fork (yet another fork without changes - removed)
8. https://www.kaggle.com/code/qacenn/blending-blending-ble/notebook (removed inactive)
9. https://www.kaggle.com/code/seowoohyeon/santa2025-ver2 (removed inactive)
10. https://www.kaggle.com/code/eyestrain/blending-multiple-oplimisation
11. https://www.kaggle.com/code/jazivxt/why-not
12. https://www.kaggle.com/code/saspav/santa-submission
13. https://www.kaggle.com/code/jekiwantaufik/santa-2025-christmas-tree-packing-challenge-3 and subsequent updates of the model .csv file
14. https://www.kaggle.com/code/roshaw/santa2025-just-keep-on-trying and added earlier runs to santa25-public dataset
15. https://www.kaggle.com/code/datafad/decent-starting-solution
16. https://www.kaggle.com/code/egortrushin/santa25-improved-sa-with-translations

```python
# Clear working directory and retrieve from one database
!rm /kaggle/working/*
!rm /kaggle/working/solutions/*.*
import gc
gc.collect()
!wget -q https://raw.githubusercontent.com/SmartManoj/Santa-Scoreboard/main/submission.csv
```

```python
subs = [
    "/kaggle/working/",
    "/kaggle/input/jwt/other/csv/19",
    "/kaggle/input/bucket-of-chump",
    "/kaggle/input/why-not",
    "/kaggle/input/santa25-improved-sa-with-translations",
    "/kaggle/input/santa-2025-try3",
    "/kaggle/input/santa25-public",
    "/kaggle/input/santa2025-ver2",
    "/kaggle/input/santa-submission",
    "/kaggle/input/santa25-simulated-annealing-with-translations",
    "/kaggle/input/santa-2025-simple-optimization-new-slow-version",
    "/kaggle/input/santa-2025-fix-direction",
    "/kaggle/input/72-71-santa-2025-jit-parallel-sa-c",
    "/kaggle/input/santa-claude",
    "/kaggle/input/blending-multiple-oplimisation",
    "/kaggle/input/telegram-public-shared-solution-for-santa-2025",
    "/kaggle/input/santa2025-just-keep-on-trying",
    "/kaggle/input/decent-starting-solution",
    "/kaggle/input/santa25-ensemble-sa-fractional-translation",
]
OUTPUT_FILE = "/kaggle/working/submission.csv"
```

```python
import os
import glob
import math
import pandas as pd
import numpy as np
from numba import njit
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

@njit
def make_polygon_template():
    tw=0.15; th=0.2; bw=0.7; mw=0.4; ow=0.25
    tip=0.8; t1=0.5; t2=0.25; base=0.0; tbot=-th
    x=np.array([0,ow/2,ow/4,mw/2,mw/4,bw/2,tw/2,tw/2,-tw/2,-tw/2,-bw/2,-mw/4,-mw/2,-ow/4,-ow/2],np.float64)
    y=np.array([tip,t1,t1,t2,t2,base,base,tbot,tbot,base,base,t2,t2,t1,t1],np.float64)
    return x,y

@njit
def score_group(xs,ys,degs,tx,ty):
    n=xs.size; V=tx.size
    mnx=1e300; mny=1e300; mxx=-1e300; mxy=-1e300
    for i in range(n):
        r=degs[i]*math.pi/180.0
        c=math.cos(r); s=math.sin(r)
        xi=xs[i]; yi=ys[i]
        for j in range(V):
            X=c*tx[j]-s*ty[j]+xi
            Y=s*tx[j]+c*ty[j]+yi
            if X<mnx: mnx=X
            if X>mxx: mxx=X
            if Y<mny: mny=Y
            if Y>mxy: mxy=Y
    side=max(mxx-mnx,mxy-mny)
    return side*side/ n

def strip(a):
    return np.array([float(str(v).replace("s","")) for v in a],np.float64)

def all_csv_files():
    out=[]
    for f in subs:
        out += glob.glob(f + "/**/*.csv", recursive=True)
        out += glob.glob(f + "/*.csv")
    return sorted(set(out))

files = all_csv_files()

tx,ty = make_polygon_template()
best = {n:{"score":1e300,"data":None,"src":None} for n in range(1,201)}

for fp in tqdm(files, desc="scanning"):
    try:
        df = pd.read_csv(fp)
    except Exception:
        continue
    if not {"id","x","y","deg"}.issubset(df.columns):
        continue
    df = df.copy()
    df["N"] = df["id"].astype(str).str.split("_").str[0].astype(int)
    for n,g in df.groupby("N"):
        if n<1 or n>200:
            continue
        xs = strip(g["x"].to_numpy()); ys = strip(g["y"].to_numpy()); ds = strip(g["deg"].to_numpy())
        sc = score_group(xs,ys,ds,tx,ty)
        if sc < best[n]["score"]:
            best[n]["score"]=float(sc)
            best[n]["data"]=g.drop(columns=["N"]).copy()
            best[n]["src"]=str(fp).split("/")[3]

# ---- Override N=1 with fixed values, compute score normally, and set Source file label ----
manual_id = "001_0"
manual_data = pd.DataFrame({
    "id": [manual_id],
    "x": ["s0.0"],
    "y": ["s0.0"],
    "deg": ["s45.0"]
})
xs = strip(manual_data["x"].to_numpy())
ys = strip(manual_data["y"].to_numpy())
ds = strip(manual_data["deg"].to_numpy())
sc = score_group(xs, ys, ds, tx, ty)
best[1]["score"] = float(sc)
best[1]["data"] = manual_data.copy()
best[1]["src"] = "optimal value"

rows=[]
used={}
total=0.0
tbl = Table(title="Best per N", show_lines=False)
tbl.add_column("N", style="cyan", justify="right")
tbl.add_column("Score", style="magenta", justify="right")
tbl.add_column("Source file", style="green")

for n in range(1,201):
    entry = best[n]
    if entry["data"] is None:
        continue
    rows.append(entry["data"])
    used[entry["src"]] = used.get(entry["src"],0)+1
    total += entry["score"]
    tbl.add_row(f"{n}", f"{entry['score']:.12f}", entry["src"])

if not rows:
    console.print("[red]No solutions collected[/red]")
    raise SystemExit

out = pd.concat(rows, ignore_index=True)
out["sn"] = out["id"].str.split("_").str[0].astype(int)
out["si"] = out["id"].str.split("_").str[1].astype(int)
out = out.sort_values(["sn","si"]).drop(columns=["sn","si"])
out = out[['id','x','y','deg']]
out.to_csv(OUTPUT_FILE, index=False)

console.print(Panel(tbl, title="[bold]Best Solutions (per N)[/bold]"))
console.print(f"[bold white]Total score:[/bold white] [bold magenta]{total:.12f}[/bold magenta]")
console.print("\n[bold white]Puzzles taken per solution:[/bold white]")
file_tbl = Table(show_header=True)
file_tbl.add_column("File", style="green")
file_tbl.add_column("Count", style="yellow", justify="right")
for k,v in sorted(used.items(), key=lambda x:-x[1]):
    file_tbl.add_row(k, str(v))
console.print(file_tbl)
console.print(f"[bold green]Saved:[/bold green] {OUTPUT_FILE}")
```

# 2. Overlap checking (removed the Fractional translation of solutions part)
Based on https://www.kaggle.com/code/egortrushin/santa25-fractional-translation-of-solutions

In [the previous notebook](https://www.kaggle.com/code/egortrushin/santa25-translation-of-small-n-solutions), we attempted to construct large-$N$ solutions from small-$N$ solutions by applying 2D translations. Translations were chosen to be equal to box length. This time, we are going to identify translations that are shorter than the box length, and then use the shorter ones.

```python
from decimal import Decimal, getcontext

from shapely import affinity
from shapely.geometry import Polygon

# Set precision for Decimal
getcontext().prec = 25
scale_factor = Decimal("1e15")


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x="0", center_y="0", angle="0"):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal("0.0") * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated, xoff=float(self.center_x * scale_factor), yoff=float(self.center_y * scale_factor)
        )

import pandas as pd
import numpy as np
from decimal import Decimal
from shapely.strtree import STRtree


def to_str(x: Decimal):
    return f"s{float(x)}"


def load_configuration_from_df(n, existing_df):
    """
    Load existing configuration from submission CSV.
    """
    group_data = existing_df[existing_df["id"].str.startswith(f"{n:03d}_")]
    trees = []
    for _, row in group_data.iterrows():
        x = row["x"][1:]  # Remove 's' prefix
        y = row["y"][1:]
        deg = row["deg"][1:]
        trees.append(ChristmasTree(x, y, deg))
    return trees


def get_score(trees, n=None):
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / 1e15 for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    score = max(max_x - min_x, max_y - min_y) ** 2
    if n is not None:
        score /= n
    return score

def has_overlap(trees):
    """Check if any trees overlap"""
    if len(trees) <= 1:
        return False

    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)

    for i, poly in enumerate(polygons):
        indices = tree_index.query(poly)
        for idx in indices:
            if idx == i:
                continue
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False

import pandas as pd

def eval_df_sub(df, verb):
    failed = []
    total_score = 0.0
    for n in range(1, 201):
        trees = load_configuration_from_df(n, df)
        score = get_score(trees, n)
        total_score += score
        if verb:
            print(f"{n:3}  {score:.6f}")
        if has_overlap(trees):
            failed.append(n)

    if len(failed) == 0:
        print("Overlap check was succesfull")
    else:
        print("Overlap check failed for", *failed)
        # Clear output and Stop execution in the notebook
        !rm /kaggle/working/*
        !rm /kaggle/working/solutions/*.*
        raise RuntimeError("Overlap check failed")

    print(f"Total score: {total_score:.12f}")


def find_short_translation(trees, length, delta = 0.0001):

    while True:

        trees_ = []
        for tree in trees:
            for x in list(product(np.arange(2), repeat=2)):
                trees_.append(
                    ChristmasTree(
                        center_x=tree.center_x + Decimal(x[0] * (length-delta)),
                        center_y=tree.center_y + Decimal(x[1] * (length-delta)),
                        angle=tree.angle,
                    )
                )

        if has_overlap(trees_):
            break
        else:
            length -= delta
    return length


def find_translations(n, df):
    trees = load_configuration_from_df(n, df)
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / 1e15 for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    length = max(max_x - min_x, max_y - min_y)
    new_length = find_short_translation(trees, length)
    return [n, length, new_length]
```

```python
# Read in current ensemble output and check for overlaps
df = pd.read_csv("submission.csv")
print("#### Initial ensemble score")
eval_df_sub(df, False)
```

# 3. Simulated Annealing
1. Quick SA based on https://www.kaggle.com/code/smartmanoj/santa-claude-code
2. More in-depth SA based on https://www.kaggle.com/code/seowoohyeon/santa2025-ver2?scriptVersionId=282488403

```python
%%writefile a.cpp
// Tree Packer v3 - C++ version matching optimizer_v3.py
// Features: Corner tree targeting, population-based search, basin hopping
// Compile: g++ -O3 -march=native -std=c++17 -o tree_packer_v3 tree_packer_v3.cpp
// Run: ./tree_packer_v3 -i input.csv -o output.csv -n 15000 -r 5

#include <bits/stdc++.h>
using namespace std;
using namespace chrono;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

mt19937_64 rng(42);
uniform_real_distribution<double> U(0, 1);
inline double rf() { return U(rng); }
inline int ri(int n) { return rng() % n; }

struct Pt { double x, y; };

struct Poly {
    Pt p[NV];
    double x0, y0, x1, y1;
    void bbox() {
        x0 = x1 = p[0].x; y0 = y1 = p[0].y;
        for (int i = 1; i < NV; i++) {
            x0 = min(x0, p[i].x); x1 = max(x1, p[i].x);
            y0 = min(y0, p[i].y); y1 = max(y1, p[i].y);
        }
    }
};

Poly getPoly(double cx, double cy, double deg) {
    Poly q;
    double r = deg * PI / 180, c = cos(r), s = sin(r);
    for (int i = 0; i < NV; i++) {
        q.p[i].x = TX[i] * c - TY[i] * s + cx;
        q.p[i].y = TX[i] * s + TY[i] * c + cy;
    }
    q.bbox();
    return q;
}

bool pip(double px, double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.p[i].y > py) != (q.p[j].y > py) &&
            px < (q.p[j].x - q.p[i].x) * (py - q.p[i].y) / (q.p[j].y - q.p[i].y) + q.p[i].x)
            in = !in;
        j = i;
    }
    return in;
}

bool segInt(Pt a, Pt b, Pt c, Pt d) {
    auto ccw = [](Pt p, Pt q, Pt r) { return (r.y - p.y) * (q.x - p.x) > (q.y - p.y) * (r.x - p.x); };
    return ccw(a, c, d) != ccw(b, c, d) && ccw(a, b, c) != ccw(a, b, d);
}

bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.p[i].x, a.p[i].y, b)) return true;
        if (pip(b.p[i].x, b.p[i].y, a)) return true;
    }
    for (int i = 0; i < NV; i++)
        for (int j = 0; j < NV; j++)
            if (segInt(a.p[i], a.p[(i + 1) % NV], b.p[j], b.p[(j + 1) % NV])) return true;
    return false;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];

    void upd(int i) { pl[i] = getPoly(x[i], y[i], a[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); }

    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++) if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }

    bool hasOvlPair(int i, int j) const {
        if (overlap(pl[i], pl[j])) return true;
        for (int k = 0; k < n; k++) {
            if (k != i && k != j) {
                if (overlap(pl[i], pl[k]) || overlap(pl[j], pl[k])) return true;
            }
        }
        return false;
    }

    bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }

    double side() const {
        if (!n) return 0;
        double x0 = pl[0].x0, x1 = pl[0].x1, y0 = pl[0].y0, y1 = pl[0].y1;
        for (int i = 1; i < n; i++) {
            x0 = min(x0, pl[i].x0); x1 = max(x1, pl[i].x1);
            y0 = min(y0, pl[i].y0); y1 = max(y1, pl[i].y1);
        }
        return max(x1 - x0, y1 - y0);
    }

    double score() const { double s = side(); return s * s / n; }

    pair<double, double> centroid() const {
        double sx = 0, sy = 0;
        for (int i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
        return {sx / n, sy / n};
    }

    tuple<double, double, double, double> getBBox() const {
        double gx0 = pl[0].x0, gx1 = pl[0].x1, gy0 = pl[0].y0, gy1 = pl[0].y1;
        for (int i = 1; i < n; i++) {
            gx0 = min(gx0, pl[i].x0); gx1 = max(gx1, pl[i].x1);
            gy0 = min(gy0, pl[i].y0); gy1 = max(gy1, pl[i].y1);
        }
        return {gx0, gy0, gx1, gy1};
    }

    // Find trees that define the bounding box corners
    vector<int> findCornerTrees() const {
        auto [gx0, gy0, gx1, gy1] = getBBox();
        double eps = 0.01;
        vector<int> corners;
        for (int i = 0; i < n; i++) {
            if (abs(pl[i].x0 - gx0) < eps || abs(pl[i].x1 - gx1) < eps ||
                abs(pl[i].y0 - gy0) < eps || abs(pl[i].y1 - gy1) < eps) {
                corners.push_back(i);
            }
        }
        return corners;
    }
};

// Enhanced SA with 8 move types (matching v3)
Cfg sa_v3(Cfg c, int iter, double T0, double Tm, double ms, double rs, uint64_t seed) {
    rng.seed(seed);
    Cfg best = c, cur = c;
    double bs = best.side(), cs = bs, T = T0;
    double alpha = pow(Tm / T0, 1.0 / iter);
    int noImp = 0;

    for (int it = 0; it < iter; it++) {
        int moveType = ri(8);  // 8 move types
        double sc = T / T0;

        if (moveType < 4) {
            // Single tree moves
            int i = ri(c.n);
            double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];
            auto [cx, cy] = cur.centroid();

            if (moveType == 0) {
                cur.x[i] += (rf() - 0.5) * 2 * ms * sc;
                cur.y[i] += (rf() - 0.5) * 2 * ms * sc;
            } else if (moveType == 1) {
                double dx = cx - cur.x[i], dy = cy - cur.y[i];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    double st = rf() * ms * sc;
                    cur.x[i] += dx / d * st;
                    cur.y[i] += dy / d * st;
                }
            } else if (moveType == 2) {
                cur.a[i] += (rf() - 0.5) * 2 * rs * sc;
                cur.a[i] = fmod(cur.a[i] + 360, 360.0);
            } else {
                cur.x[i] += (rf() - 0.5) * ms * sc;
                cur.y[i] += (rf() - 0.5) * ms * sc;
                cur.a[i] += (rf() - 0.5) * rs * sc;
                cur.a[i] = fmod(cur.a[i] + 360, 360.0);
            }

            cur.upd(i);
            if (cur.hasOvl(i)) {
                cur.x[i] = ox; cur.y[i] = oy; cur.a[i] = oa;
                cur.upd(i);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 4 && c.n > 1) {
            // Swap positions
            int i = ri(c.n), j = ri(c.n);
            while (j == i) j = ri(c.n);

            double oxi = cur.x[i], oyi = cur.y[i];
            double oxj = cur.x[j], oyj = cur.y[j];

            cur.x[i] = oxj; cur.y[i] = oyj;
            cur.x[j] = oxi; cur.y[j] = oyi;
            cur.upd(i); cur.upd(j);

            if (cur.hasOvlPair(i, j)) {
                cur.x[i] = oxi; cur.y[i] = oyi;
                cur.x[j] = oxj; cur.y[j] = oyj;
                cur.upd(i); cur.upd(j);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 5) {
            // Bbox center move
            int i = ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];

            auto [gx0, gy0, gx1, gy1] = cur.getBBox();
            double bcx = (gx0 + gx1) / 2, bcy = (gy0 + gy1) / 2;
            double dx = bcx - cur.x[i], dy = bcy - cur.y[i];
            double d = sqrt(dx * dx + dy * dy);
            if (d > 1e-6) {
                double st = rf() * ms * sc * 0.5;
                cur.x[i] += dx / d * st;
                cur.y[i] += dy / d * st;
            }

            cur.upd(i);
            if (cur.hasOvl(i)) {
                cur.x[i] = ox; cur.y[i] = oy;
                cur.upd(i);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 6) {
            // Corner tree focus
            auto corners = cur.findCornerTrees();
            if (!corners.empty()) {
                int idx = corners[ri(corners.size())];
                double ox = cur.x[idx], oy = cur.y[idx], oa = cur.a[idx];

                auto [gx0, gy0, gx1, gy1] = cur.getBBox();
                double bcx = (gx0 + gx1) / 2, bcy = (gy0 + gy1) / 2;
                double dx = bcx - cur.x[idx], dy = bcy - cur.y[idx];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    double st = rf() * ms * sc * 0.3;
                    cur.x[idx] += dx / d * st;
                    cur.y[idx] += dy / d * st;
                    cur.a[idx] += (rf() - 0.5) * rs * sc * 0.5;
                    cur.a[idx] = fmod(cur.a[idx] + 360, 360.0);
                }

                cur.upd(idx);
                if (cur.hasOvl(idx)) {
                    cur.x[idx] = ox; cur.y[idx] = oy; cur.a[idx] = oa;
                    cur.upd(idx);
                    noImp++;
                    T *= alpha; if (T < Tm) T = Tm;
                    continue;
                }
            } else {
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else {
            // Coordinated move - shift two adjacent trees together
            int i = ri(c.n);
            int j = (i + 1) % c.n;

            double oxi = cur.x[i], oyi = cur.y[i];
            double oxj = cur.x[j], oyj = cur.y[j];

            double dx = (rf() - 0.5) * ms * sc * 0.5;
            double dy = (rf() - 0.5) * ms * sc * 0.5;

            cur.x[i] += dx; cur.y[i] += dy;
            cur.x[j] += dx; cur.y[j] += dy;
            cur.upd(i); cur.upd(j);

            if (cur.hasOvlPair(i, j)) {
                cur.x[i] = oxi; cur.y[i] = oyi;
                cur.x[j] = oxj; cur.y[j] = oyj;
                cur.upd(i); cur.upd(j);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        }

        double ns = cur.side();
        double delta = ns - cs;
        if (delta < 0 || rf() < exp(-delta / T)) {
            cs = ns;
            if (ns < bs) {
                bs = ns;
                best = cur;
                noImp = 0;
            } else {
                noImp++;
            }
        } else {
            cur = best;
            cs = bs;
            noImp++;
        }

        // Reheat if stagnant (600 threshold like v3)
        if (noImp > 600) {
            T = min(T * 3.0, T0 * 0.7);
            noImp = 0;
        }

        T *= alpha;
        if (T < Tm) T = Tm;
    }
    return best;
}

// Enhanced local search with corner tree prioritization
Cfg ls_v3(Cfg c, int iter) {
    Cfg best = c;
    double bs = best.side();
    double ps[] = {0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002};
    double rs[] = {15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25};
    int dx[] = {1, -1, 0, 0, 1, 1, -1, -1};
    int dy[] = {0, 0, 1, -1, 1, -1, 1, -1};

    for (int it = 0; it < iter; it++) {
        bool imp = false;

        // First optimize corner trees
        auto corners = best.findCornerTrees();
        for (int ci : corners) {
            for (double st : ps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[ci], oy = best.y[ci];
                    best.x[ci] += dx[d] * st;
                    best.y[ci] += dy[d] * st;
                    best.upd(ci);
                    if (!best.hasOvl(ci)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.x[ci] = ox; best.y[ci] = oy;
                            best.upd(ci);
                        }
                    } else {
                        best.x[ci] = ox; best.y[ci] = oy;
                        best.upd(ci);
                    }
                }
            }
            for (double st : rs) {
                for (double da : {st, -st}) {
                    double oa = best.a[ci];
                    best.a[ci] = fmod(best.a[ci] + da + 360, 360.0);
                    best.upd(ci);
                    if (!best.hasOvl(ci)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.a[ci] = oa;
                            best.upd(ci);
                        }
                    } else {
                        best.a[ci] = oa;
                        best.upd(ci);
                    }
                }
            }
        }

        // Then all other trees
        set<int> cornerSet(corners.begin(), corners.end());
        for (int i = 0; i < c.n; i++) {
            if (cornerSet.count(i)) continue;

            for (double st : ps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    best.x[i] += dx[d] * st;
                    best.y[i] += dy[d] * st;
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.x[i] = ox; best.y[i] = oy;
                            best.upd(i);
                        }
                    } else {
                        best.x[i] = ox; best.y[i] = oy;
                        best.upd(i);
                    }
                }
            }
            for (double st : rs) {
                for (double da : {st, -st}) {
                    double oa = best.a[i];
                    best.a[i] = fmod(best.a[i] + da + 360, 360.0);
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.a[i] = oa;
                            best.upd(i);
                        }
                    } else {
                        best.a[i] = oa;
                        best.upd(i);
                    }
                }
            }
        }

        if (!imp) break;
    }
    return best;
}

// Basin hopping perturbation
Cfg perturb(Cfg c, double strength, uint64_t seed) {
    rng.seed(seed);
    int numPerturb = max(1, (int)(c.n * 0.15));

    for (int k = 0; k < numPerturb; k++) {
        int i = ri(c.n);
        c.x[i] += (rf() - 0.5) * strength;
        c.y[i] += (rf() - 0.5) * strength;
        c.a[i] = fmod(c.a[i] + (rf() - 0.5) * 60 + 360, 360.0);
    }
    c.updAll();

    // Fix overlaps
    for (int iter = 0; iter < 100; iter++) {
        bool fixed = true;
        for (int i = 0; i < c.n; i++) {
            if (c.hasOvl(i)) {
                fixed = false;
                double cx = 0, cy = 0;
                for (int j = 0; j < c.n; j++) { cx += c.x[j]; cy += c.y[j]; }
                cx /= c.n; cy /= c.n;
                double dx = cx - c.x[i], dy = cy - c.y[i];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    c.x[i] -= dx / d * 0.02;
                    c.y[i] -= dy / d * 0.02;
                }
                c.a[i] = fmod(c.a[i] + rf() * 20 - 10 + 360, 360.0);
                c.upd(i);
            }
        }
        if (fixed) break;
    }
    return c;
}

// Fractional translation - apply small fractional adjustments to positions
Cfg fractional_translation(Cfg c, int max_iter = 200) {
    Cfg best = c;
    double bs = best.side();
    
    // Fractional step sizes to try
    double frac_steps[] = {0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001};
    int num_steps = sizeof(frac_steps) / sizeof(frac_steps[0]);
    
    // 8 directions: N, S, E, W, NE, NW, SE, SW
    int dx[] = {0, 0, 1, -1, 1, -1, 1, -1};
    int dy[] = {1, -1, 0, 0, 1, 1, -1, -1};
    
    for (int iter = 0; iter < max_iter; iter++) {
        bool improved = false;
        
        // Try fractional translations for each tree
        for (int i = 0; i < c.n; i++) {
            for (int s = 0; s < num_steps; s++) {
                double step = frac_steps[s];
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    
                    // Apply fractional translation
                    best.x[i] += dx[d] * step;
                    best.y[i] += dy[d] * step;
                    best.upd(i);
                    
                    if (!best.hasOvl(i)) {
                        double ns = best.side();
                        if (ns < bs - 1e-12) {
                            bs = ns;
                            improved = true;
                        } else {
                            best.x[i] = ox;
                            best.y[i] = oy;
                            best.upd(i);
                        }
                    } else {
                        best.x[i] = ox;
                        best.y[i] = oy;
                        best.upd(i);
                    }
                }
            }
        }
        
        if (!improved) break;
    }
    
    return best;
}

// Population-based optimization
Cfg opt_v3(Cfg c, int nr, int si) {
    Cfg best = c;
    double bs = best.side();

    // Population: keep top 3 solutions
    vector<pair<double, Cfg>> pop;
    pop.push_back({bs, c});

    for (int r = 0; r < nr; r++) {
        Cfg start;
        if (r == 0) {
            start = c;
        } else if (r < (int)pop.size()) {
            start = pop[r % pop.size()].second;
        } else {
            // Basin hopping: perturb best
            start = perturb(pop[0].second, 0.1 + 0.05 * (r % 3), 42 + r * 1000 + c.n);
        }

        Cfg o = sa_v3(start, si, 1.0, 0.000005, 0.25, 70.0, 42 + r * 1000 + c.n);
        o = ls_v3(o, 300);
        o = fractional_translation(o, 150);  // Apply fractional translation
        double s = o.side();

        pop.push_back({s, o});
        sort(pop.begin(), pop.end(), [](const pair<double, Cfg>& a, const pair<double, Cfg>& b) {
            return a.first < b.first;
        });
        if (pop.size() > 3) pop.resize(3);

        if (s < bs) {
            bs = s;
            best = o;
        }
    }
    return best;
}

map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) { cerr << "Cannot open " << fn << endl; return cfg; }
    string ln; getline(f, ln);
    map<int, vector<tuple<int, double, double, double>>> data;
    while (getline(f, ln)) {
        auto p1 = ln.find(','), p2 = ln.find(',', p1 + 1), p3 = ln.find(',', p2 + 1);
        string id = ln.substr(0, p1);
        string xs = ln.substr(p1 + 1, p2 - p1 - 1);
        string ys = ln.substr(p2 + 1, p3 - p2 - 1);
        string ds = ln.substr(p3 + 1);
        if (xs[0] == 's') xs = xs.substr(1);
        if (ys[0] == 's') ys = ys.substr(1);
        if (ds[0] == 's') ds = ds.substr(1);
        int n = stoi(id.substr(0, 3)), idx = stoi(id.substr(4));
        data[n].push_back({idx, stod(xs), stod(ys), stod(ds)});
    }
    for (auto& [n, v] : data) {
        Cfg c; c.n = n;
        for (auto& [i, x, y, d] : v) if (i < n) { c.x[i] = x; c.y[i] = y; c.a[i] = d; }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(15);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++)
                f << setfill('0') << setw(3) << n << "_" << i
                  << ",s" << c.x[i] << ",s" << c.y[i] << ",s" << c.a[i] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    string in = "/kaggle/working/submission.csv", out = "submission.csv";
    int si = 15000, nr = 5;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-o" && i + 1 < argc) out = argv[++i];
        else if (a == "-n" && i + 1 < argc) si = stoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) nr = stoi(argv[++i]);
    }

    cout << "Loading " << in << "...\n";
    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data!\n"; return 1; }
    cout << "Loaded " << cfg.size() << " configs\n";

    double init = 0;
    for (auto& [n, c] : cfg) init += c.score();
    cout << fixed << setprecision(6) << "Initial: " << init << "\n\nOptimizing (v3 with fractional translation)...\n\n";

    auto t0 = high_resolution_clock::now();
    map<int, Cfg> res;

    for (int n = 200; n >= 1; n--) {
        if (!cfg.count(n)) continue;
        Cfg c = cfg[n];
        double os = c.score();

        int r = nr, it = si;
        if (n <= 20) { r = 6; it = (int)(si * 1.5); }
        else if (n <= 50) { r = 5; it = (int)(si * 1.3); }
        else if (n > 150) { r = 4; it = (int)(si * 0.8); }

        Cfg o = opt_v3(c, r, it);

        // Backward adapt
        for (auto& [m, pc] : res) {
            if (m > n && m <= n + 15) {
                Cfg ad; ad.n = n;
                for (int i = 0; i < n; i++) {
                    ad.x[i] = pc.x[i]; ad.y[i] = pc.y[i]; ad.a[i] = pc.a[i];
                }
                ad.updAll();
                if (!ad.anyOvl()) {
                    ad = sa_v3(ad, 5000, 0.5, 0.0001, 0.2, 50.0, n * 7);
                    ad = ls_v3(ad, 200);
                    ad = fractional_translation(ad, 100);  // Apply fractional translation
                    if (ad.side() < o.side()) o = ad;
                }
            }
        }

        // Final fractional translation pass
        o = fractional_translation(o, 100);
        
        res[n] = o;
        double ns = o.score();
        if (ns < os - 1e-9) {
            double imp = (os - ns) / os * 100;
            cout << "n=" << setw(3) << n << ": " << os << " -> " << ns
                 << " (" << setprecision(4) << imp << "% better)\n" << setprecision(12);
        }
    }

    auto t1 = high_resolution_clock::now();
    double el = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;

    double fin = 0;
    for (auto& [n, c] : res) fin += c.score();

    cout << "\n========================================\n";
    cout << "Initial: " << init << "\nFinal:   " << fin << "\n";
    cout << "Improve: " << (init - fin) << " (" << setprecision(2)
         << (init - fin) / init * 100 << "%)\n";
    cout << "Time:    " << setprecision(1) << el << "s\n";
    cout << "========================================\n";

    saveCSV(out, res);
    cout << "Saved " << out << endl;
    return 0;
}
```

```python
!g++ -O3 -march=native -std=c++17 -o a.exe a.cpp && ./a.exe
```

```python
# Optional: Rerun quick SA for minor improvements
#!g++ -O3 -march=native -std=c++17 -o a.exe a.cpp && ./a.exe
```

```python
%%writefile a.cpp
// Tree Packer v18 - PARALLEL + AGGRESSIVE BACK PROPAGATION
// + Free-area & Protrusion removal & reinsertion heuristic
// + Edge-based slide compaction (outline-aware)
// Compile example:
//   OMP_NUM_THREADS=32 g++ -fopenmp -O3 -march=native -std=c++17 -o a.exe a.cpp

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

alignas(64) const double TX[NV] = {
    0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,
    -0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125
};
alignas(64) const double TY[NV] = {
    0.8,0.5,0.5,0.25,0.25,0,0,-0.2,
    -0.2,0,0,0.25,0.25,0.5,0.5
};

struct FastRNG {
    uint64_t s[2];
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
        s1 ^= s0;
        s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        s[1] = rotl(s1, 37);
        return r;
    }
    inline double rf() { return (next() >> 11) * 0x1.0p-53; }
    inline double rf2() { return rf() * 2.0 - 1.0; }
    inline int ri(int n) { return (int)(next() % (uint64_t)n); }
    inline double gaussian() {
        double u1 = rf() + 1e-10, u2 = rf();
        return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
    }
};

struct Poly {
    double px[NV], py[NV];
    double x0, y0, x1, y1;
};

inline void getPoly(double cx, double cy, double deg, Poly& q) {
    double rad = deg * (PI / 180.0);
    double s = sin(rad), c = cos(rad);
    double minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
    for (int i = 0; i < NV; i++) {
        double x = TX[i] * c - TY[i] * s + cx;
        double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x;
        q.py[i] = y;
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

inline bool pip(double px, double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.py[i] > py) != (q.py[j] > py) &&
            px < (q.px[j] - q.px[i]) * (py - q.py[i]) / (q.py[j] - q.py[i]) + q.px[i])
            in = !in;
        j = i;
    }
    return in;
}

inline bool segInt(double ax, double ay, double bx, double by,
                   double cx, double cy, double dx, double dy) {
    double d1 = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx);
    double d2 = (dx-cx)*(by-cy) - (dy-cy)*(bx-cx);
    double d3 = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
    double d4 = (bx-ax)*(dy-ay) - (by-ay)*(dx-ax);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0));
}

inline bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.px[i], a.py[i], b)) return true;
        if (pip(b.px[i], b.py[i], a)) return true;
    }
    for (int i = 0; i < NV; i++) {
        int ni = (i + 1) % NV;
        for (int j = 0; j < NV; j++) {
            int nj = (j + 1) % NV;
            if (segInt(a.px[i], a.py[i], a.px[ni], a.py[ni],
                      b.px[j], b.py[j], b.px[nj], b.py[nj])) return true;
        }
    }
    return false;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    double gx0, gy0, gx1, gy1;

    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    inline void updAll() { for (int i = 0; i < n; i++) upd(i); updGlobal(); }

    inline void updGlobal() {
        gx0 = gy0 = 1e9;
        gx1 = gy1 = -1e9;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    inline bool hasOvl(int i) const {
        for (int j = 0; j < n; j++)
            if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }

    inline bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }

    inline double side() const {
        return max(gx1 - gx0, gy1 - gy0);
    }
    inline double score() const {
        double s = side();
        return s * s / n;
    }

    void getBoundary(vector<int>& b) const {
        b.clear();
        double eps = 0.01;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 - gx0 < eps || gx1 - pl[i].x1 < eps ||
                pl[i].y0 - gy0 < eps || gy1 - pl[i].y1 < eps)
                b.push_back(i);
        }
    }

    // Remove tree at index, shift others down
    Cfg removeTree(int removeIdx) const {
        Cfg c;
        c.n = n - 1;
        int j = 0;
        for (int i = 0; i < n; i++) {
            if (i != removeIdx) {
                c.x[j] = x[i];
                c.y[j] = y[i];
                c.a[j] = a[i];
                j++;
            }
        }
        c.updAll();
        return c;
    }
};

// ========== Core local transforms ==========

Cfg squeeze(Cfg c) {
    double cx = (c.gx0 + c.gx1) / 2.0;
    double cy = (c.gy0 + c.gy1) / 2.0;
    for (double scale = 0.9995; scale >= 0.98; scale -= 0.0005) {
        Cfg trial = c;
        for (int i = 0; i < c.n; i++) {
            trial.x[i] = cx + (c.x[i] - cx) * scale;
            trial.y[i] = cy + (c.y[i] - cy) * scale;
        }
        trial.updAll();
        if (!trial.anyOvl()) c = trial;
        else break;
    }
    return c;
}

Cfg compaction(Cfg c, int iters) {
    double bs = c.side();
    for (int it = 0; it < iters; it++) {
        double cx = (c.gx0 + c.gx1) / 2.0;
        double cy = (c.gy0 + c.gy1) / 2.0;
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            double ox = c.x[i], oy = c.y[i];
            double dx = cx - c.x[i];
            double dy = cy - c.y[i];
            double d = sqrt(dx*dx + dy*dy);
            if (d < 1e-6) continue;
            const double steps[] = {0.02, 0.008, 0.003, 0.001, 0.0004};
            for (double step : steps) {
                c.x[i] = ox + dx/d * step;
                c.y[i] = oy + dy/d * step;
                c.upd(i);
                if (!c.hasOvl(i)) {
                    c.updGlobal();
                    if (c.side() < bs - 1e-12) {
                        bs = c.side();
                        improved = true;
                        ox = c.x[i];
                        oy = c.y[i];
                    } else {
                        c.x[i] = ox;
                        c.y[i] = oy;
                        c.upd(i);
                    }
                } else {
                    c.x[i] = ox;
                    c.y[i] = oy;
                    c.upd(i);
                }
            }
        }
        c.updGlobal();
        if (!improved) break;
    }
    return c;
}

Cfg localSearch(Cfg c, int maxIter) {
    double bs = c.side();
    const double steps[] = {0.004, 0.002, 0.001, 0.0005, 0.00025, 0.0001};
    const double rots[]  = {2.0, 1.0, 0.5, 0.25, 0.125};
    const int dxs[] = {1,-1,0,0,1,1,-1,-1};
    const int dys[] = {0,0,1,-1,1,-1,1,-1};

    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            double cx = (c.gx0 + c.gx1) / 2.0;
            double cy = (c.gy0 + c.gy1) / 2.0;
            double ddx = cx - c.x[i];
            double ddy = cy - c.y[i];
            double dist = sqrt(ddx*ddx + ddy*ddy);
            if (dist > 1e-6) {
                for (double st : steps) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += ddx/dist * st;
                    c.y[i] += ddy/dist * st;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) {
                            bs = c.side();
                            improved = true;
                        } else {
                            c.x[i] = ox;
                            c.y[i] = oy;
                            c.upd(i);
                            c.updGlobal();
                        }
                    } else {
                        c.x[i] = ox;
                        c.y[i] = oy;
                        c.upd(i);
                    }
                }
            }
            for (double st : steps) {
                for (int d = 0; d < 8; d++) {
                    double ox = c.x[i], oy = c.y[i];
                    c.x[i] += dxs[d]*st;
                    c.y[i] += dys[d]*st;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) {
                            bs = c.side();
                            improved = true;
                        } else {
                            c.x[i] = ox;
                            c.y[i] = oy;
                            c.upd(i);
                            c.updGlobal();
                        }
                    } else {
                        c.x[i] = ox;
                        c.y[i] = oy;
                        c.upd(i);
                    }
                }
            }
            for (double rt : rots) {
                for (double da : {rt, -rt}) {
                    double oa = c.a[i];
                    c.a[i] += da;
                    while (c.a[i] < 0)   c.a[i] += 360;
                    while (c.a[i] >= 360) c.a[i] -= 360;
                    c.upd(i);
                    if (!c.hasOvl(i)) {
                        c.updGlobal();
                        if (c.side() < bs - 1e-12) {
                            bs = c.side();
                            improved = true;
                        } else {
                            c.a[i] = oa;
                            c.upd(i);
                            c.updGlobal();
                        }
                    } else {
                        c.a[i] = oa;
                        c.upd(i);
                    }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

// ========= Edge-based slide compaction (outline-aware) =========
//
// 각 트리를 여러 방향으로 "충돌 직전까지" 이분탐색으로 슬라이드
// → 외곽선 기준으로 벽/이웃에 딱 붙는 효과
Cfg edgeSlideCompaction(Cfg c, int outerIter) {
    double bestSide = c.side();

    for (int it = 0; it < outerIter; ++it) {
        bool improved = false;

        for (int i = 0; i < c.n; ++i) {
            double gcx = (c.gx0 + c.gx1) * 0.5;
            double gcy = (c.gy0 + c.gy1) * 0.5;

            double dirs[5][2] = {
                {gcx - c.x[i], gcy - c.y[i]}, // bbox 중심 방향
                { 1.0,  0.0},
                {-1.0,  0.0},
                { 0.0,  1.0},
                { 0.0, -1.0},
            };

            for (int d = 0; d < 5; ++d) {
                double dx = dirs[d][0];
                double dy = dirs[d][1];
                double len = sqrt(dx*dx + dy*dy);
                if (len < 1e-9) continue;
                dx /= len;
                dy /= len;

                double maxStep = 0.30;
                double lo = 0.0, hi = maxStep;
                double bestStep = 0.0;

                double ox = c.x[i];
                double oy = c.y[i];

                for (int it2 = 0; it2 < 20; ++it2) {
                    double mid = 0.5 * (lo + hi);

                    c.x[i] = ox + dx * mid;
                    c.y[i] = oy + dy * mid;
                    c.upd(i);
                    c.updGlobal();

                    bool okOverlap = !c.hasOvl(i);
                    bool okSide    = (c.side() <= bestSide + 1e-9);

                    if (okOverlap && okSide) {
                        bestStep = mid;
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }

                if (bestStep > 1e-6) {
                    c.x[i] = ox + dx * bestStep;
                    c.y[i] = oy + dy * bestStep;
                    c.upd(i);
                    c.updGlobal();

                    double ns = c.side();
                    if (ns < bestSide - 1e-12) {
                        bestSide = ns;
                        improved = true;
                    }
                } else {
                    c.x[i] = ox;
                    c.y[i] = oy;
                    c.upd(i);
                    c.updGlobal();
                }
            }
        }

        if (!improved) break;
    }

    return c;
}

// ========== SA + perturb + parallel optimize ==========

Cfg sa_opt(Cfg c, int iter, double T0, double Tm, uint64_t seed) {
    FastRNG rng(seed);
    Cfg best = c, cur = c;
    double bs = best.side(), cs = bs, T = T0;
    double alpha = pow(Tm / T0, 1.0 / iter);
    int noImp = 0;

    for (int it = 0; it < iter; it++) {
        int mt = rng.ri(10);
        double sc = T / T0;
        bool valid = true;

        if (mt == 0) {
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            cur.x[i] += rng.gaussian() * 0.5 * sc;
            cur.y[i] += rng.gaussian() * 0.5 * sc;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 1) {
            int i = rng.ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            double bcx = (cur.gx0+cur.gx1)/2.0;
            double bcy = (cur.gy0+cur.gy1)/2.0;
            double dx = bcx - cur.x[i];
            double dy = bcy - cur.y[i];
            double d  = sqrt(dx*dx + dy*dy);
            if (d > 1e-6) {
                cur.x[i] += dx/d * rng.rf() * 0.6 * sc;
                cur.y[i] += dy/d * rng.rf() * 0.6 * sc;
            }
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 2) {
            int i = rng.ri(c.n);
            double oa = cur.a[i];
            cur.a[i] += rng.gaussian() * 80 * sc;
            while (cur.a[i] < 0)   cur.a[i] += 360;
            while (cur.a[i] >= 360) cur.a[i] -= 360;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        else if (mt == 3) {
            int i = rng.ri(c.n);
            double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
            cur.x[i] += rng.rf2() * 0.5 * sc;
            cur.y[i] += rng.rf2() * 0.5 * sc;
            cur.a[i] += rng.rf2() * 60 * sc;
            while (cur.a[i] < 0)   cur.a[i] += 360;
            while (cur.a[i] >= 360) cur.a[i] -= 360;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        else if (mt == 4) {
            vector<int> boundary;
            cur.getBoundary(boundary);
            if (!boundary.empty()) {
                int i = boundary[rng.ri((int)boundary.size())];
                double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
                double bcx = (cur.gx0+cur.gx1)/2.0;
                double bcy = (cur.gy0+cur.gy1)/2.0;
                double dx = bcx - cur.x[i];
                double dy = bcy - cur.y[i];
                double d  = sqrt(dx*dx + dy*dy);
                if (d > 1e-6) {
                    cur.x[i] += dx/d * rng.rf() * 0.7 * sc;
                    cur.y[i] += dy/d * rng.rf() * 0.7 * sc;
                }
                cur.a[i] += rng.rf2() * 50 * sc;
                while (cur.a[i] < 0)   cur.a[i] += 360;
                while (cur.a[i] >= 360) cur.a[i] -= 360;
                cur.upd(i);
                if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
            } else valid = false;
        }
        else if (mt == 5) {
            double factor = 1.0 - rng.rf() * 0.004 * sc;
            double cx = (cur.gx0 + cur.gx1) / 2.0;
            double cy = (cur.gy0 + cur.gy1) / 2.0;
            Cfg trial = cur;
            for (int i = 0; i < c.n; i++) {
                trial.x[i] = cx + (cur.x[i] - cx) * factor;
                trial.y[i] = cy + (cur.y[i] - cy) * factor;
            }
            trial.updAll();
            if (!trial.anyOvl()) cur = trial;
            else valid = false;
        }
        else if (mt == 6) {
            int i = rng.ri(c.n);
            double ox=cur.x[i], oy=cur.y[i];
            double levy = pow(rng.rf() + 0.001, -1.3) * 0.008;
            cur.x[i] += rng.rf2() * levy;
            cur.y[i] += rng.rf2() * levy;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 7 && c.n > 1) {
            int i = rng.ri(c.n), j = (i + 1) % c.n;
            double oxi=cur.x[i], oyi=cur.y[i];
            double oxj=cur.x[j], oyj=cur.y[j];
            double dx = rng.rf2() * 0.3 * sc;
            double dy = rng.rf2() * 0.3 * sc;
            cur.x[i]+=dx; cur.y[i]+=dy;
            cur.x[j]+=dx; cur.y[j]+=dy;
            cur.upd(i); cur.upd(j);
            if (cur.hasOvl(i) || cur.hasOvl(j)) {
                cur.x[i]=oxi; cur.y[i]=oyi;
                cur.x[j]=oxj; cur.y[j]=oyj;
                cur.upd(i); cur.upd(j);
                valid=false;
            }
        }
        else {
            int i = rng.ri(c.n);
            double ox=cur.x[i], oy=cur.y[i];
            cur.x[i] += rng.rf2() * 0.002;
            cur.y[i] += rng.rf2() * 0.002;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }

        if (!valid) {
            noImp++;
            T *= alpha;
            if (T < Tm) T = Tm;
            continue;
        }

        cur.updGlobal();
        double ns = cur.side();
        double delta = ns - cs;

        if (delta < 0 || rng.rf() < exp(-delta / T)) {
            cs = ns;
            if (ns < bs) {
                bs = ns;
                best = cur;
                noImp = 0;
            } else noImp++;
        } else {
            cur = best;
            cs  = bs;
            noImp++;
        }

        if (noImp > 200) {
            T = min(T * 5.0, T0);
            noImp = 0;
        }
        T *= alpha;
        if (T < Tm) T = Tm;
    }
    return best;
}

Cfg perturb(Cfg c, double str, FastRNG& rng) {
    Cfg original = c;
    int np = max(1, (int)(c.n * 0.08 + str * 3));
    for (int k = 0; k < np; k++) {
        int i = rng.ri(c.n);
        c.x[i] += rng.gaussian() * str * 0.5;
        c.y[i] += rng.gaussian() * str * 0.5;
        c.a[i] += rng.gaussian() * 30.0;
        while (c.a[i] < 0)   c.a[i] += 360;
        while (c.a[i] >= 360) c.a[i] -= 360;
    }
    c.updAll();
    for (int iter = 0; iter < 150; iter++) {
        bool fixed = true;
        for (int i = 0; i < c.n; i++) {
            if (c.hasOvl(i)) {
                fixed = false;
                double cx = (c.gx0+c.gx1)/2.0;
                double cy = (c.gy0+c.gy1)/2.0;
                double dx = c.x[i] - cx;
                double dy = c.y[i] - cy;
                double d  = sqrt(dx*dx + dy*dy);
                if (d > 1e-6) {
                    c.x[i] += dx/d*0.02;
                    c.y[i] += dy/d*0.02;
                }
                c.a[i] += rng.rf2() * 15.0;
                while (c.a[i] < 0)   c.a[i] += 360;
                while (c.a[i] >= 360) c.a[i] -= 360;
                c.upd(i);
            }
        }
        if (fixed) break;
    }
    c.updGlobal();
    if (c.anyOvl()) return original;
    return c;
}

Cfg optimizeParallel(Cfg c, int iters, int restarts) {
    Cfg globalBest = c;
    double globalBestSide = c.side();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        FastRNG rng(42 + tid * 1000 + c.n);
        Cfg localBest = c;
        double localBestSide = c.side();

        #pragma omp for schedule(dynamic)
        for (int r = 0; r < restarts; r++) {
            Cfg start;
            if (r == 0) {
                start = c;
            } else {
                start = perturb(c, 0.02 + 0.02 * (r % 8), rng);
                if (start.anyOvl()) continue;
            }

            uint64_t seed = 42 + r * 1000 + tid * 100000 + c.n;
            Cfg o = sa_opt(start, iters, 2.5, 0.0000005, seed);
            o = squeeze(o);
            o = compaction(o, 50);
            o = edgeSlideCompaction(o, 10);
            o = localSearch(o, 80);

            if (!o.anyOvl() && o.side() < localBestSide) {
                localBestSide = o.side();
                localBest = o;
            }
        }

        #pragma omp critical
        {
            if (!localBest.anyOvl() && localBestSide < globalBestSide) {
                globalBestSide = localBestSide;
                globalBest = localBest;
            }
        }
    }

    globalBest = squeeze(globalBest);
    globalBest = compaction(globalBest, 80);
    globalBest = edgeSlideCompaction(globalBest, 12);
    globalBest = localSearch(globalBest, 150);

    if (globalBest.anyOvl()) return c;
    return globalBest;
}

// ========== Free-area & protrusion removal & reinsertion heuristic ==========

struct TreeState {
    double x, y, a;
};

void computeFreeArea(const Cfg& c, vector<double>& freeArea) {
    int n = c.n;
    freeArea.assign(n, 0.0);
    vector<double> area(n), overlapSum(n, 0.0);

    for (int i = 0; i < n; ++i) {
        double w = max(0.0, c.pl[i].x1 - c.pl[i].x0);
        double h = max(0.0, c.pl[i].y1 - c.pl[i].y0);
        area[i] = w * h;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            double ix0 = max(c.pl[i].x0, c.pl[j].x0);
            double iy0 = max(c.pl[i].y0, c.pl[j].y0);
            double ix1 = min(c.pl[i].x1, c.pl[j].x1);
            double iy1 = min(c.pl[i].y1, c.pl[j].y1);
            double dx = ix1 - ix0;
            double dy = iy1 - iy0;
            if (dx > 0.0 && dy > 0.0) {
                overlapSum[i] += dx * dy;
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        double occ = overlapSum[i];
        if (occ > area[i]) occ = area[i];
        freeArea[i] = max(0.0, area[i] - occ);  // 0 = 매우 답답, area[i] = 완전 여유
    }
}

// "튀어나와있는" 정도: 전체 bbox의 경계에 거의 붙어 있으면서
// 중심에서 멀리 떨어진 트리에 높은 점수 부여
void computeProtrudeScore(const Cfg& c, vector<double>& protrudeScore) {
    int n = c.n;
    protrudeScore.assign(n, 0.0);
    double cx = (c.gx0 + c.gx1) * 0.5;
    double cy = (c.gy0 + c.gy1) * 0.5;
    double side = c.side();
    double eps = side * 0.02;  // 2% 이내면 경계에 있다고 봄

    for (int i = 0; i < n; ++i) {
        bool onBoundary =
            (c.pl[i].x0 - c.gx0 < eps) ||
            (c.gx1 - c.pl[i].x1 < eps) ||
            (c.pl[i].y0 - c.gy0 < eps) ||
            (c.gy1 - c.pl[i].y1 < eps);

        if (!onBoundary) {
            protrudeScore[i] = 0.0;
            continue;
        }

        double tx = 0.5 * (c.pl[i].x0 + c.pl[i].x1);
        double ty = 0.5 * (c.pl[i].y0 + c.pl[i].y1);
        double d  = sqrt((tx - cx)*(tx - cx) + (ty - cy)*(ty - cy));

        // 거리 자체를 score로 사용 (멀수록 더 튀어나왔다고 가정)
        protrudeScore[i] = d;
    }
}

Cfg reinsertTrees(const Cfg& base, const vector<TreeState>& removed, uint64_t seed) {
    Cfg cur = base;
    FastRNG rng(seed);

    for (const auto& t : removed) {
        if (cur.n >= MAX_N) return base; // 안전장치

        int idx = cur.n;
        cur.n++;
        cur.x[idx] = t.x;
        cur.y[idx] = t.y;
        cur.a[idx] = t.a;
        cur.upd(idx);
        cur.updGlobal();

        bool placed = false;
        for (int attempt = 0; attempt < 200; ++attempt) {
            if (!cur.hasOvl(idx)) { placed = true; break; }

            double cx = (cur.gx0 + cur.gx1) * 0.5;
            double cy = (cur.gy0 + cur.gy1) * 0.5;
            double radius = 0.1 + 0.6 * rng.rf();
            double ang    = 2.0 * PI * rng.rf();

            cur.x[idx] = cx + radius * cos(ang);
            cur.y[idx] = cy + radius * sin(ang);
            cur.a[idx] = fmod(t.a + rng.rf2() * 120.0 + 360.0, 360.0);

            cur.upd(idx);
            cur.updGlobal();
        }

        if (!placed) {
            cur.n--;
            return base;
        }
    }

    if (cur.anyOvl()) return base;
    return cur;
}

Cfg freeAreaHeuristic(const Cfg& c, double removeRatio, uint64_t seed) {
    Cfg best = c;
    int n = c.n;
    if (n <= 5) return best;   // 너무 작은 n은 스킵

    int k = (int)floor(n * removeRatio + 1e-9);
    if (k < 1) k = 1;
    if (k >= n) k = n - 1;

    vector<double> freeArea;
    vector<double> protrudeScore;
    computeFreeArea(c, freeArea);
    computeProtrudeScore(c, protrudeScore);

    // freeArea 큰 순 (여유 많은 애들)
    vector<pair<double,int>> freeList;
    freeList.reserve(n);
    for (int i = 0; i < n; ++i)
        freeList.emplace_back(freeArea[i], i);
    sort(freeList.begin(), freeList.end(),
         [](const pair<double,int>& a, const pair<double,int>& b){
             if (a.first != b.first) return a.first > b.first;
             return a.second < b.second;
         });

    // protrudeScore 큰 순 (튀어나온 애들, 경계+원점에서 먼)
    vector<pair<double,int>> protList;
    protList.reserve(n);
    for (int i = 0; i < n; ++i)
        if (protrudeScore[i] > 0.0)
            protList.emplace_back(protrudeScore[i], i);
    sort(protList.begin(), protList.end(),
         [](const pair<double,int>& a, const pair<double,int>& b){
             if (a.first != b.first) return a.first > b.first;
             return a.second < b.second;
         });

    // 제거할 개수: 대략 2/3는 튀어나온 애들, 1/3는 여유 많은 애들
    int kProt = min((int)protList.size(), (int)(k * 2 / 3));
    int kFree = k - kProt;
    if (kFree < 0) kFree = 0;

    vector<bool> removeFlag(n, false);
    vector<TreeState> removed;
    removed.reserve(k);

    // 1) 튀어나온 애들부터 제거
    int removedCnt = 0;
    for (int i = 0; i < (int)protList.size() && removedCnt < kProt; ++i) {
        int idx = protList[i].second;
        if (removeFlag[idx]) continue;
        removeFlag[idx] = true;
        removed.push_back(TreeState{c.x[idx], c.y[idx], c.a[idx]});
        removedCnt++;
    }

    // 2) 남은 수만큼 freeArea 큰 애들 제거
    for (int i = 0; i < (int)freeList.size() && removedCnt < k; ++i) {
        int idx = freeList[i].second;
        if (removeFlag[idx]) continue;
        removeFlag[idx] = true;
        removed.push_back(TreeState{c.x[idx], c.y[idx], c.a[idx]});
        removedCnt++;
    }

    if (removed.empty()) return best;

    Cfg reduced;
    reduced.n = n - (int)removed.size();
    int ptr = 0;
    for (int i = 0; i < n; ++i) {
        if (!removeFlag[i]) {
            reduced.x[ptr] = c.x[i];
            reduced.y[ptr] = c.y[i];
            reduced.a[ptr] = c.a[i];
            ptr++;
        }
    }
    reduced.updAll();
    if (reduced.anyOvl()) return best;

    // 남은 subset 다시 최적화 (살짝 강하게)
    Cfg reducedOpt = optimizeParallel(reduced, max(2000, 8000), 8);

    // 제거한 트리 재삽입
    Cfg withInserted = reinsertTrees(reducedOpt, removed, seed);
    if (withInserted.n != n || withInserted.anyOvl()) return best;

    // 한 번 더 조이기 (+ edge slide)
    withInserted = squeeze(withInserted);
    withInserted = compaction(withInserted, 40);
    withInserted = edgeSlideCompaction(withInserted, 10);
    withInserted = localSearch(withInserted, 80);

    if (!withInserted.anyOvl() && withInserted.side() < best.side() - 1e-12) {
        return withInserted;
    }
    return best;
}

// ========== IO & main ==========

map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) return cfg;
    string ln; getline(f, ln);
    map<int, vector<tuple<int,double,double,double>>> data;
    while (getline(f, ln)) {
        size_t p1=ln.find(','), p2=ln.find(',',p1+1), p3=ln.find(',',p2+1);
        string id = ln.substr(0,p1);
        string xs = ln.substr(p1+1,p2-p1-1);
        string ys = ln.substr(p2+1,p3-p2-1);
        string ds = ln.substr(p3+1);
        if(!xs.empty() && xs[0]=='s') xs=xs.substr(1);
        if(!ys.empty() && ys[0]=='s') ys=ys.substr(1);
        if(!ds.empty() && ds[0]=='s') ds=ds.substr(1);
        int n   = stoi(id.substr(0,3));
        int idx = stoi(id.substr(4));
        data[n].push_back({idx, stod(xs), stod(ys), stod(ds)});
    }
    for (auto& kv : data) {
        int n = kv.first;
        auto& v = kv.second;
        Cfg c;
        c.n = n;
        for (auto& tup : v) {
            int i; double x, y, d;
            tie(i,x,y,d) = tup;
            if (i < n) {
                c.x[i] = x;
                c.y[i] = y;
                c.a[i] = d;
            }
        }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(15);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++) {
                f << setfill('0') << setw(3) << n
                  << "_" << i << ",s" << c.x[i]
                  << ",s" << c.y[i]
                  << ",s" << c.a[i] << "\n";
            }
        }
    }
}

int main(int argc, char** argv) {
    string in="submission.csv", out="submission_v18.csv";
    int iters=15000, restarts=16;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a=="-i" && i+1<argc) in=argv[++i];
        else if (a=="-o" && i+1<argc) out=argv[++i];
        else if (a=="-n" && i+1<argc) iters=stoi(argv[++i]);
        else if (a=="-r" && i+1<argc) restarts=stoi(argv[++i]);
    }

    int numThreads = omp_get_max_threads();
    printf("Tree Packer v18 (PARALLEL + BACK PROP + free-area & protrusion removal, %d threads)\n", numThreads);
    printf("Iterations: %d, Restarts: %d\n", iters, restarts);
    printf("Loading %s...\n", in.c_str());

    auto cfg = loadCSV(in);
    if (cfg.empty()) {
        printf("No data!\n");
        return 1;
    }
    printf("Loaded %d configs\n", (int)cfg.size());

    double init = 0.0;
    for (auto& kv : cfg) init += kv.second.score();
    printf("Initial: %.6f\n\nPhase 1: Parallel optimization...\n\n", init);

    auto t0 = chrono::high_resolution_clock::now();
    map<int, Cfg> res;
    int totalImproved = 0;

    // Phase 1: main optimization + per-n free-area & protrusion heuristic
    for (int n = 200; n >= 1; n--) {
        if (!cfg.count(n)) continue;
        Cfg c = cfg[n];
        double os = c.score();

        int it = iters, r = restarts;
        if (n <= 10)      { it = (int)(iters * 2.5); r = restarts * 2; }
        else if (n <= 30) { it = (int)(iters * 1.8); r = (int)(restarts * 1.5); }
        else if (n <= 60) { it = (int)(iters * 1.3); r = restarts; }
        else if (n > 150) { it = (int)(iters * 0.7); r = (int)(restarts * 0.8); }

        Cfg o = optimizeParallel(c, it, max(4, r));

        // Simple backward propagation from n+1, n+2
        for (auto& kv : res) {
            int m   = kv.first;
            Cfg& pc = kv.second;
            if (m > n && m <= n + 2) {
                Cfg ad;
                ad.n = n;
                for (int i = 0; i < n; i++) {
                    ad.x[i] = pc.x[i];
                    ad.y[i] = pc.y[i];
                    ad.a[i] = pc.a[i];
                }
                ad.updAll();
                if (!ad.anyOvl()) {
                    ad = compaction(ad, 40);
                    ad = edgeSlideCompaction(ad, 8);
                    ad = localSearch(ad, 60);
                    if (!ad.anyOvl() && ad.side() < o.side()) o = ad;
                }
            }
        }

        if (o.anyOvl() || o.side() > c.side() + 1e-14) {
            o = c;
        }

        // free-area & protrusion heuristic (remove 10% 정도, 재배치)
        if (!o.anyOvl() && n >= 10) {
            Cfg oh = freeAreaHeuristic(o, 0.10, 1234567ULL + (uint64_t)n * 101ULL);
            if (!oh.anyOvl() && oh.side() < o.side() - 1e-12) {
                o = oh;
            }
        }

        res[n] = o;
        double ns = o.score();
        if (ns < os - 1e-10) {
            printf("n=%3d: %.6f -> %.6f (%.4f%%)\n", n, os, ns, (os-ns)/os*100.0);
            fflush(stdout);
            totalImproved++;
        }
    }

    // Phase 2: aggressive back propagation (removing trees)
    printf("\nPhase 2: Aggressive back propagation (removing trees)...\n\n");

    int backPropImproved = 0;
    bool changed = true;
    int passNum = 0;

    while (changed && passNum < 10) {
        changed = false;
        passNum++;

        // k vs (k-1)
        for (int k = 200; k >= 2; k--) {
            if (!res.count(k) || !res.count(k-1)) continue;

            double sideK  = res[k].side();
            double sideK1 = res[k-1].side();

            if (sideK < sideK1 - 1e-12) {
                Cfg& cfgK = res[k];
                double bestSide = sideK1;
                Cfg bestCfg     = res[k-1];

                #pragma omp parallel
                {
                    double localBestSide = bestSide;
                    Cfg localBestCfg     = bestCfg;

                    #pragma omp for schedule(dynamic)
                    for (int removeIdx = 0; removeIdx < k; removeIdx++) {
                        Cfg reduced = cfgK.removeTree(removeIdx);

                        if (!reduced.anyOvl()) {
                            reduced = squeeze(reduced);
                            reduced = compaction(reduced, 60);
                            reduced = edgeSlideCompaction(reduced, 10);
                            reduced = localSearch(reduced, 100);

                            if (!reduced.anyOvl() && reduced.side() < localBestSide) {
                                localBestSide = reduced.side();
                                localBestCfg  = reduced;
                            }
                        }
                    }

                    #pragma omp critical
                    {
                        if (localBestSide < bestSide) {
                            bestSide = localBestSide;
                            bestCfg  = localBestCfg;
                        }
                    }
                }

                if (bestSide < sideK1 - 1e-12) {
                    double oldScore = res[k-1].score();
                    double newScore = bestCfg.score();
                    res[k-1] = bestCfg;
                    printf("n=%3d: %.6f -> %.6f (from n=%d removal, %.4f%%)\n",
                           k-1, oldScore, newScore, k, (oldScore-newScore)/oldScore*100.0);
                    fflush(stdout);
                    backPropImproved++;
                    changed = true;
                }
            }
        }

        // k vs src>k (k+1..k+5)
        for (int k = 200; k >= 3; k--) {
            for (int src = k + 1; src <= min(200, k + 5); src++) {
                if (!res.count(src) || !res.count(k)) continue;

                double sideSrc = res[src].side();
                double sideK   = res[k].side();

                if (sideSrc < sideK - 1e-12) {
                    int toRemove = src - k;
                    Cfg cfgSrc   = res[src];

                    vector<vector<int>> combos;
                    if (toRemove == 1) {
                        for (int i = 0; i < src; i++) combos.push_back({i});
                    } else if (toRemove == 2 && src <= 50) {
                        for (int i = 0; i < src; i++)
                            for (int j = i+1; j < src; j++)
                                combos.push_back({i,j});
                    } else {
                        FastRNG rng((uint64_t)k * 1000ULL + (uint64_t)src);
                        for (int t = 0; t < min(200, src * 3); t++) {
                            vector<int> combo;
                            unordered_set<int> used;
                            for (int r = 0; r < toRemove; r++) {
                                int idx;
                                do { idx = rng.ri(src); } while (used.count(idx));
                                used.insert(idx);
                                combo.push_back(idx);
                            }
                            sort(combo.begin(), combo.end());
                            combos.push_back(combo);
                        }
                    }

                    double bestSide = sideK;
                    Cfg bestCfg     = res[k];

                    #pragma omp parallel
                    {
                        double localBestSide = bestSide;
                        Cfg localBestCfg     = bestCfg;

                        #pragma omp for schedule(dynamic)
                        for (int ci = 0; ci < (int)combos.size(); ci++) {
                            Cfg reduced = cfgSrc;
                            vector<int> toRem = combos[ci];
                            sort(toRem.rbegin(), toRem.rend());
                            for (int idx : toRem) {
                                reduced = reduced.removeTree(idx);
                            }

                            if (!reduced.anyOvl()) {
                                reduced = squeeze(reduced);
                                reduced = compaction(reduced, 50);
                                reduced = edgeSlideCompaction(reduced, 10);
                                reduced = localSearch(reduced, 80);

                                if (!reduced.anyOvl() && reduced.side() < localBestSide) {
                                    localBestSide = reduced.side();
                                    localBestCfg  = reduced;
                                }
                            }
                        }

                        #pragma omp critical
                        {
                            if (localBestSide < bestSide) {
                                bestSide = localBestSide;
                                bestCfg  = localBestCfg;
                            }
                        }
                    }

                    if (bestSide < sideK - 1e-12) {
                        double oldScore = res[k].score();
                        double newScore = bestCfg.score();
                        res[k] = bestCfg;
                        printf("n=%3d: %.6f -> %.6f (from n=%d removal, %.4f%%)\n",
                               k, oldScore, newScore, src, (oldScore-newScore)/oldScore*100.0);
                        fflush(stdout);
                        backPropImproved++;
                        changed = true;
                    }
                }
            }
        }

        if (changed) {
            printf("Pass %d complete, continuing...\n", passNum);
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double el = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() / 1000.0;

    double fin = 0.0;
    for (auto& kv : res) fin += kv.second.score();

    printf("\n========================================\n");
    printf("Initial: %.6f\nFinal:   %.6f\n", init, fin);
    printf("Improve: %.6f (%.4f%%)\n", init-fin, (init-fin)/init*100.0);
    printf("Phase 1 improved: %d configs\n", totalImproved);
    printf("Phase 2 back-prop improved: %d configs\n", backPropImproved);
    printf("Time:    %.1fs (with %d threads)\n", el, numThreads);
    printf("========================================\n");

    saveCSV(out, res);
    printf("Saved %s\n", out.c_str());
    return 0;
}
```

```python
INPUT_CSV = "submission.csv"

# C++ 컴파일
!OMP_NUM_THREADS=32 g++ -fopenmp -O3 -march=native -std=c++17 -o a.exe a.cpp

# 실행 (입력은 데이터셋 CSV, 출력은 working 디렉토리의 submission.csv)
!./a.exe -i $INPUT_CSV -o submission.csv -n 150000 -r 32
```

# 4. Additional Optimization

1. Based on https://www.kaggle.com/code/chistyakov/santa-2025-fix-direction and earlier version
2. Based on https://www.kaggle.com/code/seshurajup/72-73-santa-2025-jit-parallel-sa-c

```python
# Add quick, simple, and slow optimizations from @chistyakov
import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

getcontext().prec = 25
scale_factor = 1

class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))
        

    def clone(self) -> "ChristmasTree":
        return ChristmasTree(
            center_x=str(self.center_x),
            center_y=str(self.center_y),
            angle=str(self.angle),
        )    

def get_tree_list_side_lenght(tree_list: list[ChristmasTree]) -> Decimal:
    all_polygons = [t.polygon for t in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor

def get_total_score(dict_of_side_length: dict[str, Decimal]):
    score = 0
    for k, v in dict_of_side_length.items():
        score += v ** 2 / Decimal(k)
    return score

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_layout(ax, placed_trees, title):
    num_trees = len(placed_trees)
    colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    for i, tree in enumerate(placed_trees):
        # Rescale for plotting
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(val) / scale_factor for val in x_scaled]
        y = [Decimal(val) / scale_factor for val in y_scaled]
        ax.plot(x, y, color=colors[i])
        ax.fill(x, y, alpha=0.5, color=colors[i])

    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny
    # side_length = max(width, height)

    side_length = width if width > height else height
    

    square_x = minx if width >= height else minx - (side_length - width) / 2
    square_y = miny if height >= width else miny - (side_length - height) / 2
    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor='red',
        linewidth=2,
        linestyle='--',
    )
    ax.add_patch(bounding_square)

    padding = 0.5
    ax.set_xlim(
        float(square_x - Decimal(str(padding))),
        float(square_x + side_length + Decimal(str(padding))))
    ax.set_ylim(float(square_y - Decimal(str(padding))),
                float(square_y + side_length + Decimal(str(padding))))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title(f'{title}. Side: {side_length:0.8f}')

def plot_difference(layout1, layout2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plot_layout(ax1, layout1, "Orininal")
    plot_layout(ax2, layout2, "After fix")
    plt.tight_layout()
    plt.show()

def parse_csv(csv_path) -> dict[str, list[ChristmasTree]]:
    print(f'parse_csv: {csv_path=}')

    result = pd.read_csv(csv_path)
    result['x'] = result['x'].str.strip('s')
    result['y'] = result['y'].str.strip('s')
    result['deg'] = result['deg'].str.strip('s')
    result[['group_id', 'item_id']] = result['id'].str.split('_', n=2, expand=True)

    dict_of_tree_list = {}
    dict_of_side_length = {}
    for group_id, group_data in result.groupby('group_id'):
        tree_list = [ChristmasTree(center_x=row['x'], center_y=row['y'], angle=row['deg']) for _, row in group_data.iterrows()]
        dict_of_tree_list[group_id] = tree_list
        dict_of_side_length[group_id] = get_tree_list_side_lenght(tree_list)

    return dict_of_tree_list, dict_of_side_length

from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar
import numpy as np

def calculate_bbox_side_at_angle(angle_deg, points):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix_T = np.array([[c, s], [-s, c]])
    rotated_points = points.dot(rot_matrix_T)
    min_xy = np.min(rotated_points, axis=0); max_xy = np.max(rotated_points, axis=0)
    return max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

def optimize_rotation(trees):
    all_points = []
    for tree in trees: all_points.extend(list(tree.polygon.exterior.coords))
    points_np = np.array(all_points)
    
    hull_points = points_np[ConvexHull(points_np).vertices]
    
    initial_side = calculate_bbox_side_at_angle(0, hull_points)
    
    res = minimize_scalar(lambda a: calculate_bbox_side_at_angle(a, hull_points), bounds=(0.001, 89.999), method='bounded')
    found_angle_deg = res.x
    found_side = res.fun
    
    improvement = initial_side - found_side
    
    EPSILON = 1e-5

    if improvement > EPSILON:
        best_angle_deg = found_angle_deg
        best_side = Decimal(found_side) / scale_factor
    else:
        best_angle_deg = 0.0
        best_side = Decimal(initial_side) / scale_factor
        
    return best_side, best_angle_deg

def apply_rotation(trees, angle_deg):
    if not trees or abs(angle_deg) < 1e-9: return [t.clone() for t in trees]
    
    bounds = [t.polygon.bounds for t in trees]
    min_x = min(b[0] for b in bounds); min_y = min(b[1] for b in bounds)
    max_x = max(b[2] for b in bounds); max_y = max(b[3] for b in bounds)
    rotation_center = np.array([(min_x + max_x) / 2.0, (min_y + max_y) / 2.0])
    
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    points = np.array([[float(t.center_x), float(t.center_y)] for t in trees])
    shifted = points - rotation_center
    rotated = shifted.dot(rot_matrix.T) + rotation_center
    
    rotated_trees = []
    for i in range(len(trees)):
        new_tree = ChristmasTree(Decimal(rotated[i, 0]) , Decimal(rotated[i, 1]), Decimal(trees[i].angle + Decimal(angle_deg)))
        rotated_trees.append(new_tree)
    return rotated_trees

from shapely.geometry import box

def get_bbox_touching_tree_indices(tree_list: list[ChristmasTree]) -> list[int]:
    """
    Given a list of trees, this function:

      1. Computes the minimal axis-aligned bounding box around all trees.
      2. Returns the list of indices of trees whose boundaries touch
         the boundary of that bounding box.

    Returns:
        touching_indices: list[int]  -- indices in tree_list
    """

    if not tree_list:
        return []

    # Collect polygons
    polys = [t.polygon for t in tree_list]

    # Compute global bounding box from all polygon bounds
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)

    bbox = box(minx, miny, maxx, maxy)

    # Check boundary intersection: only trees touching the box border
    touching_indices = [
        i
        for i, poly in enumerate(polys)
        if poly.boundary.intersects(bbox.boundary)
    ]

    return touching_indices
```

```python
# Load current best solution
current_solution_path = 'submission.csv'
dict_of_tree_list, dict_of_side_length = parse_csv(current_solution_path)

# Calculate current total score
current_score = get_total_score(dict_of_side_length)
print(f'\n{current_score=:0.8f}')

for group_id_main in range(200, 2, -1):
    group_id_main = f'{int(group_id_main):03n}'

    initial_trees = dict_of_tree_list[group_id_main]
    best_side, best_angle_deg = optimize_rotation(initial_trees)
    fixed_trees = apply_rotation(initial_trees, best_angle_deg)

    cur_side = dict_of_side_length[group_id_main]
    if best_side < cur_side:
        print(f'n={int(group_id_main)}, {best_side:0.8f} -> {cur_side:0.8f} ({best_side-cur_side:0.8f})')

        plot_difference(initial_trees, fixed_trees)
        dict_of_tree_list[group_id_main] = fixed_trees
        dict_of_side_length[group_id_main] = best_side

new_score = get_total_score(dict_of_side_length)
print(f'\n{new_score=:0.8f} ({current_score-new_score:0.8f})')

# Save results
tree_data = []
for group_name, tree_list in dict_of_tree_list.items():
    for item_id, tree in enumerate(tree_list):
        tree_data.append({
            'id': f'{group_name}_{item_id}',
            'x': f's{tree.center_x}',
            'y': f's{tree.center_y}',
            'deg': f's{tree.angle}'
        })
tree_data = pd.DataFrame(tree_data)
tree_data.to_csv('submission.csv', index=False)
```

```python
scale_factor = Decimal('1e18') #1
```

```python
# Build report with possible optimizations
report = pd.DataFrame(pd.Series(dict_of_side_length), columns=['side_length'])
report['side_length_prev'] = report['side_length'].shift(1)
report['side_length_increase'] = report['side_length'] - report['side_length_prev']
report = report[report['side_length_increase'] <= 0].sort_index(ascending=False)
print('Solutions with easy optimization')
print(report)

for group_id_main in range(200, 2, -1):
    group_id_main = f'{int(group_id_main):03n}'
#    print(f'\nCurrent box: {group_id_main}')
    
    candidate_tree_list = [tree.clone() for tree in dict_of_tree_list[group_id_main]]
    candidate_tree_list = sorted(candidate_tree_list, key=lambda a: -a.center_y)

    while len(candidate_tree_list) > 1:
        group_id_prev = f'{len(candidate_tree_list) - 1:03n}'
        best_side_length = dict_of_side_length[group_id_prev]
        best_side_length_temp = 100
        best_tree_idx_to_delete = None

        # Try to delete each tree one by one and select the best option
        tree_idx_list = get_bbox_touching_tree_indices(candidate_tree_list)      
        for tree_idx_to_delete in tree_idx_list:
            
            candidate_tree_list_short = [tree.clone() for tree in candidate_tree_list]
            del candidate_tree_list_short[tree_idx_to_delete]
    
            candidate_side_length = get_tree_list_side_lenght(candidate_tree_list_short)
                
            if candidate_side_length < best_side_length_temp:
                best_side_length_temp = candidate_side_length
                best_tree_idx_to_delete = tree_idx_to_delete

        # Save the best
        if best_tree_idx_to_delete is not None:
            del candidate_tree_list[best_tree_idx_to_delete]
#            print(len(candidate_tree_list), end=' ')

            if candidate_side_length < best_side_length:
                print(f'\nCurrent box: {group_id_main}')                
                print(f'\nimprovement {best_side_length:0.8f} -> {candidate_side_length:0.8f}')
            
                dict_of_tree_list[group_id_prev] = [tree.clone() for tree in candidate_tree_list]
                dict_of_side_length[group_id_prev] = get_tree_list_side_lenght(dict_of_tree_list[group_id_prev])

        break
    
# Recalculate current total score
new_score = get_total_score(dict_of_side_length)
print(f'\n{current_score=:0.12f} {new_score=:0.12f} ({current_score - new_score:0.12f})')

# Save results
tree_data = []
for group_name, tree_list in dict_of_tree_list.items():
    for item_id, tree in enumerate(tree_list):
        tree_data.append({
            'id': f'{group_name}_{item_id}',
            'x': f's{tree.center_x}',
            'y': f's{tree.center_y}',
            'deg': f's{tree.angle}'
        })
tree_data = pd.DataFrame(tree_data)
tree_data.to_csv('submission.csv', index=False)
print("**Done**")
```

```python
# Build slow optimization

current_solution_path = 'submission.csv'
dict_of_tree_list, dict_of_side_length = parse_csv(current_solution_path)

# Calculate current total score
current_score = get_total_score(dict_of_side_length)

for group_id_main in range(200, 2, -1):
    group_id_main = f'{int(group_id_main):03n}'
#    print(f'\nCurrent box: {group_id_main}')

    candidate_tree_list = [tree.clone() for tree in dict_of_tree_list[group_id_main]]

    while len(candidate_tree_list) > 1:
        group_id_prev = f'{len(candidate_tree_list) - 1:03n}'
        best_side_length = dict_of_side_length[group_id_prev]
        best_side_length_temp = 100
        best_tree_idx_to_delete = None

        # Try to delete each tree one by one and select the best option
        tree_idx_list = get_bbox_touching_tree_indices(candidate_tree_list)      
        for tree_idx_to_delete in tree_idx_list:
            
            candidate_tree_list_short = [tree.clone() for tree in candidate_tree_list]
            del candidate_tree_list_short[tree_idx_to_delete]
    
            candidate_side_length = get_tree_list_side_lenght(candidate_tree_list_short)
                
            if candidate_side_length < best_side_length_temp:
                best_side_length_temp = candidate_side_length
                best_tree_idx_to_delete = tree_idx_to_delete

        # Save the best
        if best_tree_idx_to_delete is not None:
            # print(F'   Deleting: {best_tree_idx_to_delete}')
            del candidate_tree_list[best_tree_idx_to_delete]
            print(len(candidate_tree_list), end=' ')

            if candidate_side_length < best_side_length:
                print(f'\nCurrent box: {group_id_main}; \nimprovement {best_side_length:0.8f} -> {candidate_side_length:0.8f}')
            
                dict_of_tree_list[group_id_prev] = [tree.clone() for tree in candidate_tree_list]
                dict_of_side_length[group_id_prev] = get_tree_list_side_lenght(dict_of_tree_list[group_id_prev])

        if int(group_id_main) - int(group_id_prev) > 5:
            break

    # break

    
# Recalculate current total score
new_score = get_total_score(dict_of_side_length)
print(f'\n{current_score=:0.8f} {new_score=:0.8f} ({current_score - new_score:0.8f})')
```

Based on codes from @seshurajup (https://www.kaggle.com/code/seshurajup/72-73-santa-2025-jit-parallel-sa-c)

```python
%%writefile /kaggle/working/sa_v1_parallel.cpp
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o sa_v1_parallel sa_v1_parallel.cpp
// Run: ./sa_v1_parallel -i /kaggle/working/submission_ensemble.csv -o /kaggle/working/submission.csv -n 15000 -r 5

#include <bits/stdc++.h>
using namespace std;
using namespace chrono;
#include <omp.h>

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;
const double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
const double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

mt19937_64 rng(42);
uniform_real_distribution<double> U(0, 1);
inline double rf() { return U(rng); }
inline int ri(int n) { return rng() % n; }

struct Pt { double x, y; };

struct Poly {
    Pt p[NV];
    double x0, y0, x1, y1;
    void bbox() {
        x0 = x1 = p[0].x; y0 = y1 = p[0].y;
        for (int i = 1; i < NV; i++) {
            x0 = min(x0, p[i].x); x1 = max(x1, p[i].x);
            y0 = min(y0, p[i].y); y1 = max(y1, p[i].y);
        }
    }
};

Poly getPoly(double cx, double cy, double deg) {
    Poly q;
    double r = deg * PI / 180, c = cos(r), s = sin(r);
    for (int i = 0; i < NV; i++) {
        q.p[i].x = TX[i] * c - TY[i] * s + cx;
        q.p[i].y = TX[i] * s + TY[i] * c + cy;
    }
    q.bbox();
    return q;
}

bool pip(double px, double py, const Poly& q) {
    bool in = false;
    int j = NV - 1;
    for (int i = 0; i < NV; i++) {
        if ((q.p[i].y > py) != (q.p[j].y > py) &&
            px < (q.p[j].x - q.p[i].x) * (py - q.p[i].y) / (q.p[j].y - q.p[i].y) + q.p[i].x)
            in = !in;
        j = i;
    }
    return in;
}

bool segInt(Pt a, Pt b, Pt c, Pt d) {
    auto ccw = [](Pt p, Pt q, Pt r) { return (r.y - p.y) * (q.x - p.x) > (q.y - p.y) * (r.x - p.x); };
    return ccw(a, c, d) != ccw(b, c, d) && ccw(a, b, c) != ccw(a, b, d);
}

bool overlap(const Poly& a, const Poly& b) {
    if (a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0) return false;
    for (int i = 0; i < NV; i++) {
        if (pip(a.p[i].x, a.p[i].y, b)) return true;
        if (pip(b.p[i].x, b.p[i].y, a)) return true;
    }
    for (int i = 0; i < NV; i++)
        for (int j = 0; j < NV; j++)
            if (segInt(a.p[i], a.p[(i + 1) % NV], b.p[j], b.p[(j + 1) % NV])) return true;
    return false;
}

struct Cfg {
    int n;
    double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    void upd(int i) { pl[i] = getPoly(x[i], y[i], a[i]); }
    void updAll() { for (int i = 0; i < n; i++) upd(i); }
    bool hasOvl(int i) const {
        for (int j = 0; j < n; j++) if (i != j && overlap(pl[i], pl[j])) return true;
        return false;
    }
    bool hasOvlPair(int i, int j) const {
        if (overlap(pl[i], pl[j])) return true;
        for (int k = 0; k < n; k++) {
            if (k != i && k != j) {
                if (overlap(pl[i], pl[k]) || overlap(pl[j], pl[k])) return true;
            }
        }
        return false;
    }
    bool anyOvl() const {
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (overlap(pl[i], pl[j])) return true;
        return false;
    }
    double side() const {
        if (!n) return 0;
        double x0 = pl[0].x0, x1 = pl[0].x1, y0 = pl[0].y0, y1 = pl[0].y1;
        for (int i = 1; i < n; i++) {
            x0 = min(x0, pl[i].x0); x1 = max(x1, pl[i].x1);
            y0 = min(y0, pl[i].y0); y1 = max(y1, pl[i].y1);
        }
        return max(x1 - x0, y1 - y0);
    }
    double score() const { double s = side(); return s * s / n; }
    pair<double, double> centroid() const {
        double sx = 0, sy = 0;
        for (int i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
        return {sx / n, sy / n};
    }
    tuple<double, double, double, double> getBBox() const {
        double gx0 = pl[0].x0, gx1 = pl[0].x1, gy0 = pl[0].y0, gy1 = pl[0].y1;
        for (int i = 1; i < n; i++) {
            gx0 = min(gx0, pl[i].x0); gx1 = max(gx1, pl[i].x1);
            gy0 = min(gy0, pl[i].y0); gy1 = max(gy1, pl[i].y1);
        }
        return {gx0, gy0, gx1, gy1};
    }
    vector<int> findCornerTrees() const {
        auto [gx0, gy0, gx1, gy1] = getBBox();
        double eps = 0.01;
        vector<int> corners;
        for (int i = 0; i < n; i++) {
            if (abs(pl[i].x0 - gx0) < eps || abs(pl[i].x1 - gx1) < eps ||
                abs(pl[i].y0 - gy0) < eps || abs(pl[i].y1 - gy1) < eps) {
                corners.push_back(i);
            }
        }
        return corners;
    }
};

Cfg sa_v3(Cfg c, int iter, double T0, double Tm, double ms, double rs, uint64_t seed) {
    rng.seed(seed);
    Cfg best = c, cur = c;
    double bs = best.side(), cs = bs, T = T0;
    double alpha = pow(Tm / T0, 1.0 / iter);
    int noImp = 0;
    for (int it = 0; it < iter; it++) {
        int moveType = ri(8);
        double sc = T / T0;
        if (moveType < 4) {
            int i = ri(c.n);
            double ox = cur.x[i], oy = cur.y[i], oa = cur.a[i];
            auto [cx, cy] = cur.centroid();
            if (moveType == 0) {
                cur.x[i] += (rf() - 0.5) * 2 * ms * sc;
                cur.y[i] += (rf() - 0.5) * 2 * ms * sc;
            } else if (moveType == 1) {
                double dx = cx - cur.x[i], dy = cy - cur.y[i];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    double st = rf() * ms * sc;
                    cur.x[i] += dx / d * st;
                    cur.y[i] += dy / d * st;
                }
            } else if (moveType == 2) {
                cur.a[i] += (rf() - 0.5) * 2 * rs * sc;
                cur.a[i] = fmod(cur.a[i] + 360, 360.0);
            } else {
                cur.x[i] += (rf() - 0.5) * ms * sc;
                cur.y[i] += (rf() - 0.5) * ms * sc;
                cur.a[i] += (rf() - 0.5) * rs * sc;
                cur.a[i] = fmod(cur.a[i] + 360, 360.0);
            }
            cur.upd(i);
            if (cur.hasOvl(i)) {
                cur.x[i] = ox; cur.y[i] = oy; cur.a[i] = oa;
                cur.upd(i);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 4 && c.n > 1) {
            int i = ri(c.n), j = ri(c.n);
            while (j == i) j = ri(c.n);
            double oxi = cur.x[i], oyi = cur.y[i];
            double oxj = cur.x[j], oyj = cur.y[j];
            cur.x[i] = oxj; cur.y[i] = oyj;
            cur.x[j] = oxi; cur.y[j] = oyi;
            cur.upd(i); cur.upd(j);
            if (cur.hasOvlPair(i, j)) {
                cur.x[i] = oxi; cur.y[i] = oyi;
                cur.x[j] = oxj; cur.y[j] = oyj;
                cur.upd(i); cur.upd(j);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 5) {
            int i = ri(c.n);
            double ox = cur.x[i], oy = cur.y[i];
            auto [gx0, gy0, gx1, gy1] = cur.getBBox();
            double bcx = (gx0 + gx1) / 2, bcy = (gy0 + gy1) / 2;
            double dx = bcx - cur.x[i], dy = bcy - cur.y[i];
            double d = sqrt(dx * dx + dy * dy);
            if (d > 1e-6) {
                double st = rf() * ms * sc * 0.5;
                cur.x[i] += dx / d * st;
                cur.y[i] += dy / d * st;
            }
            cur.upd(i);
            if (cur.hasOvl(i)) {
                cur.x[i] = ox; cur.y[i] = oy;
                cur.upd(i);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else if (moveType == 6) {
            auto corners = cur.findCornerTrees();
            if (!corners.empty()) {
                int idx = corners[ri(corners.size())];
                double ox = cur.x[idx], oy = cur.y[idx], oa = cur.a[idx];
                auto [gx0, gy0, gx1, gy1] = cur.getBBox();
                double bcx = (gx0 + gx1) / 2, bcy = (gy0 + gy1) / 2;
                double dx = bcx - cur.x[idx], dy = bcy - cur.y[idx];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                double st = rf() * ms * sc * 0.3;
                    cur.x[idx] += dx / d * st;
                    cur.y[idx] += dy / d * st;
                    cur.a[idx] += (rf() - 0.5) * rs * sc * 0.5;
                    cur.a[idx] = fmod(cur.a[idx] + 360, 360.0);
                }
                cur.upd(idx);
                if (cur.hasOvl(idx)) {
                    cur.x[idx] = ox; cur.y[idx] = oy; cur.a[idx] = oa;
                    cur.upd(idx);
                    noImp++;
                    T *= alpha; if (T < Tm) T = Tm;
                    continue;
                }
            } else {
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        } else {
            int i = ri(c.n);
            int j = (i + 1) % c.n;
            double oxi = cur.x[i], oyi = cur.y[i];
            double oxj = cur.x[j], oyj = cur.y[j];
            double dx = (rf() - 0.5) * ms * sc * 0.5;
            double dy = (rf() - 0.5) * ms * sc * 0.5;
            cur.x[i] += dx; cur.y[i] += dy;
            cur.x[j] += dx; cur.y[j] += dy;
            cur.upd(i); cur.upd(j);
            if (cur.hasOvlPair(i, j)) {
                cur.x[i] = oxi; cur.y[i] = oyi;
                cur.x[j] = oxj; cur.y[j] = oyj;
                cur.upd(i); cur.upd(j);
                noImp++;
                T *= alpha; if (T < Tm) T = Tm;
                continue;
            }
        }
        double ns = cur.side();
        double delta = ns - cs;
        if (delta < 0 || rf() < exp(-delta / T)) {
            cs = ns;
            if (ns < bs) {
                bs = ns;
                best = cur;
                noImp = 0;
            } else {
                noImp++;
            }
        } else {
            cur = best;
            cs = bs;
            noImp++;
        }
        if (noImp > 600) {
            T = min(T * 3.0, T0 * 0.7);
            noImp = 0;
        }
        T *= alpha;
        if (T < Tm) T = Tm;
    }
    return best;
}

Cfg ls_v3(Cfg c, int iter) {
    Cfg best = c;
    double bs = best.side();
    double ps[] = {0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002};
    double rs[] = {15.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25};
    int dx[] = {1, -1, 0, 0, 1, 1, -1, -1};
    int dy[] = {0, 0, 1, -1, 1, -1, 1, -1};
    for (int it = 0; it < iter; it++) {
        bool imp = false;
        auto corners = best.findCornerTrees();
        for (int ci : corners) {
            for (double st : ps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[ci], oy = best.y[ci];
                    best.x[ci] += dx[d] * st;
                    best.y[ci] += dy[d] * st;
                    best.upd(ci);
                    if (!best.hasOvl(ci)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.x[ci] = ox; best.y[ci] = oy;
                            best.upd(ci);
                        }
                    } else {
                        best.x[ci] = ox; best.y[ci] = oy;
                        best.upd(ci);
                    }
                }
            }
            for (double st : rs) {
                for (double da : {st, -st}) {
                    double oa = best.a[ci];
                    best.a[ci] = fmod(best.a[ci] + da + 360, 360.0);
                    best.upd(ci);
                    if (!best.hasOvl(ci)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.a[ci] = oa;
                            best.upd(ci);
                        }
                    } else {
                        best.a[ci] = oa;
                        best.upd(ci);
                    }
                }
            }
        }
        set<int> cornerSet(corners.begin(), corners.end());
        for (int i = 0; i < c.n; i++) {
            if (cornerSet.count(i)) continue;
            for (double st : ps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    best.x[i] += dx[d] * st;
                    best.y[i] += dy[d] * st;
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.x[i] = ox; best.y[i] = oy;
                            best.upd(i);
                        }
                    } else {
                        best.x[i] = ox; best.y[i] = oy;
                        best.upd(i);
                    }
                }
            }
            for (double st : rs) {
                for (double da : {st, -st}) {
                    double oa = best.a[i];
                    best.a[i] = fmod(best.a[i] + da + 360, 360.0);
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        double ns = best.side();
                        if (ns < bs - 1e-10) {
                            bs = ns;
                            imp = true;
                        } else {
                            best.a[i] = oa;
                            best.upd(i);
                        }
                    } else {
                        best.a[i] = oa;
                        best.upd(i);
                    }
                }
            }
        }
        if (!imp) break;
    }
    return best;
}

Cfg perturb(Cfg c, double strength, uint64_t seed) {
    rng.seed(seed);
    int numPerturb = max(1, (int)(c.n * 0.15));
    for (int k = 0; k < numPerturb; k++) {
        int i = ri(c.n);
        c.x[i] += (rf() - 0.5) * strength;
        c.y[i] += (rf() - 0.5) * strength;
        c.a[i] = fmod(c.a[i] + (rf() - 0.5) * 60 + 360, 360.0);
    }
    c.updAll();
    for (int iter = 0; iter < 100; iter++) {
        bool fixed = true;
        for (int i = 0; i < c.n; i++) {
            if (c.hasOvl(i)) {
                fixed = false;
                double cx = 0, cy = 0;
                for (int j = 0; j < c.n; j++) { cx += c.x[j]; cy += c.y[j]; }
                cx /= c.n; cy /= c.n;
                double dx = cx - c.x[i], dy = cy - c.y[i];
                double d = sqrt(dx * dx + dy * dy);
                if (d > 1e-6) {
                    c.x[i] -= dx / d * 0.02;
                    c.y[i] -= dy / d * 0.02;
                }
                c.a[i] = fmod(c.a[i] + rf() * 20 - 10 + 360, 360.0);
                c.upd(i);
            }
        }
        if (fixed) break;
    }
    return c;
}

Cfg fractional_translation(Cfg c, int max_iter = 200) {
    Cfg best = c;
    double bs = best.side();
    double frac_steps[] = {0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001};
    double dx[] = {0, 0, 1, -1, 1, 1, -1, -1};
    double dy[] = {1, -1, 0, 0, 1, -1, 1, -1};
    for (int iter = 0; iter < max_iter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            for (double step : frac_steps) {
                for (int d = 0; d < 8; d++) {
                    double ox = best.x[i], oy = best.y[i];
                    best.x[i] += dx[d] * step;
                    best.y[i] += dy[d] * step;
                    best.upd(i);
                    if (!best.hasOvl(i)) {
                        double ns = best.side();
                        if (ns < bs - 1e-12) {
                            bs = ns;
                            improved = true;
                        } else {
                            best.x[i] = ox; best.y[i] = oy; best.upd(i);
                        }
                    } else {
                        best.x[i] = ox; best.y[i] = oy; best.upd(i);
                    }
                }
            }
        }
        if (!improved) break;
    }
    return best;
}

Cfg opt_v3(Cfg c, int nr, int si) {
    Cfg best = c;
    double bs = best.side();
    vector<pair<double, Cfg>> pop;
    pop.push_back({bs, c});
    for (int r = 0; r < nr; r++) {
        Cfg start;
        if (r == 0) {
            start = c;
        } else if (r < (int)pop.size()) {
            start = pop[r % pop.size()].second;
        } else {
            start = perturb(pop[0].second, 0.1 + 0.05 * (r % 3), 42 + r * 1000 + c.n);
        }
        Cfg o = sa_v3(start, si, 1.0, 0.000005, 0.25, 70.0, 42 + r * 1000 + c.n);
        o = ls_v3(o, 300);
        o = fractional_translation(o, 150);
        double s = o.side();
        pop.push_back({s, o});
        sort(pop.begin(), pop.end(), [](const pair<double, Cfg>& a, const pair<double, Cfg>& b) {
            return a.first < b.first;
        });
        if (pop.size() > 3) pop.resize(3);
        if (s < bs) {
            bs = s;
            best = o;
        }
    }
    return best;
}

map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) { cerr << "Cannot open " << fn << endl; return cfg; }
    string ln; getline(f, ln);
    map<int, vector<tuple<int, double, double, double>>> data;
    while (getline(f, ln)) {
        auto p1 = ln.find(','), p2 = ln.find(',', p1 + 1), p3 = ln.find(',', p2 + 1);
        string id = ln.substr(0, p1);
        string xs = ln.substr(p1 + 1, p2 - p1 - 1);
        string ys = ln.substr(p2 + 1, p3 - p2 - 1);
        string ds = ln.substr(p3 + 1);
        if (xs[0] == 's') xs = xs.substr(1);
        if (ys[0] == 's') ys = ys.substr(1);
        if (ds[0] == 's') ds = ds.substr(1);
        int n = stoi(id.substr(0, 3)), idx = stoi(id.substr(4));
        data[n].push_back({idx, stod(xs), stod(ys), stod(ds)});
    }
    for (auto& [n, v] : data) {
        Cfg c; c.n = n;
        for (auto& [i, x, y, d] : v) if (i < n) { c.x[i] = x; c.y[i] = y; c.a[i] = d; }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(15);
    f << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++)
                f << setfill('0') << setw(3) << n << "_" << i
                  << ",s" << c.x[i] << ",s" << c.y[i] << ",s" << c.a[i] << "\n";
        }
    }
}

// ← put this somewhere near the top (after includes)
void ensure_dir() {
    #ifdef _WIN32
        system("if not exist solutions mkdir solutions");
    #else
        system("mkdir -p solutions");
    #endif
}

// ← REPLACE YOUR WHOLE main() WITH THIS ONE
int main(int argc, char** argv) {
    ensure_dir();

    string in = "./submission_best.csv";
    int si = 20000, nr = 80;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-i" && i + 1 < argc) in = argv[++i];
        else if (a == "-n" && i + 1 < argc) si = stoi(argv[++i]);
        else if (a == "-r" && i + 1 < argc) nr = stoi(argv[++i]);
    }

    int threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    cout << "Using " << threads << " threads — endless mode ON\n";

    auto cfg = loadCSV(in);
    if (cfg.empty()) { cerr << "No data!\n"; return 1; }

    map<int, Cfg> best_so_far = cfg;
    double global_best_score = 0;
    for (const auto& [n, c] : best_so_far) global_best_score += c.score();

    cout << fixed << setprecision(6);
    cout << "Starting score: " << global_best_score << "\n\n";

    int generation = 0;
    int no_improvement_count = 0;
    int max_retries = 3; // KEEP High
    int max_retry_retries = 3; // KEEP High
    int retry_count = 0;
    while (true) {
        generation++;
        cout << "\n=== Generation " << generation << " ===" << endl;

        map<int, Cfg> current = best_so_far;        // start from current best every round
        double round_start_score = global_best_score;

        {
            map<int, Cfg> local;

            #pragma omp for schedule(dynamic, 1) nowait
            for (int n = 1; n <= 200; n++) {
                if (!current.count(n)) continue;

                Cfg c = current[n];

                int it = si, r = nr;
                if (n <= 20) { r = max(6, nr); it = int(si * 1.5); }
                else if (n <= 50) { r = max(5, nr); it = int(si * 1.3); }
                else if (n > 150) { r = max(4, nr); it = int(si * 0.8); }

                Cfg candidate = opt_v3(c, r, it);
                candidate = fractional_translation(candidate, 120);

                local[n] = candidate;
            }

            {
                for (auto& p : local) {
                    int n = p.first;
                    Cfg& cand = p.second;
                    double old_n_score = current[n].score();
                    double new_n_score = cand.score();

                    if (new_n_score < old_n_score - 1e-9) {
                        current[n] = cand;
                        double improvement = (old_n_score - new_n_score) / old_n_score * 100.0;
                        cout << "n=" << setw(3) << n << "  "
                             << old_n_score << " → " << new_n_score
                             << "  (+" << fixed << setprecision(4) << improvement << "%)" << endl;
                    }
                }
            }
        }

        double new_total = 0;
        for (const auto& [n, c] : current) new_total += c.score();

        bool improved = (new_total < global_best_score - 1e-8);

        if (improved) {
            global_best_score = new_total;
            best_so_far = current;

            char filename[64];
            snprintf(filename, sizeof(filename), "solutions/submission_%.6f.csv", global_best_score);

            saveCSV(filename, best_so_far);

            cout << "\nNEW GLOBAL BEST! → " << global_best_score
                 << "   saved as  " << filename << endl;
            no_improvement_count = 0;
        } else {
            cout << "Generation " << generation << " finished — no global improvement ("
                 << new_total << ")" << endl;
            no_improvement_count += 1;
        }
        retry_count += 1;
        if (no_improvement_count > max_retries) {
            break;
        }

        if (retry_count > max_retry_retries) {
            break;
        }
        
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    return 0;
}
```

```python
! g++ -O3 -march=native -std=c++17 -fopenmp -o /kaggle/working/sa_v1_parallel /kaggle/working/sa_v1_parallel.cpp
```

```python
! /kaggle/working/sa_v1_parallel -i /kaggle/working/submission.csv -o /kaggle/working/submission.csv -n 15000 -r 5
```

```python
# Done
```