# Santa Claude

- **Author:** Smart Manoj
- **Votes:** 384
- **Ref:** smartmanoj/santa-claude
- **URL:** https://www.kaggle.com/code/smartmanoj/santa-claude
- **Last run:** 2025-12-28 07:48:03.217000

---

```python
!nproc
```

```python
# Draft version
```

# [Python version](https://www.kaggle.com/code/smartmanoj/santa-claude-code?scriptVersionId=281257757)

```python
%%writefile a.cpp
// Tree Packer v21 - ENHANCED v19 with SWAP MOVES + MULTI-START
// All n values (1-200) processed in parallel + aggressive exploration
// NEW: Swap move operator, multi-angle restarts, higher temperature SA
// Compile: g++ -O3 -march=native -std=c++17 -fopenmp -o tree_packer_v21 tree_packer_v21.cpp

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

alignas(64) const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

struct FastRNG {
    uint64_t s[2];
    FastRNG(uint64_t seed = 42) {
        s[0] = seed ^ 0x853c49e6748fea9bULL;
        s[1] = (seed * 0x9e3779b97f4a7c15ULL) ^ 0xc4ceb9fe1a85ec53ULL;
    }
    inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    inline uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1], r = s0 + s1;
        s1 ^= s0; s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); s[1] = rotl(s1, 37);
        return r;
    }
    inline long double rf() { return (next() >> 11) * 0x1.0p-53L; }
    inline long double rf2() { return rf() * 2.0L - 1.0L; }
    inline int ri(int n) { return next() % n; }
    inline long double gaussian() {
        long double u1 = rf() + 1e-10L, u2 = rf();
        return sqrtl(-2.0L * logl(u1)) * cosl(2.0L * PI * u2);
    }
};

struct Poly {
    long double px[NV], py[NV];
    long double x0, y0, x1, y1;
};

inline void getPoly(long double cx, long double cy, long double deg, Poly& q) {
    long double rad = deg * (PI / 180.0L);
    long double s = sinl(rad), c = cosl(rad);
    long double minx = 1e9L, miny = 1e9L, maxx = -1e9L, maxy = -1e9L;
    for (int i = 0; i < NV; i++) {
        long double x = TX[i] * c - TY[i] * s + cx;
        long double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x; q.py[i] = y;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

inline bool pip(long double px, long double py, const Poly& q) {
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

inline bool segInt(long double ax, long double ay, long double bx, long double by,
                   long double cx, long double cy, long double dx, long double dy) {
    long double d1 = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx);
    long double d2 = (dx-cx)*(by-cy) - (dy-cy)*(bx-cx);
    long double d3 = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax);
    long double d4 = (bx-ax)*(dy-ay) - (by-ay)*(dx-ax);
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
    long double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    long double gx0, gy0, gx1, gy1;

    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }
    inline void updAll() { for (int i = 0; i < n; i++) upd(i); updGlobal(); }

    inline void updGlobal() {
        gx0 = gy0 = 1e9L; gx1 = gy1 = -1e9L;
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

    inline long double side() const { return max(gx1 - gx0, gy1 - gy0); }
    inline long double score() const { long double s = side(); return s * s / n; }

    void getBoundary(vector<int>& b) const {
        b.clear();
        long double eps = 0.01L;
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

// Squeeze
Cfg squeeze(Cfg c) {
    long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
    for (long double scale = 0.9995L; scale >= 0.98L; scale -= 0.0005L) {
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

// Compaction
Cfg compaction(Cfg c, int iters) {
    long double bs = c.side();
    for (int it = 0; it < iters; it++) {
        long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            long double ox = c.x[i], oy = c.y[i];
            long double dx = cx - c.x[i], dy = cy - c.y[i];
            long double d = sqrtl(dx*dx + dy*dy);
            if (d < 1e-6L) continue;
            for (long double step : {0.02L, 0.008L, 0.003L, 0.001L, 0.0004L}) {
                c.x[i] = ox + dx/d * step; c.y[i] = oy + dy/d * step; c.upd(i);
                if (!c.hasOvl(i)) {
                    c.updGlobal();
                    if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; ox = c.x[i]; oy = c.y[i]; }
                    else { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
                } else { c.x[i] = ox; c.y[i] = oy; c.upd(i); }
            }
        }
        c.updGlobal();
        if (!improved) break;
    }
    return c;
}

// Local search
Cfg localSearch(Cfg c, int maxIter) {
    long double bs = c.side();
    const long double steps[] = {0.01L, 0.004L, 0.0015L, 0.0006L, 0.00025L, 0.0001L};
    const long double rots[] = {5.0L, 2.0L, 0.8L, 0.3L, 0.1L};
    const int dx[] = {1,-1,0,0,1,1,-1,-1};
    const int dy[] = {0,0,1,-1,1,-1,1,-1};

    for (int iter = 0; iter < maxIter; iter++) {
        bool improved = false;
        for (int i = 0; i < c.n; i++) {
            long double cx = (c.gx0 + c.gx1) / 2.0L, cy = (c.gy0 + c.gy1) / 2.0L;
            long double ddx = cx - c.x[i], ddy = cy - c.y[i];
            long double dist = sqrtl(ddx*ddx + ddy*ddy);
            if (dist > 1e-6L) {
                for (long double st : steps) {
                    long double ox = c.x[i], oy = c.y[i];
                    c.x[i] += ddx/dist * st; c.y[i] += ddy/dist * st; c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); } }
                    else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            for (long double st : steps) {
                for (int d = 0; d < 8; d++) {
                    long double ox=c.x[i], oy=c.y[i];
                    c.x[i] += dx[d]*st; c.y[i] += dy[d]*st; c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.x[i]=ox; c.y[i]=oy; c.upd(i); c.updGlobal(); } }
                    else { c.x[i]=ox; c.y[i]=oy; c.upd(i); }
                }
            }
            for (long double rt : rots) {
                for (long double da : {rt, -rt}) {
                    long double oa = c.a[i]; c.a[i] += da;
                    while (c.a[i] < 0) c.a[i] += 360.0L;
                    while (c.a[i] >= 360.0L) c.a[i] -= 360.0L;
                    c.upd(i);
                    if (!c.hasOvl(i)) { c.updGlobal(); if (c.side() < bs - 1e-12L) { bs = c.side(); improved = true; }
                        else { c.a[i]=oa; c.upd(i); c.updGlobal(); } }
                    else { c.a[i]=oa; c.upd(i); }
                }
            }
        }
        if (!improved) break;
    }
    return c;
}

// V21 NEW: Swap move operator
bool swapTrees(Cfg& c, int i, int j) {
    if (i == j || i >= c.n || j >= c.n) return false;
    swap(c.x[i], c.x[j]);
    swap(c.y[i], c.y[j]);
    swap(c.a[i], c.a[j]);
    c.upd(i);
    c.upd(j);
    return !c.hasOvl(i) && !c.hasOvl(j);
}

// SA optimization (V21: Enhanced with swap moves)
Cfg sa_opt(Cfg c, int iter, long double T0, long double Tm, uint64_t seed) {
    FastRNG rng(seed);
    Cfg best = c, cur = c;
    long double bs = best.side(), cs = bs, T = T0;
    long double alpha = powl(Tm / T0, 1.0L / iter);
    int noImp = 0;

    for (int it = 0; it < iter; it++) {
        int mt = rng.ri(11);  // V21: Increased to 11 for swap move
        long double sc = T / T0;
        bool valid = true;

        if (mt == 0) {
            int i = rng.ri(c.n);
            long double ox = cur.x[i], oy = cur.y[i];
            cur.x[i] += rng.gaussian() * 0.5L * sc;
            cur.y[i] += rng.gaussian() * 0.5L * sc;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 1) {
            int i = rng.ri(c.n);
            long double ox = cur.x[i], oy = cur.y[i];
            long double bcx = (cur.gx0+cur.gx1)/2.0L, bcy = (cur.gy0+cur.gy1)/2.0L;
            long double dx = bcx - cur.x[i], dy = bcy - cur.y[i];
            long double d = sqrtl(dx*dx + dy*dy);
            if (d > 1e-6L) { cur.x[i] += dx/d * rng.rf() * 0.6L * sc; cur.y[i] += dy/d * rng.rf() * 0.6L * sc; }
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 2) {
            int i = rng.ri(c.n);
            long double oa = cur.a[i];
            cur.a[i] += rng.gaussian() * 80.0L * sc;
            while (cur.a[i] < 0) cur.a[i] += 360.0L;
            while (cur.a[i] >= 360.0L) cur.a[i] -= 360.0L;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        else if (mt == 3) {
            int i = rng.ri(c.n);
            long double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
            cur.x[i] += rng.rf2() * 0.5L * sc;
            cur.y[i] += rng.rf2() * 0.5L * sc;
            cur.a[i] += rng.rf2() * 60.0L * sc;
            while (cur.a[i] < 0) cur.a[i] += 360.0L;
            while (cur.a[i] >= 360.0L) cur.a[i] -= 360.0L;
            cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
        }
        else if (mt == 4) {
            vector<int> boundary; cur.getBoundary(boundary);
            if (!boundary.empty()) {
                int i = boundary[rng.ri(boundary.size())];
                long double ox=cur.x[i], oy=cur.y[i], oa=cur.a[i];
                long double bcx = (cur.gx0+cur.gx1)/2.0L, bcy = (cur.gy0+cur.gy1)/2.0L;
                long double dx = bcx - cur.x[i], dy = bcy - cur.y[i];
                long double d = sqrtl(dx*dx + dy*dy);
                if (d > 1e-6L) { cur.x[i] += dx/d * rng.rf() * 0.7L * sc; cur.y[i] += dy/d * rng.rf() * 0.7L * sc; }
                cur.a[i] += rng.rf2() * 50.0L * sc;
                while (cur.a[i] < 0) cur.a[i] += 360.0L;
                while (cur.a[i] >= 360.0L) cur.a[i] -= 360.0L;
                cur.upd(i);
                if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.a[i]=oa; cur.upd(i); valid=false; }
            } else valid = false;
        }
        else if (mt == 5) {
            long double factor = 1.0L - rng.rf() * 0.004L * sc;
            long double cx = (cur.gx0 + cur.gx1) / 2.0L, cy = (cur.gy0 + cur.gy1) / 2.0L;
            Cfg trial = cur;
            for (int i = 0; i < c.n; i++) { trial.x[i] = cx + (cur.x[i] - cx) * factor; trial.y[i] = cy + (cur.y[i] - cy) * factor; }
            trial.updAll();
            if (!trial.anyOvl()) cur = trial; else valid = false;
        }
        else if (mt == 6) {
            int i = rng.ri(c.n);
            long double ox=cur.x[i], oy=cur.y[i];
            long double levy = powl(rng.rf() + 0.001L, -1.3L) * 0.008L;
            cur.x[i] += rng.rf2() * levy; cur.y[i] += rng.rf2() * levy; cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }
        else if (mt == 7 && c.n > 1) {
            int i = rng.ri(c.n), j = (i + 1) % c.n;
            long double oxi=cur.x[i], oyi=cur.y[i], oxj=cur.x[j], oyj=cur.y[j];
            long double dx = rng.rf2() * 0.3L * sc, dy = rng.rf2() * 0.3L * sc;
            cur.x[i]+=dx; cur.y[i]+=dy; cur.x[j]+=dx; cur.y[j]+=dy;
            cur.upd(i); cur.upd(j);
            if (cur.hasOvl(i) || cur.hasOvl(j)) { cur.x[i]=oxi; cur.y[i]=oyi; cur.x[j]=oxj; cur.y[j]=oyj; cur.upd(i); cur.upd(j); valid=false; }
        }
        // V21 NEW: Swap move
        else if (mt == 10 && c.n > 1) {
            int i = rng.ri(c.n), j = rng.ri(c.n);
            Cfg old = cur;
            if (!swapTrees(cur, i, j)) {
                cur = old;
                valid = false;
            }
        }
        else {
            int i = rng.ri(c.n);
            long double ox=cur.x[i], oy=cur.y[i];
            cur.x[i] += rng.rf2() * 0.002L; cur.y[i] += rng.rf2() * 0.002L; cur.upd(i);
            if (cur.hasOvl(i)) { cur.x[i]=ox; cur.y[i]=oy; cur.upd(i); valid=false; }
        }

        if (!valid) { noImp++; T *= alpha; if (T < Tm) T = Tm; continue; }

        cur.updGlobal();
        long double ns = cur.side();
        long double delta = ns - cs;

        if (delta < 0 || rng.rf() < expl(-delta / T)) {
            cs = ns;
            if (ns < bs) { bs = ns; best = cur; noImp = 0; }
            else noImp++;
        } else { cur = best; cs = bs; noImp++; }

        if (noImp > 200) { T = min(T * 5.0L, T0); noImp = 0; }
        T *= alpha;
        if (T < Tm) T = Tm;
    }
    return best;
}

// Perturb
Cfg perturb(Cfg c, long double str, FastRNG& rng) {
    Cfg original = c;
    int np = max(1, (int)(c.n * 0.08L + str * 3.0L));
    for (int k = 0; k < np; k++) {
        int i = rng.ri(c.n);
        c.x[i] += rng.gaussian() * str * 0.5L;
        c.y[i] += rng.gaussian() * str * 0.5L;
        c.a[i] += rng.gaussian() * 30.0L;
        while (c.a[i] < 0) c.a[i] += 360.0L;
        while (c.a[i] >= 360.0L) c.a[i] -= 360.0L;
    }
    c.updAll();
    for (int iter = 0; iter < 150; iter++) {
        bool fixed = true;
        for (int i = 0; i < c.n; i++) {
            if (c.hasOvl(i)) {
                fixed = false;
                long double cx = (c.gx0+c.gx1)/2.0L, cy = (c.gy0+c.gy1)/2.0L;
                long double dx = c.x[i] - cx, dy = c.y[i] - cy;
                long double d = sqrtl(dx*dx + dy*dy);
                if (d > 1e-6L) { c.x[i] += dx/d*0.02L; c.y[i] += dy/d*0.02L; }
                c.a[i] += rng.rf2() * 15.0L;
                while (c.a[i] < 0) c.a[i] += 360.0L;
                while (c.a[i] >= 360.0L) c.a[i] -= 360.0L;
                c.upd(i);
            }
        }
        if (fixed) break;
    }
    c.updGlobal();
    if (c.anyOvl()) return original;
    return c;
}

// PARALLEL optimization
Cfg optimizeParallel(Cfg c, int iters, int restarts) {
    Cfg globalBest = c;
    long double globalBestSide = c.side();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        FastRNG rng(42 + tid * 1000 + c.n);
        Cfg localBest = c;
        long double localBestSide = c.side();

        #pragma omp for schedule(dynamic)
        for (int r = 0; r < restarts; r++) {
            Cfg start;
            if (r == 0) {
                start = c;
            }
            // V21: Every 4th restart, try rotating all trees by a fixed angle
            else if (r % 4 == 0 && r < restarts / 2) {
                start = c;
                long double angleOffset = (r / 4) * 45.0L;  // Try 0, 45, 90, 135, etc.
                for (int i = 0; i < start.n; i++) {
                    start.a[i] += angleOffset;
                    while (start.a[i] >= 360.0L) start.a[i] -= 360.0L;
                }
                start.updAll();
                if (start.anyOvl()) {
                    start = perturb(c, 0.02L + 0.02L * (r % 8), rng);
                    if (start.anyOvl()) continue;
                }
            }
            else {
                start = perturb(c, 0.02L + 0.02L * (r % 8), rng);
                if (start.anyOvl()) continue;
            }

            uint64_t seed = 42 + r * 1000 + tid * 100000 + c.n;
            Cfg o = sa_opt(start, iters, 3.0L, 0.0000005L, seed);  // V21: Increased T0 from 2.5 to 3.0
            o = squeeze(o);
            o = compaction(o, 50);
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
    globalBest = localSearch(globalBest, 150);

    if (globalBest.anyOvl()) return c;
    return globalBest;
}

map<int, Cfg> loadCSV(const string& fn) {
    map<int, Cfg> cfg;
    ifstream f(fn);
    if (!f) return cfg;
    string ln; getline(f, ln);
    map<int, vector<tuple<int,long double,long double,long double>>> data;
    while (getline(f, ln)) {
        size_t p1=ln.find(','), p2=ln.find(',',p1+1), p3=ln.find(',',p2+1);
        string id=ln.substr(0,p1), xs=ln.substr(p1+1,p2-p1-1), ys=ln.substr(p2+1,p3-p2-1), ds=ln.substr(p3+1);
        if(!xs.empty() && xs[0]=='s') xs=xs.substr(1);
        if(!ys.empty() && ys[0]=='s') ys=ys.substr(1);
        if(!ds.empty() && ds[0]=='s') ds=ds.substr(1);
        int n=stoi(id.substr(0,3)), idx=stoi(id.substr(4));
        data[n].push_back({idx, stold(xs), stold(ys), stold(ds)});
    }
    for (auto& [n,v] : data) {
        Cfg c; c.n = n;
        for (auto& [i,x,y,d] : v) if (i < n) { c.x[i]=x; c.y[i]=y; c.a[i]=d; }
        c.updAll();
        cfg[n] = c;
    }
    return cfg;
}

void saveCSV(const string& fn, const map<int, Cfg>& cfg) {
    ofstream f(fn);
    f << fixed << setprecision(17) << "id,x,y,deg\n";
    for (int n = 1; n <= 200; n++) {
        if (cfg.count(n)) {
            const Cfg& c = cfg.at(n);
            for (int i = 0; i < n; i++)
                f << setfill('0') << setw(3) << n << "_" << i << ",s" << c.x[i] << ",s" << c.y[i] << ",s" << c.a[i] << "\n";
        }
    }
}

int main(int argc, char** argv) {
    string in="submission.csv", out="submission_v21.csv";
    int iters=15000, restarts=16;

    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a=="-i" && i+1<argc) in=argv[++i];
        else if (a=="-o" && i+1<argc) out=argv[++i];
        else if (a=="-n" && i+1<argc) iters=stoi(argv[++i]);
        else if (a=="-r" && i+1<argc) restarts=stoi(argv[++i]);
    }

    int numThreads = omp_get_max_threads();
    printf("Tree Packer v21 - ENHANCED (%d threads)\n", numThreads);
    printf("NEW: Swap moves, multi-angle restarts, higher SA temperature\n");
    printf("Iterations: %d, Restarts: %d\n", iters, restarts);
    printf("Processing all n=1..200 concurrently\n");
    printf("Loading %s...\n", in.c_str());

    auto cfg = loadCSV(in);
    if (cfg.empty()) { printf("No data!\n"); return 1; }
    printf("Loaded %d configs\n", (int)cfg.size());

    long double init = 0;
    for (auto& [n,c] : cfg) init += c.score();
    printf("Initial: %.6Lf\n\nPhase 1: Parallel optimization...\n\n", init);

    auto t0 = chrono::high_resolution_clock::now();
    map<int, Cfg> res;
    int totalImproved = 0;

    // Phase 1: Main optimization - PARALLEL OVER ALL N
    vector<int> nvals;
    for (auto& [n,c] : cfg) nvals.push_back(n);

    #pragma omp parallel for schedule(dynamic)
    for (int idx = 0; idx < (int)nvals.size(); idx++) {
        int n = nvals[idx];
        Cfg c = cfg[n];
        long double os = c.score();

        int it = iters, r = restarts;
        if (n <= 10) { it = (int)(iters * 2.5); r = restarts * 2; }
        else if (n <= 30) { it = (int)(iters * 1.8); r = (int)(restarts * 1.5); }
        else if (n <= 60) { it = (int)(iters * 1.3); r = restarts; }
        else if (n > 150) { it = (int)(iters * 0.7); r = (int)(restarts * 0.8); }

        Cfg o = optimizeParallel(c, it, max(4, r));

        // Smart overlap handling: prefer non-overlapping configs
        bool o_ovl = o.anyOvl();
        bool c_ovl = c.anyOvl();

        if (!c_ovl && o_ovl) {
            // Original is valid but optimized has overlap, use original
            o = c;
        } else if (c_ovl && !o_ovl) {
            // Original has overlap but optimized doesn't, use optimized even if worse
            // Keep o (no change needed)
        } else if (!c_ovl && !o_ovl && o.side() > c.side() + 1e-14L) {
            // Both valid, but optimized is worse, use original
            o = c;
        } else if (c_ovl && o_ovl) {
            // Both have overlap, use the one with smaller side
            if (o.side() > c.side() + 1e-14L) {
                o = c;
            }
        }

        long double ns = o.score();

        #pragma omp critical
        {
            res[n] = o;
            if (c_ovl && !o_ovl) {
                printf("n=%3d: %.6Lf -> %.6Lf (FIXED OVERLAP, %.4Lf%%)\n", n, os, ns, (os-ns)/os*100.0L);
                fflush(stdout);
                totalImproved++;
            } else if (o_ovl) {
                printf("n=%3d: WARNING - still has overlap! (score %.6Lf)\n", n, ns);
                fflush(stdout);
            } else if (ns < os - 1e-10L) {
                printf("n=%3d: %.6Lf -> %.6Lf (%.4Lf%%)\n", n, os, ns, (os-ns)/os*100.0L);
                fflush(stdout);
                totalImproved++;
            }
        }
    }

    // Phase 2: AGGRESSIVE BACK PROPAGATION
    // If side(k) < side(k-1), try removing trees from k-config to improve (k-1)
    printf("\nPhase 2: Aggressive back propagation (removing trees)...\n\n");

    int backPropImproved = 0;
    bool changed = true;
    int passNum = 0;

    while (changed && passNum < 10) {
        changed = false;
        passNum++;

        for (int k = 200; k >= 2; k--) {
            if (!res.count(k) || !res.count(k-1)) continue;

            long double sideK = res[k].side();
            long double sideK1 = res[k-1].side();

            // If k trees fit in smaller box than (k-1) trees
            if (sideK < sideK1 - 1e-12L) {
                // Try removing each tree from k-config
                Cfg& cfgK = res[k];
                long double bestSide = sideK1;
                Cfg bestCfg = res[k-1];

                #pragma omp parallel
                {
                    long double localBestSide = bestSide;
                    Cfg localBestCfg = bestCfg;

                    #pragma omp for schedule(dynamic)
                    for (int removeIdx = 0; removeIdx < k; removeIdx++) {
                        Cfg reduced = cfgK.removeTree(removeIdx);

                        if (!reduced.anyOvl()) {
                            // Optimize the reduced config
                            reduced = squeeze(reduced);
                            reduced = compaction(reduced, 60);
                            reduced = localSearch(reduced, 100);

                            if (!reduced.anyOvl() && reduced.side() < localBestSide) {
                                localBestSide = reduced.side();
                                localBestCfg = reduced;
                            }
                        }
                    }

                    #pragma omp critical
                    {
                        if (localBestSide < bestSide) {
                            bestSide = localBestSide;
                            bestCfg = localBestCfg;
                        }
                    }
                }

                if (bestSide < sideK1 - 1e-12L && !bestCfg.anyOvl()) {
                    long double oldScore = res[k-1].score();
                    long double newScore = bestCfg.score();
                    res[k-1] = bestCfg;
                    printf("n=%3d: %.6Lf -> %.6Lf (from n=%d removal, %.4Lf%%)\n",
                           k-1, oldScore, newScore, k, (oldScore-newScore)/oldScore*100.0L);
                    fflush(stdout);
                    backPropImproved++;
                    changed = true;
                }
            }
        }

        // Also check k+2, k+3 etc for potential improvements
        for (int k = 200; k >= 3; k--) {
            for (int src = k + 1; src <= min(200, k + 5); src++) {
                if (!res.count(src) || !res.count(k)) continue;

                long double sideSrc = res[src].side();
                long double sideK = res[k].side();

                if (sideSrc < sideK - 1e-12L) {
                    // Try removing (src-k) trees from src-config
                    int toRemove = src - k;
                    Cfg cfgSrc = res[src];

                    // Generate combinations to try (sample if too many)
                    vector<vector<int>> combos;
                    if (toRemove == 1) {
                        for (int i = 0; i < src; i++) combos.push_back({i});
                    } else if (toRemove == 2 && src <= 50) {
                        for (int i = 0; i < src; i++)
                            for (int j = i+1; j < src; j++)
                                combos.push_back({i, j});
                    } else {
                        // Random sampling
                        FastRNG rng(k * 1000 + src);
                        for (int t = 0; t < min(200, src * 3); t++) {
                            vector<int> combo;
                            set<int> used;
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

                    long double bestSide = sideK;
                    Cfg bestCfg = res[k];

                    #pragma omp parallel
                    {
                        long double localBestSide = bestSide;
                        Cfg localBestCfg = bestCfg;

                        #pragma omp for schedule(dynamic)
                        for (int ci = 0; ci < (int)combos.size(); ci++) {
                            Cfg reduced = cfgSrc;

                            // Remove trees in reverse order to maintain indices
                            vector<int> toRem = combos[ci];
                            sort(toRem.rbegin(), toRem.rend());
                            for (int idx : toRem) {
                                reduced = reduced.removeTree(idx);
                            }

                            if (!reduced.anyOvl()) {
                                reduced = squeeze(reduced);
                                reduced = compaction(reduced, 50);
                                reduced = localSearch(reduced, 80);

                                if (!reduced.anyOvl() && reduced.side() < localBestSide) {
                                    localBestSide = reduced.side();
                                    localBestCfg = reduced;
                                }
                            }
                        }

                        #pragma omp critical
                        {
                            if (localBestSide < bestSide) {
                                bestSide = localBestSide;
                                bestCfg = localBestCfg;
                            }
                        }
                    }

                    if (bestSide < sideK - 1e-12L && !bestCfg.anyOvl()) {
                        long double oldScore = res[k].score();
                        long double newScore = bestCfg.score();
                        res[k] = bestCfg;
                        printf("n=%3d: %.6Lf -> %.6Lf (from n=%d removal, %.4Lf%%)\n",
                               k, oldScore, newScore, src, (oldScore-newScore)/oldScore*100.0L);
                        fflush(stdout);
                        backPropImproved++;
                        changed = true;
                    }
                }
            }
        }

        if (changed) printf("Pass %d complete, continuing...\n", passNum);
    }

    auto t1 = chrono::high_resolution_clock::now();
    long double el = chrono::duration_cast<chrono::milliseconds>(t1-t0).count() / 1000.0L;

    long double fin = 0;
    for (auto& [n,c] : res) fin += c.score();

    printf("\n========================================\n");
    printf("Initial: %.6Lf\nFinal:   %.6Lf\n", init, fin);
    printf("Improve: %.6Lf (%.4Lf%%)\n", init-fin, (init-fin)/init*100.0L);
    printf("Phase 1 improved: %d configs\n", totalImproved);
    printf("Phase 2 back-prop improved: %d configs\n", backPropImproved);
    printf("Time:    %.1Lfs (with %d threads)\n", el, numThreads);
    printf("========================================\n");

    saveCSV(out, res);
    printf("Saved %s\n", out.c_str());
    return 0;
}
```

```python
%%writefile bp.cpp
// Backward Propagation Optimizer
// Based on: https://www.kaggle.com/code/guntasdhanjal/santa-2025-simple-optimization-v2
//
// Key idea: If removing one tree from N-tree config gives better (N-1)-tree config, propagate it backward
// Compile: g++ -O3 -std=c++17 -o bp bp.cpp

#include <bits/stdc++.h>
using namespace std;

constexpr int MAX_N = 200;
constexpr int NV = 15;
constexpr double PI = 3.14159265358979323846;

alignas(64) const long double TX[NV] = {0,0.125,0.0625,0.2,0.1,0.35,0.075,0.075,-0.075,-0.075,-0.35,-0.1,-0.2,-0.0625,-0.125};
alignas(64) const long double TY[NV] = {0.8,0.5,0.5,0.25,0.25,0,0,-0.2,-0.2,0,0,0.25,0.25,0.5,0.5};

struct Poly {
    long double px[NV], py[NV];
    long double x0, y0, x1, y1;
};

inline void getPoly(long double cx, long double cy, long double deg, Poly& q) {
    long double rad = deg * (PI / 180.0L);
    long double s = sinl(rad), c = cosl(rad);
    long double minx = 1e9L, miny = 1e9L, maxx = -1e9L, maxy = -1e9L;
    for (int i = 0; i < NV; i++) {
        long double x = TX[i] * c - TY[i] * s + cx;
        long double y = TX[i] * s + TY[i] * c + cy;
        q.px[i] = x; q.py[i] = y;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }
    q.x0 = minx; q.y0 = miny; q.x1 = maxx; q.y1 = maxy;
}

struct Cfg {
    int n;
    long double x[MAX_N], y[MAX_N], a[MAX_N];
    Poly pl[MAX_N];
    long double gx0, gy0, gx1, gy1;

    inline void upd(int i) { getPoly(x[i], y[i], a[i], pl[i]); }

    inline void calc_bounds() {
        gx0 = gy0 = 1e9L;
        gx1 = gy1 = -1e9L;
        for (int i = 0; i < n; i++) {
            if (pl[i].x0 < gx0) gx0 = pl[i].x0;
            if (pl[i].y0 < gy0) gy0 = pl[i].y0;
            if (pl[i].x1 > gx1) gx1 = pl[i].x1;
            if (pl[i].y1 > gy1) gy1 = pl[i].y1;
        }
    }

    inline long double side() const {
        return max(gx1 - gx0, gy1 - gy0);
    }

    void remove_tree(int idx) {
        // Remove tree at index idx by shifting remaining trees
        for (int i = idx; i < n - 1; i++) {
            x[i] = x[i + 1];
            y[i] = y[i + 1];
            a[i] = a[i + 1];
            pl[i] = pl[i + 1];
        }
        n--;
    }

    void rebuild_polys() {
        for (int i = 0; i < n; i++) {
            upd(i);
        }
        calc_bounds();
    }
};

// Global storage for all configurations
Cfg configs[MAX_N + 1];  // configs[n] stores the best n-tree configuration
long double best_sides[MAX_N + 1];

void parse_csv(const string& filename) {
    ifstream f(filename);
    string line;
    getline(f, line); // Skip header

    map<int, vector<tuple<long double, long double, long double>>> data;

    while (getline(f, line)) {
        stringstream ss(line);
        string id_str, x_str, y_str, deg_str;

        getline(ss, id_str, ',');
        getline(ss, x_str, ',');
        getline(ss, y_str, ',');
        getline(ss, deg_str);

        // Parse id like "010_0"
        int n = stoi(id_str.substr(0, 3));

        // Remove 's' prefix
        long double x = stold(x_str.substr(1));
        long double y = stold(y_str.substr(1));
        long double deg = stold(deg_str.substr(1));

        data[n].push_back({x, y, deg});
    }

    // Populate configs
    for (auto& [n, trees] : data) {
        configs[n].n = n;
        for (int i = 0; i < n; i++) {
            auto [x, y, a] = trees[i];
            configs[n].x[i] = x;
            configs[n].y[i] = y;
            configs[n].a[i] = a;
        }
        configs[n].rebuild_polys();
        best_sides[n] = configs[n].side();
    }
}

void save_csv(const string& filename) {
    ofstream f(filename);
    f << "id,x,y,deg\n";
    f << fixed << setprecision(17);

    for (int n = 1; n <= MAX_N; n++) {
        for (int i = 0; i < configs[n].n; i++) {
            f << setw(3) << setfill('0') << n << "_" << i << ",";
            f << "s" << configs[n].x[i] << ",";
            f << "s" << configs[n].y[i] << ",";
            f << "s" << configs[n].a[i] << "\n";
        }
    }
}

long double calc_total_score() {
    long double score = 0.0L;
    for (int n = 1; n <= MAX_N; n++) {
        long double side = configs[n].side();
        score += (side * side) / n;
    }
    return score;
}

vector<int> get_bbox_touching_tree_indices(const Cfg& cfg) {
    vector<int> touching_indices;
    const long double eps = 1e-9L;
    
    for (int i = 0; i < cfg.n; i++) {
        const Poly& p = cfg.pl[i];
        bool touches = false;
        
        // Check if tree touches left boundary: tree's left edge aligns with global left
        if (abs(p.x0 - cfg.gx0) < eps && p.y1 >= cfg.gy0 - eps && p.y0 <= cfg.gy1 + eps) {
            touches = true;
        }
        // Check if tree touches right boundary: tree's right edge aligns with global right
        if (abs(p.x1 - cfg.gx1) < eps && p.y1 >= cfg.gy0 - eps && p.y0 <= cfg.gy1 + eps) {
            touches = true;
        }
        // Check if tree touches bottom boundary: tree's bottom edge aligns with global bottom
        if (abs(p.y0 - cfg.gy0) < eps && p.x1 >= cfg.gx0 - eps && p.x0 <= cfg.gx1 + eps) {
            touches = true;
        }
        // Check if tree touches top boundary: tree's top edge aligns with global top
        if (abs(p.y1 - cfg.gy1) < eps && p.x1 >= cfg.gx0 - eps && p.x0 <= cfg.gx1 + eps) {
            touches = true;
        }
        
        if (touches) {
            touching_indices.push_back(i);
        }
    }
    
    // If no trees touch boundary (shouldn't happen), return all indices as fallback
    if (touching_indices.empty()) {
        for (int i = 0; i < cfg.n; i++) {
            touching_indices.push_back(i);
        }
    }
    
    return touching_indices;
}

void backward_propagation() {
    cout << "Starting Backward Propagation...\n";
    cout << fixed << setprecision(8) << "Initial score: " << calc_total_score() << "\n\n";

    int total_improvements = 0;

    // Go from N=200 down to N=2
    for (int n = MAX_N; n >= 2; n--) {

        // Start with a working copy of the n-tree configuration
        Cfg candidate = configs[n];

        // Keep removing trees until we can't improve anymore
        while (candidate.n > 1) {
            int target_size = candidate.n - 1;
            long double best_current_side = best_sides[target_size];
            long double best_new_side = 1e9L;
            int best_tree_to_delete = -1;

            // Get trees that touch the bounding box boundary
            vector<int> touching_indices = get_bbox_touching_tree_indices(candidate);

            // Try deleting each boundary-touching tree
            for (int tree_idx : touching_indices) {
                // Create a test copy
                Cfg test_candidate = candidate;
                test_candidate.remove_tree(tree_idx);
                test_candidate.calc_bounds();

                long double test_side = test_candidate.side();

                // Track the best deletion
                if (test_side < best_new_side) {
                    best_new_side = test_side;
                    best_tree_to_delete = tree_idx;
                }
            }

            // If we found a deletion candidate, always remove it and continue
            if (best_tree_to_delete != -1) {
                // Remove the best tree
                candidate.remove_tree(best_tree_to_delete);
                candidate.calc_bounds();

                // If this improves the target_size configuration, save it
                if (best_new_side < best_current_side) {
                    cout << "improved " << candidate.n << " from n=" << n << " " << best_current_side << " -> " << best_new_side << "\n";
                    configs[target_size] = candidate;
                    best_sides[target_size] = best_new_side;
                    total_improvements++;
                }
                // Continue the loop even if not better than stored - keep optimizing
            } else {
                // Can't find any valid deletion, stop for this configuration
                break;
            }
        }
    }

    long double final_score = calc_total_score();
    cout << "\n\nBackward Propagation Complete!\n";
    cout << "Total improvements: " << total_improvements << "\n";
    cout << fixed << setprecision(12) << "Final score: " << final_score << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: ./bp input.csv output.csv\n";
        return 1;
    }

    string input_file = argv[1];
    string output_file = argv[2];

    cout << "Backward Propagation Optimizer\n";
    cout << "===============================\n";
    cout << "Loading " << input_file << "...\n";

    parse_csv(input_file);

    cout << "Loaded " << MAX_N << " configurations\n";

    backward_propagation();

    cout << "Saving to " << output_file << "...\n";
    save_csv(output_file);

    cout << "Done!\n";

    return 0;
}
```

```python
!wget -q https://raw.githubusercontent.com/SmartManoj/Santa-Scoreboard/main/submission.csv?cb=$(date +%s) -O submission.csv
```

```python
!head submission.csv
```

```python
!g++ -fopenmp -O3 -march=native -std=c++17 -o tree_packer_v21.exe a.cpp
!g++ -O3 -std=c++17 -o bp.exe bp.cpp
```

```python
small_iter = 1
loop = 0
```

```python
import subprocess
import shutil
import hashlib
import os
import sys

from datetime import datetime, timedelta

version = 'v21'
if small_iter:
    os.environ['OMP_NUM_THREADS'] = '96'
    n= 5000
    r = 16
    fact = 1
    cmd = f'./tree_packer_{version}.exe  -n {n*fact} -r {r*fact}'
else:
    os.environ['OMP_NUM_THREADS'] = '96'
    n= 10000
    r = 256
    fact = 2
    cmd = f'./tree_packer_{version}.exe  -n {n*fact} -r {r*fact}'


def file_hash(filepath):
    """Calculate MD5 hash of a file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def run_tree_packer():
    """Run tree_packer with live output and return final score."""
    print(f"Running tree_packer_{version}.exe...")
    sys.stdout.flush()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True,
        universal_newlines=True
    )
    
    final_score = None
    # Print output line by line in real-time and parse score
    for line in process.stdout:
        print(line, end='', flush=True)
        # Parse "Final:   X.XXXXXX" line
        if line.startswith('Final:'):
            try:
                final_score = float(line.split()[1])
            except (ValueError, IndexError):
                pass
    
    process.wait()
    return process.returncode == 0, final_score

def move_file(iteration):
    """Move submission file to submission.csv."""
    src = f'submission_{version}.csv'
    dst = r'submission.csv'
    dst2 = f'submission_{version}_{iteration}.csv'
    if os.path.exists(src):
        shutil.copy(src, dst)
        shutil.copy(src, dst2)
        return True
    return False


def main():
    iteration = 1
    previous_score = None
    start_time = datetime.now()
    while True:
        elapsed_time = datetime.now() - start_time
        print(f"Time elapsed: {elapsed_time}")
        if elapsed_time > timedelta(hours=11):
            print("Time limit exceeded!")
            break
        print(f"\n{'='*50}")
        print(f"Iteration {iteration}")
        print(f"{'='*50}\n")
        
        # Check hash of current submission.csv before processing
        initial_hash = file_hash('submission.csv')
        
        # Run tree_packer
        success, final_score = run_tree_packer()
        if not success:
            print(f"tree_packer_{version}.exe failed!")
            break
        
        # Check if submission file exists
        submission_file = f'submission_{version}.csv'
        if not os.path.exists(submission_file):
            print(f"{submission_file} not created!")
            break
        
        # Check if score changed
        if previous_score is not None and final_score is not None:
            if abs(final_score - previous_score) < 1e-9:
                print("\nNo score change detected. Convergence achieved!")
                break
        
        # Check if files are different
        submission_file = f'submission_{version}.csv'
        new_hash = file_hash(submission_file)
        if initial_hash == new_hash:
            print("\nNo changes detected. Convergence achieved!")
            break
        
        # Update previous score
        if final_score is not None:
            previous_score = final_score
        
        # Move file
        move_file(iteration)
        print(f"Files differ. Moved {submission_file} -> submission.csv")
        !./bp.exe submission.csv submission.csv
        if not loop:
            break
        iteration += 1
    
    print(f"\nCompleted after {iteration} iteration(s)")

if __name__ == '__main__':
    main()
```

```python
!python -m pip install  /kaggle/input/browser-notifications-in-a-kaggle-kernel/*.whl 
!python -m pip install jupyter -q --no-index --find-links=/kaggle/input/browser-notifications-in-a-kaggle-kernel/ 
!python -m pip install -q /kaggle/input/browser-notifications-in-a-kaggle-kernel/jupyter-notify/dist/jupyternotify-0.1.15-py2.py3-none-any.whl --no-index
```

```python
%reload_ext jupyternotify
```

```python
%notify
```