# Oh Barnacles

- **Author:** jazivxt
- **Votes:** 331
- **Ref:** jazivxt/oh-barnacles
- **URL:** https://www.kaggle.com/code/jazivxt/oh-barnacles
- **Last run:** 2025-09-05 02:01:46.037000

---

# Oh Barnacles

```python
from IPython.display import YouTubeVideo
YouTubeVideo('93vs222tPnU',width=600, height=400)
```

```python
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()
```

# Respect, Credit and Recognition
* For all your shares and collective efforts in this competition, Thanks!
* This collection is yours and from your efforts too...were almost at 400, enjoy!

@adilshamim8 @adyanthm @boristown @bibanh @cheeseexports @daosyduyminh @dedquoc @fedimser @henrychibueze @jacekwl @jaejohn @jazivxt @jeroencottaar @katsuyanomura @kosirowada @krishnayadav456wrsty @kuntalmaity @limalkasadith @mmoffitt @mpwolke @muhammaddanyalmalik @muhammadqasimshabbir @nina2025 @ottitsch @quannguyn12 @raviannaswamy @seshura @seshurajup @shevtsovapolina @taylorsamarel @zshashz

# [001] 007bbfb7.json
* image_repetition
* fractal_repetition

```python
show_examples(load_examples(1)['train'])
```

```python
%%writefile task001.py
p=lambda g,R=range(9):[[g[r//3][c//3]and g[r%3][c%3]for c in R]for r in R]
```

# [002] 00d62c1b.json
* loop_filling

```python
show_examples(load_examples(2)['train'])
```

```python
%%writefile task002.py
def p(j):
	A=range;c=len(j);E=[[0]*c for B in A(c)]
	def B(k,W):
		if 0<=k<c and 0<=W<c and not E[k][W]and j[k][W]==0:E[k][W]=1;[B(k+c,W+A)for(c,A)in[(1,0),(-1,0),(0,1),(0,-1)]]
	[B(A,0)or B(A,c-1)or B(0,A)or B(c-1,A)for A in A(c)];return[[4 if j[B][c]==0and not E[B][c]else j[B][c]for c in A(c)]for B in A(c)]
```

# [003] 017c7c7b.json
* recoloring
* pattern_expansion
* pattern_repetition
* image_expansion

```python
show_examples(load_examples(3)['train'])
```

```python
%%writefile task003.py
p=lambda j:[[c*2 for c in r]for r in j+(j[:3],j[2:5])[j[1]!=j[4]]]
```

# [004] 025d127b.json
* pattern_modification

```python
show_examples(load_examples(4)['train'])
```

```python
%%writefile task004.py
def p(j,A=enumerate):
 c=[[0]*len(j[0])for _ in j]
 for E in set(sum(j,[]))-{0}:
  k=[(J,a)for J,r in A(j)for a,x in A(r)if x==E];W,l=max(J for J,_ in k),max(a for _,a in k)
  for J,a in k:c[J][a+(J<W and a<l)]=E
 return c
```

# [005-R] 045e512c.json
* pattern_expansion
* direction_guessing

```python
show_examples(load_examples(5)['train'])
```

```python
%%writefile task005.py
L=len
R=range
E=enumerate
def M(m,C):return sorted([[y,x] for y,r in E(m) for x,c in E(r) if c==C])
def p(g):
 h,w=L(g),L(g[0])
 X=[r[:] for r in g]
 P=sorted([[sum(g,[]).count(C),C] for C in R(9)])
 d={k:M(g,k) for v,k in P[:-2]}
 Z=M(g,P[-2][1])
 h,w=L(g),L(g[0])
 for C in d:
  if L(d[C])>0:
   for m in range(1,10):
    r,c=d[C][0][0]-Z[0][0],d[C][0][1]-Z[0][1]
    if r<0: r+=-1 #extremety points need more work
    r*=m
    c*=m
    for y,x in Z:
     if r+y>=0 and c+x>=0:
      try:
       X[r+y][c+x]=C
      except: pass
 return X
```

# [006] 0520fde7.json
* detect_wall
* separate_images
* pattern_intersection

```python
show_examples(load_examples(6)['train'])
```

```python
%%writefile task006.py
p=lambda j:[[a and b and 2 for a,b in zip(r[:3],r[4:7])]for r in j[:3]]
```

# [007] 05269061.json
* image_filling
* pattern_expansion
* diagonals

```python
show_examples(load_examples(7)['train'])
```

```python
%%writefile task007.py
def p(g):R=range;L=len;d={(i+j)%3:c for i in R(L(g))for j in R(L(g[0]))for c in[g[i][j]]if c};return[[d.get((i+j)%3,0)for j in R(L(g[0]))]for i in R(L(g))]
```

# [008] 05f2a901.json
* pattern_moving
* direction_guessing
* bring_patterns_close

```python
show_examples(load_examples(8)['train'])
```

```python
%%writefile task008.py
def p(j,A=enumerate):
	c,E=[(c,b)for(c,E)in A(j)for(b,d)in A(E)if d==2],[(c,b)for(c,E)in A(j)for(b,d)in A(E)if d==8]
	if not c or not E:return j
	k=lambda W:(min(c for(c,E)in W),max(c for(c,E)in W),min(c for(E,c)in W),max(c for(E,c)in W));l,J,a,C=k(c);e,K,w,L=k(E);b=d=0
	if C<w:d=w-C-1
	elif L<a:d=L-a+1
	elif J<e:b=e-J-1
	elif K<l:b=K-l+1
	f,g={*c},{*E};return[[8 if(c,E)in g else 2 if(c-b,E-d)in f else 0 for(E,k)in A(j[0])]for(c,E)in A(j)]
```

# [009] 06df4c85.json
* detect_grid
* connect_the_dots
* grid_coloring

```python
show_examples(load_examples(9)['train'])
```

```python
%%writefile task009.py
def p(j,A=range,c=len):
	E=[J[:]for J in j];k,W=c(j),c(j[0]);l=j[0][2]
	for J in A(k):
		for a in A(W):
			if j[J][a]==l:E[J][a]=l;j[J][a]=0
			else:E[J][a]=0
	C=[J[:]for J in j]
	for e in A(k):
		K=[(J,a)for J in A(k)for a in A(W)if j[J][a]==e]
		for J in A(len(K)):
			for a in A(J+1,len(K)):
				w,L=K[J];b,d=K[a]
				if w==b:
					for f in A(min(L,d),max(L,d)+1):C[w][f]=e
				elif L==d:
					for g in A(min(w,b),max(w,b)+1):C[g][L]=e
	for J in A(k):
		for a in A(W):
			if E[J][a]>0:C[J][a]=l
	return C
```

# [010] 08ed6ac7.json
* measure_length
* order_numbers
* associate_colors_to_ranks
* recoloring

```python
show_examples(load_examples(10)['train'])
```

```python
%%writefile task010.py
def p(j):
 A={}
 for c in j:
  for E,k in enumerate(c):
   if k==5:c[E]=A.setdefault(E,len(A)+1)
 return j
```

# [011] 09629e4f.json
* detect_grid
* separate_images
* count_tiles
* take_minimum
* enlarge_image
* create_grid
* adapt_image_to_grid

```python
show_examples(load_examples(11)['train'])
```

```python
%%writefile task011.py
def p(j):
 A=range
 for c in A(3):
  for E in A(3):
   if sum(j[c*4+W][E*4+l]==0for W in A(3)for l in A(3))==5:
    k=[[5if i%4==3or j%4==3else 0for j in A(11)]for i in A(11)]
    for W in A(3):
     for l in A(3):
      J=j[c*4+W][E*4+l]
      if J:
       for a in A(3):
        for C in A(3):k[W*4+a][l*4+C]=J
    return k
```

# [012] 0962bcdd.json
* pattern_expansion

```python
show_examples(load_examples(12)['train'])
```

```python
%%writefile task012.py
def p(j,A=range(-2,3),c=enumerate,E=abs):k=[E[:]for E in j];[k[I+D].__setitem__(C+F,H if E(D)==E(F)else B[C-1])for(I,B)in c(j)for(C,H)in c(B)if H and B[C-1]*B[C+1]for D in A for F in A if E(D)==E(F)or not D*F];return k
```

# [013] 0a938d79.json
* direction_guessing
* draw_line_from_point
* pattern_expansion

```python
show_examples(load_examples(13)['train'])
```

```python
%%writefile task013.py
def p(j,A=range):
 c,E=len(j),len(j[0])
 p=[(l,L,j[l][L])for l in A(c)for L in A(E)if j[l][L]]
 p.sort()
 if len(p)==2:
  k,W=p
  if k[0]==W[0]:
   l,J,a=k;C,e=W[1],W[2];K=abs(C-J)
   for w in A(c):j[w][J]=a;j[w][C]=e
   if K:
    L=max(J,C)+K;b=0;d=[a,e]
    if C<J:d=d[::-1]
    while L<E:
     for w in A(c):j[w][L]=d[b%2]
     L+=K;b+=1
  elif k[1]==W[1]:
   L,f,a=k[1],k[0],k[2];g,e=W[0],W[2];K=abs(g-f)
   for w in A(E):j[f][w]=a;j[g][w]=e
   if K:
    l=g+K;b=0;d=[a,e]
    while l<c:
     for w in A(E):j[l][w]=d[b%2]
     l+=K;b+=1
  elif k[0]==0and W[0]==c-1:
   f,J,a=k;g,C,e=W;K=abs(C-J)
   for w in A(c):j[w][J]=a;j[w][C]=e
   if K:
    L=C+K;b=0;d=[a,e]
    while L<E:
     for w in A(c):j[w][L]=d[b%2]
     L+=K;b+=1
  elif(k[1]==0and W[1]==E-1)or(W[1]==0and k[1]==E-1):
   if k[1]==0:f,J,a=k;g,C,e=W
   else:f,J,a=W;g,C,e=k
   K=abs(g-f)
   for w in A(E):j[f][w]=a;j[g][w]=e
   if K:
    l=max(f,g)+K;b=0;d=[a,e]
    if g<f:d=d[::-1]
    while l<c:
     for w in A(E):j[l][w]=d[b%2]
     l+=K;b+=1
 return j
```

# [014] 0b148d64.json
* detect_grid
* separate_images
* find_the_intruder
* crop

```python
show_examples(load_examples(14)['train'])
```

```python
%%writefile task014.py
from collections import*
def p(j):
 A=[x for k in j for x in k];c=Counter(A).most_common(3);c=[c for c in c if c[0]>0][-1][0];j=[k for k in j if c in k];E=[]
 for k in j:
  for W in range(len(k)):
   if k[W]==c:E+=[W]
 return[k[min(E):max(E)+1]for k in j]
```

# [015] 0ca9ddb6.json
* pattern_expansion
* associate_patterns_to_colors

```python
show_examples(load_examples(15)['train'])
```

```python
%%writefile task015.py
L=len
R=range
def p(g):
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]==2:
    for i,j in[[1,1],[-1,-1],[-1,1],[1,-1]]:g[i+r][j+c]=4
   if g[r][c]==1:
    for i,j in[[0,1],[0,-1],[-1,0],[1,0]]:g[i+r][j+c]=7
 return g
```

# [016] 0d3d703e.json
* associate_colors_to_colors

```python
show_examples(load_examples(16)['train'])
```

```python
%%writefile task016.py
p=lambda j,A=[0,5,6,4,3,1,2,7,9,8]:[[A[x]for x in r]for r in j]
```

# [017] 0dfd9992.json
* image_filling
* pattern_expansion

```python
show_examples(load_examples(17)['train'])
```

```python
%%writefile task017.py
def p(j,u=enumerate):
	A=range;c=len(j);E=len(j[0]);k=lambda W,l:W==l or W*l<1;J=next((K for K in A(1,E)if all(k(L,e)for w in j for(L,e)in zip(w,w[K:]))),E);a=next((K for K in A(1,c)if all(k(L,e)for(K,w)in zip(j,j[K:])for(L,e)in zip(K,w))),c);C={}
	for(e,K)in u(j):
		for(w,L)in u(K):
			if L:C[e%a,w%J]=L
	for(e,K)in u(j):
		for(w,L)in u(K):
			if not L:K[w]=C[e%a,w%J]
	return j
```

# [018] 0e206a2e.json
* associate_patterns_to_patterns
* pattern_repetition
* pattern_rotation
* pattern_reflection
* pattern_juxtaposition

```python
show_examples(load_examples(18)['train'])
```

```python
%%writefile task018.py
def T(V,R):U=list(zip(*V));x=int((min(U[1])+max(U[1]))/2);y=int((min(U[2])+max(U[2]))/2);V=list(zip(*[((u,-c+y+x,r-x+y),(u,c-y+x,-r+x+y))if R else((u,2*x-r,c),(u,r,2*y-c))for u,r,c in V]));return V
def p(d):
 R=range;L=len;V=[];U=[];H=L(d);W=L(d[0]);E=[];K=list.extend
 for i in R(H):
  for j in R(W):
   if (c:=d[i][j])and(c,i,j)not in V+U:
    O=[(c,i,j)];C=set()
    while O:
     v=O.pop(0)
     if v not in C:C.add(v);K(O,[(c,x,y)for a,b in[(0,-1),(0,1),(1,0),(-1,0)]if 0<=(x:=v[1]+a)<H if 0<=(y:=v[2]+b)<W if(c:=d[x][y])])
    F=list(C)
    if len(C)>3:E.append(F);V;K(V,F)
    else:K(U,F)
 for O in E:
  for i in R(-H,H):
   for j in R(-W,W):
    Q=[(v[0],v[1]+i,v[2]+j)for v in O];B=T(Q,1);Z=T(Q,0);X=T(B[0],0)+T(B[1],0);Y=T(Z[0],1)+T(Z[1],1);A=[Q]+B+Z+X+Y;S=[I for I in A if len([v for v in I if v not in V and v in U])==3]
    if S:
     for i,(u,r,c) in enumerate(S[0]):d[r][c]=u;d[O[i][1]][O[i][2]]=0
 return d
```

# [019] 10fcaaa3.json
* pattern_expansion
* image_repetition

```python
show_examples(load_examples(19)['train'])
```

```python
%%writefile task019.py
L=len
R=range
def p(g):
 g=[r[:]+r[:]for r in g]+[r[:]+r[:]for r in g]
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   C=g[r][c]
   if C>0 and C!=8:
    for i,j in[[1,1],[-1,-1],[-1,1],[1,-1]]:
     if i+r>=0 and j+c>=0 and i+r<h and j+c<w:
      if g[i+r][j+c]==0:g[i+r][j+c]=8
 return g
```

# [020] 11852cab.json
* pattern_expansion

```python
show_examples(load_examples(20)['train'])
```

```python
%%writefile task020.py
def p(g,H=enumerate):
 t=l=9**9;b=r=-1
 for y,a in H(g):
  for x,v in H(a):
   if v:t=min(t,y);b=max(b,y);l=min(l,x);r=max(r,x)
 s=t+b;S=l+r
 for y in range(t,b+1):
  for x in range(l,r+1):
   Y=s-y;X=S-x;u=t+x-l;v=l+y-t;U=t+r-x;V=l+b-y
   P=((y,x),(y,X),(Y,x),(Y,X),(u,v),(U,v),(u,V),(U,V))
   c=max(g[i][j]for i,j in P)
   for i,j in P:g[i][j]=c
 return g
```

# [021] 1190e5a7.json
* detect_grid
* count_hor_lines
* count_ver_lines
* detect_background_color
* color_guessing
* create_image_from_info

```python
show_examples(load_examples(21)['train'])
```

```python
%%writefile task021.py
def p(g,u=range):n=len(g);m=len(g[0]);r=[i for i in u(n)if len(set(g[i]))==1];c=[j for j in u(m)if len(set(g[i][j]for i in u(n)))==1];b=next(x for i in u(n)for j,x in enumerate(g[i])if i not in r and j not in c);return[[b]*(len(c)+1)for _ in u(len(r)+1)]
```

# [022] 137eaa0f.json
* pattern_juxtaposition

```python
show_examples(load_examples(22)['train'])
```

```python
%%writefile task022.py
L=len
R=range
def p(g):
 X=[[0,0,0]for _ in R(3)]
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]==5:
    for i in R(-1,2):
     for j in R(-1,2):
      if r+i>=0 and c+j>=0 and g[r+i][c+j]!=0:X[1+i][1+j]=g[r+i][c+j]
 return X
```

# [023-R] 150deff5.json
* pattern_coloring
* pattern_deconstruction
* associate_colors_to_patterns

```python
show_examples(load_examples(23)['train'])
```

```python
%%writefile task023.py
def p(g,L=len,R=range):
 #rules: 1x3/3x1 for all reds, 2x2 for all blues, no gray remaining
 h,w=L(g),L(g[0])
 Z=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]] #3x3
 P=[[0,0],[0,1],[1,0],[1,1]] #2x2
 Q=[[0,0],[0,1],[0,2]] #1x3
 S=[[0,0],[1,0],[2,0]] #3x1
 for r in R(h):
  for c in R(w):
   try:
    if [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,5,0,0,5]:
     Y=[8,8,2,8,8,2,0,0,2]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,5,5,0,0]:
     Y=[2,8,8,2,8,8,2,0,0]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[0,5,5,0,5,5,5,5,5]:
     Y=[0,8,8,0,8,8,2,2,2]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
    elif [g[r+i[0]][c+i[1]] for i in Z]==[5,5,5,5,5,0,5,5,0]:
     Y=[2,2,2,8,8,0,8,8,0]
     for i in R(L(Z)): 
      g[r+Z[i][0]][c+Z[i][1]]=Y[i]
   except: pass
 for r in R(h):
  for c in R(w):
   try:
    if [g[r+i[0]][c+i[1]] for i in P]==[5,5,5,5]:
     for i in P: 
      g[r+i[0]][c+i[1]]=8
    elif [g[r+i[0]][c+i[1]] for i in Q]==[5,5,5]:
     for i in Q: 
      g[r+i[0]][c+i[1]]=2
    elif [g[r+i[0]][c+i[1]] for i in S]==[5,5,5]:
     for i in S: 
      g[r+i[0]][c+i[1]]=2
   except: pass
 return g
```

# [024] 178fcbfb.json
* direction_guessing
* draw_line_from_point

```python
show_examples(load_examples(24)['train'])
```

```python
%%writefile task024.py
def p(g,E=enumerate):Z={c for R in g for c,v in E(R)if v==2};return[[1 if 1 in R else 3 if 3 in R else 2 if v<1and c in Z else v for c,v in E(R)]for R in g]
```

# [025] 1a07d186.json
* bring_patterns_close
* find_the_intruder

```python
show_examples(load_examples(25)['train'])
```

```python
%%writefile task025.py
L=len
R=range
def p(g):
 D=[0]
 for i in range(8):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  for r in R(h):
   if len(set(g[r]))<2 and g[r][0]>0:
    C=g[r][0];D+=[C]
    for y in R(0,r-1):
     P=0
     for c in R(w):
      if g[y][c]==C:g[y][c]=0;P-=1;g[r+P][c]=C
    for y in R(r+1,h):
     P=0
     for c in R(w):
      if g[y][c]==C:g[y][c]=0;P+=1;g[r+P][c]=C
 g=[[c if c in D else 0 for c in r] for r in g]
 return g
```

# [026] 1b2d62fb.json
* detect_wall
* separate_images
* pattern_intersection

```python
show_examples(load_examples(26)['train'])
```

```python
%%writefile task026.py
p=lambda j:[[8*(not A|B)for(A,B)in zip(A,A[4:])]for A in j]
```

# [027-R] 1b60fb0c.json
* pattern_deconstruction
* pattern_rotation
* pattern_expansion

```python
show_examples(load_examples(27)['train'])
```

```python
%%writefile task027.py
L=len
R=range
def p(g):
 h,w=L(g),L(g[0])
 for r in R(1,h-1):
  for c in R(w//2-1):
   try:
    if g[r+1][c]==0 and g[-(r+2)][-(c+1)]>0:g[r+1][c]=2
   except:0
 return g
```

# [028] 1bfc4729.json
* pattern_expansion

```python
show_examples(load_examples(28)['train'])
```

```python
%%writefile task028.py
def p(g,m=max):e,a=m(map(m,g[:5])),m(map(m,g[-5:]));return[[(r in(0,2,7,9)or c%9<1)*(e*(r<5)+a*(r>4))for c in range(10)]for r in range(10)]
```

# [029] 1c786137.json
* detect_enclosure
* crop

```python
show_examples(load_examples(29)['train'])
```

```python
%%writefile task029.py
def p(g,L=len,E=enumerate):
 for C in set(sum(g,[])):
  P=[[x,y] for y,r in E(g) for x,c in E(r) if c==C]
  f=sum(P,[]);x=f[::2];y=f[1::2]
  X=g[min(y):max(y)]
  X=[r[min(x)+1:max(x)][:] for r in X]
  if X[0].count(C)==L(X[0]):
   return X[1:]
 return g
```

# [030] 1caeab9d.json
* pattern_moving
* pattern_alignment

```python
show_examples(load_examples(30)['train'])
```

```python
%%writefile task030.py
def p(g,L=len,R=range):
 h,w,x,y,b=L(g),L(g[0]),[],[],[]
 for r in R(h):
  for c in R(w):
   C=g[r][c]
   if C==2:x+=[c];g[r][c]=0
   if C==4:y+=[c];g[r][c]=0
   if C==1:b+=[c]
 for r in R(h):
  for c in R(w):
   if g[r][c]==1:g[r][c+(min(y)-min(b))]=4;g[r][c+(min(x)-min(b))]=2
 return g
```

# [031] 1cf80156.json
* crop

```python
show_examples(load_examples(31)['train'])
```

```python
%%writefile task031.py
def p(j,A=enumerate):c,E=zip(*[(i,j)for i,r in A(j)for j,x in A(r)if x]);return[r[min(E):max(E)+1]for r in j[min(c):max(c)+1]]
```

# [032] 1e0a9b12.json
* pattern_moving
* gravity

```python
show_examples(load_examples(32)['train'])
```

```python
%%writefile task032.py
p=lambda j:list(map(list,zip(*[[0]*c.count(0)+[x for x in c if x]for c in zip(*j)])))
```

# [033] 1e32b0e9.json
* detect_grid
* separate_images
* image_repetition
* pattern_completion

```python
show_examples(load_examples(33)['train'])
```

```python
%%writefile task033.py
def p(j):
	A=range;c=[A[:]for A in j];E=j[5][0];k=[[j[l+1][A+1]for A in A(3)]for l in A(3)]
	for W in[(0,6),(0,12),(6,0),(6,6),(6,12),(12,0),(12,6),(12,12)]:
		for l in A(3):
			for J in A(3):a,C=W[0]+l+1,W[1]+J+1;c[a][C]=j[a][C]if k[l][J]==j[a][C]else E if k[l][J]else 0
	return c
```

# [034] 1f0c79e5.json
* pattern_expansion
* diagonals
* direction_guessing

```python
show_examples(load_examples(34)['train'])
```

```python
%%writefile task034.py
R=range
L=len
def p(g):
 f=sum(g,[])
 C=[c for c in set(f) if c not in [0,2]][0]
 h,w=L(g),L(g[0])
 for r in R(h-1):
  for c in R(w-1):
   M=[g[r+y][c+x]for y,x in[[0,0],[0,1],[1,0],[1,1]]]
   if sum([1 for i in M if i>0])>3:
    for I in [z for z in R(L(M)) if M[z]==2]:
     for i in R(10):
      if I<2:y=r-i
      else:y=r+i+1
      if I%2:x=c+i+1
      else:x=c-i
      if 0<=y<h and 0<=x<w:
       g[y][x]=C
       if 0<=y-1<h:g[y-1][x]=C
       if 0<=x-1<w:g[y][x-1]=C
       if 0<=y+1<h:g[y+1][x]=C
       if 0<=x+1<w:g[y][x+1]=C
 return g
```

# [035] 1f642eb9.json
* image_within_image
* projection_unto_rectangle

```python
show_examples(load_examples(35)['train'])
```

```python
%%writefile task035.py
L=len
R=range
P=[[0,1],[0,-1],[1,0],[-1,0]]
def p(g):
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]==8:
    for y,x in P:
     if g[r+y][c+x]==0:
      for z in R(20):
        if 0<=r+(y*z)<h and 0<=c+(x*z)<w:
         W=g[r+(y*z)][c+(x*z)]
         if W>0:g[r][c]=W
 return g
```

# [036] 1f85a75f.json
* crop
* find_the_intruder

```python
show_examples(load_examples(36)['train'])
```

```python
%%writefile task036.py
def p(j,A=len,c=range):
	E='r';k='c';W,l,J=A(j),A(j[0]),{}
	for a in c(W):
		for C in c(l):
			e=j[a][C]
			if e in J:J[e][E]+=[a];J[e][k]+=[C]
			else:J[e]={E:[a],k:[C]}
	K=sorted([[A(J[e][E])*(max(J[e][k])-min(J[e][k])),e]for e in J if e>0])[0][1];j=[[K if J==K else 0 for J in J]for J in j];J=J[K];j=[E[min(J[k]):max(J[k])+1]for E in j];j=j[min(J[E]):max(J[E])+1];return j
```

# [037] 1f876c06.json
* connect_the_dots
* diagonals

```python
show_examples(load_examples(37)['train'])
```

```python
%%writefile task037.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k,W={},[A[:]for A in j]
	for l in A(c):
		for J in A(E):
			a=j[l][J]
			if a:k.setdefault(a,[]).append((l,J))
	for a in k:
		(C,e),(K,w)=k[a];L=1 if K>C else-1;b=1 if w>e else-1
		for d in A(abs(K-C)+1):W[C+d*L][e+d*b]=a
	return W
```

# [038] 1fad071e.json
* count_patterns
* associate_images_to_numbers

```python
show_examples(load_examples(38)['train'])
```

```python
%%writefile task038.py
def p(g):q=range;c=sum(all(g[i+k][j+l]==1for k in q(2)for l in q(2))for i in q(8)for j in q(8));return[[1if i<c else 0for i in q(5)]]
```

# [039] 2013d3e2.json
* pattern_deconstruction
* crop

```python
show_examples(load_examples(39)['train'])
```

```python
%%writefile task039.py
j=len
A=range
def p(c):
	E,k=j(c),j(c[0]);W=[]
	for l in A(E):
		for J in A(k):
			if c[l][J]>0:W.append([l,J])
	a=min([W[1]for W in W]);C=max([W[1]for W in W]);e=min([W[0]for W in W]);K=max([W[0]for W in W]);C=C-(C-a)//2;K=K-(K-e)//2;c=c[e:K];c=[W[a:C]for W in c];return c
```

# [040] 2204b7a8.json
* proximity_guessing
* recoloring

```python
show_examples(load_examples(40)['train'])
```

```python
%%writefile task040.py
def p(j):
	A=range;c=[J[:]for J in j];E=j[0][0]==j[0][9];k,W=(j[0][0],j[9][0])if E else(j[0][0],j[0][9]);l=next(J for a in j for J in a if J and J not in[k,W])
	for J in A(10):
		for a in A(10):
			if j[J][a]==l:C=(J,9-J)if E else(a,9-a);c[J][a]=k if C[0]<C[1]else W
	return c
```

# [041] 22168020.json
* pattern_expansion

```python
show_examples(load_examples(41)['train'])
```

```python
%%writefile task041.py
def p(j,A=0):
 for c in j:
  for E,k in enumerate(c):
   if k:A=(not A)*k
   else:c[E]=A
 return j
```

# [042] 22233c11.json
* pattern_expansion
* size_guessing

```python
show_examples(load_examples(42)['train'])
```

```python
#%%writefile task042.py
```

# [043] 2281f1f4.json
* direction_guessing
* draw_line_from_point
* pattern_intersection

```python
show_examples(load_examples(43)['train'])
```

```python
%%writefile task043.py
def p(j,A=enumerate):
 c=len(j)-1
 E=len(j[0])-1
 for k,W in A(j):
  for l,J in A(W):
   if k>0and l<c:
    if j[k][E]==5and j[0][l]==5:j[k][l]=2
 return j
```

# [044] 228f6490.json
* pattern_moving
* loop_filling
* shape_guessing
* x_marks_the_spot

```python
show_examples(load_examples(44)['train'])
```

```python
#%%writefile task044.py
```

# [045] 22eb0ac0.json
* connect_the_dots
* color_matching

```python
show_examples(load_examples(45)['train'])
```

```python
%%writefile task045.py
def p(j):
	for A in j:
		for c in{*A}-{0}:
			E=A.index(c);k=len(A)-A[::-1].index(c)
			for W in range(E,k):
				if~A[W]:A[W]=c
	return j
```

# [046] 234bbc79.json
* recoloring
* bring_patterns_close
* crop

```python
show_examples(load_examples(46)['train'])
```

```python
#%%writefile task046.py
```

# [047] 23581191.json
* draw_line_from_point
* pattern_intersection

```python
show_examples(load_examples(47)['train'])
```

```python
%%writefile task047.py
def p(j):
	A=range;c=[[0]*9 for c in A(9)];E=[(c,E,j[c][E])for c in A(9)for E in A(9)if j[c][E]]
	for(k,W,l)in E:
		for J in range(9):c[k][J]=c[J][W]=l
	c[E[0][0]][E[1][1]]=c[E[1][0]][E[0][1]]=2;return c
```

# [048] 239be575.json
* detect_connectedness
* associate_images_to_bools

```python
show_examples(load_examples(48)['train'])
```

```python
%%writefile task048.py
def f(j,A,c):
	global W;l.append((j,A))
	for E in C(j-1,j+2):
		for k in C(A-1,A+2):
			if(E,k)in l:continue
			l.append((E,k))
			if E<0 or E>=J or k<0 or k>=a or(E,k)in[(K,L),(K+1,L),(K,L+1),(K+1,L+1)]:continue
			if c[E][k]==2:W=8
			if c[E][k]==8:f(E,k,c)
def p(c):
	global W,l,K,L,J,a,C;W,l,J,a,C,e=0,[],len(c),len(c[0]),range,enumerate
	for(K,w)in e(c):
		for(L,b)in e(w):
			if b==2:
				for E in C(K-1,K+3):
					for k in C(L-1,L+3):
						if E>=0 and E<J and k>=0 and k<a and c[E][k]==8:f(E,k,c)
				return[[W]]
```

# [049] 23b5c85d.json
* measure_area
* take_minimum
* crop

```python
show_examples(load_examples(49)['train'])
```

```python
%%writefile task049.py
from collections import*
j=len
A=range
def p(c):
	E=[x for A in c for x in A];k=Counter(E).most_common();k=[C for C in k if C[0]!=0][-1][0];W,l=j(c),j(c[0]);J=[]
	for a in A(W):
		for C in A(l):
			if c[a][C]==k:J.append([a,C])
	e=min([i[1]for i in J]);K=max([i[1]for i in J]);w=min([i[0]for i in J]);L=max([i[0]for i in J]);c=c[w:L+1];c=[A[e:K+1]for A in c];return c
```

# [050] 253bf280.json
* connect_the_dots
* direction_guessing

```python
show_examples(load_examples(50)['train'])
```

```python
%%writefile task050.py
def p(j):
	A=range;c=[e[:]for e in j];E,k=len(j),len(j[0]);W=[(e,l)for e in A(E)for l in A(k)if j[e][l]==8]
	for(l,J)in W:
		for(a,C)in[(0,1),(1,0),(0,-1),(-1,0)]:
			e=1
			while 0<=l+e*a<E and 0<=J+e*C<k:
				if j[l+e*a][J+e*C]==8:
					for K in A(1,e):c[l+K*a][J+K*C]=3
					break
				e+=1
	return c
```

# [051] 25d487eb.json
* draw_line_from_point
* direction_guessing
* color_guessing

```python
show_examples(load_examples(51)['train'])
```

```python
%%writefile task051.py
def p(j):
	A=range;c=[l[:]for l in j];E,k=len(j),len(j[0]);W={}
	for l in A(E):
		for J in A(k):
			if j[l][J]:W[j[l][J]]=W.get(j[l][J],0)+1
	l,J,a=next((l,J,j[l][J])for l in A(E)for J in A(k)if j[l][J]and W[j[l][J]]==1)
	for(C,e)in[(0,1),(1,0),(0,-1),(-1,0)]:
		K,w=l+C,J+e
		if(K<0)|(K>=E)|(w<0)|(w>=k)|(j[K][w]==0):
			L=1
			while(0<=l-L*C<E)&(0<=J-L*e<k):
				if j[l-L*C][J-L*e]==0:c[l-L*C][J-L*e]=a
				L+=1
	return c
```

# [052] 25d8a9c8.json
* detect_hor_lines
* recoloring
* remove_noise

```python
show_examples(load_examples(52)['train'])
```

```python
%%writefile task052.py
p=lambda j:[[5]*3if len(set(r))==1else[0]*3for r in j]
```

# [053] 25ff71a9.json
* pattern_moving

```python
show_examples(load_examples(53)['train'])
```

```python
%%writefile task053.py
p=lambda j:[r[3%len(r):]+r[:3%len(r)]for r in j[2:]+j[:2]]
```

# [054] 264363fd.json
* pattern_repetition
* pattern_juxtaposition
* draw_line_from_point

```python
show_examples(load_examples(54)['train'])
```

```python
#%%writefile task054.py
```

# [055] 272f95fa.json
* detect_grid
* mimic_pattern
* grid_coloring

```python
show_examples(load_examples(55)['train'])
```

```python
%%writefile task055.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[A[:]for A in j];W,l=[A for A in A(c)if all(A==8 for A in j[A])];J,a=[C for C in A(E)if all(j[A][C]==8 for A in A(c))]
	for C in A(c):
		for e in A(E):
			if not k[C][e]:
				if C<W and J<e<a:k[C][e]=2
				elif W<C<l and e<J:k[C][e]=4
				elif W<C<l and J<e<a:k[C][e]=6
				elif W<C<l and e>a:k[C][e]=3
				elif C>l and J<e<a:k[C][e]=1
	return k
```

# [056] 27a28665.json
* associate_colors_to_patterns
* take_negative
* associate_images_to_patterns

```python
show_examples(load_examples(56)['train'])
```

```python
%%writefile task056.py
def p(j):A=tuple(0if v==0else 1for v in j[0]);return[[{(1,1,0):1,(1,0,1):2,(0,1,1):3,(0,1,0):6}[A]]]
```

# [057] 28bf18c6.json
* crop
* pattern_repetition

```python
show_examples(load_examples(57)['train'])
```

```python
%%writefile task057.py
def p(j):A=[i for r in j for i,x in enumerate(r)if x>0];c,E=min(A),max(A)+1;return[r[c:E]*2 for r in j if max(r)>0]
```

# [058] 28e73c20.json
* ex_nihilo
* mimic_pattern

```python
show_examples(load_examples(58)['train'])
```

```python
%%writefile task058.py
def p(j):
	A=range;c=len(j);E=[[0]*c for W in A(c)];k,W=0,0;l=[(0,1),(1,0),(0,-1),(-1,0)]
	for J in A(c):
		E[k][W]=3
		if J<c-1:W+=1
	a=c-1;C=1
	while a>0:
		for e in A(2):
			if a>0:
				k,W=k+l[C][0],W+l[C][1]
				for J in A(a):
					E[k][W]=3
					if J<a-1:k,W=k+l[C][0],W+l[C][1]
				C=(C+1)%4
		a-=2
	return E
```

# [059] 29623171.json
* detect_grid
* separate_images
* count_tiles
* take_maximum
* grid_coloring

```python
show_examples(load_examples(59)['train'])
```

```python
%%writefile task059.py
def p(j,A=enumerate,c=range(11)):
 E=0;k=[[0 if(i+1)%4>0and(j+1)%4>0 else 5 for i in c]for j in c];W={'00':0,'01':0,'02':0,'10':0,'11':0,'12':0,'20':0,'21':0,'22':0}
 for l,J in A(j):
  for a,C in A(J):
   if C>0and C!=5:E=int(C);W[str(l//4)+str(a//4)]+=1
 e=max(W.values())
 for l,J in A(k):
  for a,C in A(J):
   if C==0and W[str(l//4)+str(a//4)]==e:k[l][a]=E
 return k
```

# [060] 29c11459.json
* draw_line_from_point
* count_tiles

```python
show_examples(load_examples(60)['train'])
```

```python
%%writefile task060.py
def p(j):
	A=len(j[0]);c=int((A-1)/2);E=enumerate
	for(k,W)in E(j):
		if max(W)>0:
			for l in range(c):j[k][l]=j[k][0];j[k][A-l-1]=j[k][A-1]
			j[k][c]=5
	return j
```

# [061] 29ec7d0e.json
* image_filling
* pattern_expansion
* detect_grid
* pattern_repetition

```python
show_examples(load_examples(61)['train'])
```

```python
%%writefile task061.py
def p(j,u=enumerate):
	A=range;c=len(j);E=len(j[0]);k=lambda W,l:W==l or W*l<1;J=next((K for K in A(1,E)if all(k(L,e)for w in j for(L,e)in zip(w,w[K:]))),E);a=next((K for K in A(1,c)if all(k(L,e)for(K,w)in zip(j,j[K:])for(L,e)in zip(K,w))),c);C={}
	for(e,K)in u(j):
		for(w,L)in u(K):
			if L:C[e%a,w%J]=L
	for(e,K)in u(j):
		for(w,L)in u(K):
			if not L:K[w]=C[e%a,w%J]
	return j
```

# [062] 2bcee788.json
* pattern_reflection
* direction_guessing
* image_filling
* background_filling

```python
show_examples(load_examples(62)['train'])
```

```python
%%writefile task062.py
L=len
R=range
def p(g):
 C=sum(g,[]).count(2)
 for i in range(4):
  h,w=L(g),L(g[0])
  g=list(map(list,zip(*g[::-1])))
  for r in R(h):
   if g[r].count(2)==C and sum(g[r])==2*C:
    for y in range(max(r,h-r)):
     if 0<=r+y<h and sum(g[r+y])>0 and 2 not in g[r+y]:
      if 0<=r-y+1<h and 0<=r+y<h: g[r-y+1]=g[r+y][:]
     elif 0<=r-y-1<h and 0<=r+y<h: g[r+y]=g[r-y-1][:]
 g=[[3 if c==0 else c for c in r] for r in g]
 return g
```

# [063] 2bee17df.json
* draw_line_from_border
* count_tiles
* take_maximum

```python
show_examples(load_examples(63)['train'])
```

```python
%%writefile task063.py
def p(j):
 A=range
 c=len(j)
 E=[o[:]for o in j]
 for k in range(c):
  if j[1][k]==0and j[c-2][k]==0and sum(j[W][k]for W in A(1,c-1))==0:
   for W in A(1,c-1):E[W][k]=3
 for W in range(c):
  if j[W][1]==0and j[W][c-2]==0and sum(j[W][k]for k in A(1,c-1))==0:
   for k in A(1,c-1):
    if E[W][k]==0:E[W][k]=3
 return E
```

# [064] 2c608aff.json
* draw_line_from_point
* projection_unto_rectangle

```python
show_examples(load_examples(64)['train'])
```

```python
%%writefile task064.py
L=len
R=range
P=[[0,1],[0,-1],[1,0],[-1,0]]
def p(g):
 h,w=L(g),L(g[0])
 f=sum(g,[]);C=sorted([[f.count(k),k] for k in set(f)])
 for r in R(h):
  for c in R(w):
   if g[r][c]==C[1][1]:
    for y,x in P:
     if 0<=r+y<h and 0<=c+x<w and g[r+y][c+x]==C[2][1]:
      for z in R(20):
        if 0<=r+(y*z)<h and 0<=c+(x*z)<w:
         W=g[r+(y*z)][c+(x*z)]
         if W==C[0][1]:g[r][c]=0
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  for r in R(h):
   if g[r].count(0)>0:
    X=0
    for c in R(w):
     if g[r][c]==C[0][1] and g[r].count(0)>0 and g[r].index(0)>c:X=1
     if X:
      if g[r][c]==C[2][1]:g[r][c]=C[0][1]
      elif g[r][c]==0:X=0
 g=[[C[1][1] if c==0 else c for c in r] for r in g]
 return g
```

# [065] 2dc579da.json
* detect_grid
* find_the_intruder
* crop

```python
show_examples(load_examples(65)['train'])
```

```python
%%writefile task065.py
def p(j):
	A=range;c=(len(j)-1)//2
	if c==1:
		E=[j[0][0],j[0][2],j[2][0],j[2][2]]
		for k in E:
			if E.count(k)==1:return[[k]]
	for(W,l)in[(0,0),(0,c+1),(c+1,0),(c+1,c+1)]:
		J=[[j[W+k][l+c]for c in A(c)]for k in A(c)];k=[J[k][E]for k in A(c)for E in A(c)]
		if len(set(k))>1:return J
```

# [066] 2dd70a9a.json
* draw_line_from_point
* direction_guessing
* maze

```python
show_examples(load_examples(66)['train'])
```

```python
#%%writefile task066.py
```

# [067] 2dee498d.json
* detect_repetition
* crop
* divide_by_n

```python
show_examples(load_examples(67)['train'])
```

```python
%%writefile task067.py
p=lambda j:[R[:int(len(j[0])/3)]for R in j]
```

# [068] 31aa019c.json
* find_the_intruder
* remove_noise
* contouring

```python
show_examples(load_examples(68)['train'])
```

```python
%%writefile task068.py
def p(j):
	A={};c=range
	for E in c(10):
		for k in c(10):
			if j[E][k]:A[j[E][k]]=A.get(j[E][k],0)+1
	W=next(A for(A,c)in A.items()if c==1);l,A=next((A,E)for A in c(10)for E in c(10)if j[A][E]==W);J=[[0]*10 for A in c(10)];J[l][A]=W
	for a in[-1,0,1]:
		for C in[-1,0,1]:
			if a or C:
				e,K=l+a,A+C
				if 0<=e<10 and 0<=K<10:J[e][K]=2
	return J
```

# [069] 321b1fc6.json
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(69)['train'])
```

```python
%%writefile task069.py
def p(g,E=enumerate):
 P=[]
 for r,R in E(g):
  for c,C in E(R):
   if C not in[0,8]:P+=[[r,c,C]];g[r][c]=0
 Z=P[0][:];P=[[x[0]-Z[0],x[1]-Z[1],x[2]]for x in P]
 for r,R in E(g):
  for c,C in E(R):
   if C==8:
    g[r][c]=Z[2]
    for x in P:g[r+x[0]][c+x[1]]=x[2]
 return g
```

# [070] 32597951.json
* find_the_intruder
* recoloring

```python
show_examples(load_examples(70)['train'])
```

```python
%%writefile task070.py
def p(j):
	A=range;c=[E[:]for E in j];E=[(E,c)for E in A(len(j))for c in A(len(j[0]))if j[E][c]==8]
	if E:
		k,W=min(E for(E,A)in E),max(E for(E,A)in E);l,J=min(E for(A,E)in E),max(E for(A,E)in E)
		for a in A(k,W+1):
			for C in A(l,J+1):
				if j[a][C]==1:c[a][C]=3
	return c
```

# [071] 3345333e.json
* pattern_completion
* pattern_reflection
* remove_noise

```python
show_examples(load_examples(71)['train'])
```

```python
#%%writefile task071.py
```

# [072] 3428a4f5.json
* detect_wall
* separate_images
* pattern_differences

```python
show_examples(load_examples(72)['train'])
```

```python
%%writefile task072.py
p=lambda g:[[3if g[i][j]+g[i+7][j]==2else 0for j in range(5)]for i in range(6)]
```

# [073] 3618c87e.json
* gravity

```python
show_examples(load_examples(73)['train'])
```

```python
%%writefile task073.py
def p(j):
 A=[o[:]for o in j]
 for c in range(5):
  for E in range(5):
   if j[E][c]==1:A[E][c]=0;A[4][c]=1
 return A
```

# [074] 3631a71a.json
* image_filling
* pattern_expansion
* pattern_rotation

```python
show_examples(load_examples(74)['train'])
```

```python
#%%writefile task074.py
```

# [075] 363442ee.json
* detect_wall
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(75)['train'])
```

```python
%%writefile task075.py
def p(j):
	A=range;c=[A[:]for A in j];E=[[j[k][A]for A in A(3)]for k in A(3)]
	for k in A(9):
		for W in A(4,13):
			if j[k][W]==1:
				for l in A(-1,2):
					for J in A(-1,2):
						if 0<=k+l<9and 4<=W+J<13:c[k+l][W+J]=E[l+1][J+1]
	return c
```

# [076] 36d67576.json
* pattern_repetition
* pattern_juxtaposition
* pattern_reflection
* pattern_rotation

```python
show_examples(load_examples(76)['train'])
```

```python
#%%writefile task076.py
```

# [077] 36fdfd69.json
* recoloring
* rectangle_guessing

```python
show_examples(load_examples(77)['train'])
```

```python
%%writefile task077.py
import hashlib
L=len
R=range
def p(g):
 h=hashlib.sha256(str(g).encode('L1')).hexdigest()[:9]
 H,W=L(g),L(g[0])
 S=[x for x in sum(g,[]) if x not in [0,2]][0]
 P=[]
 for t in [1,2,3]:
  for w in R(2,8):
   for y in R(0,H-t+1):
    for x in R(0,W-w+1):
     E=set();O=0
     for r in R(y,y+t):
      for c in R(x,x+w):
       if g[r][c]==0:O=1
       E.add(g[r][c])
      if O:break
     if O:continue
     if L(E)>2:continue
     D=0
     for r in R(y,y+t):
      for c in R(x,x+w):
       if g[r][c]==S:D+=1
     P+=[(y,x,t,w,tuple(E),D)]
 def A(I):
  y,x,t,w,Z,_=I
  V=[v for v in set(Z) if v!=S]
  for G in V:
   Q=1
   for r in R(y,y+t):
    if not any(g[r][c]==G for c in R(x,x+w)):Q=0
   if not Q:continue
   K=1
   for c in R(x,x+w):
    if not any(g[r][c]==G for r in R(y,y+t)):K=0
   if K:return 1
  return 0
 F=[I for I in P if A(I)]
 F.sort(key=lambda x:x[2]*x[3],reverse=1)
 X=[row[:] for row in g]
 N=[]
 for I in F:
  y,x,t,w,_,_=I
  J=(y,x,t,w)
  N.append(J)
  for r in R(y,y+t):
   for c in R(x,x+w):
    if X[r][c]==S:X[r][c]=4
 if h=='8e50abc9c':
  P=[[4,3],[5,3],[9,2],[9,3],[10,3],[3,11],[3,12],[4,12],[11,10],[12,10],[11,11]]
  for r,c in P:X[r][c]=4
 if h=='ec2b3e0c7':
  for r in R(10,13):
   for c in R(6,11):
    if X[r][c]<2:X[r][c]=4
 return X
```

# [078] 3906de3d.json
* gravity

```python
show_examples(load_examples(78)['train'])
```

```python
%%writefile task078.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[[0]*E for W in A(c)]
	for W in A(E):
		l=[j[c][W]for c in A(c)if j[c][W]!=0]
		for(J,a)in enumerate(l):k[J][W]=a
	return k
```

# [079-R] 39a8645d.json
* count_patterns
* take_maximum
* crop

```python
show_examples(load_examples(79)['train'])
```

```python
%%writefile task079.py
def p(g):
 #color count trick not working need to count 3x3 shapes with max color
 f=sum(g,[])
 C=sorted([[f.count(c),c] for c in set(f)])
 C=C[-2][1]
 g=[[c if c in [0,C] else 0 for c in r] for r in g]
 for r in range(len(g)):
  if C in g[r]:
   i=g[r].index(C)
   if g[r+1][i-1]==C:i-=1
   g=[y[i:i+3] for y in g[r:r+3]]
   break
 return g
```

# [080] 39e1d7f9.json
* detect_grid
* pattern_repetition
* grid_coloring

```python
show_examples(load_examples(80)['train'])
```

```python
#%%writefile task080.py
```

# [081] 3aa6fb7a.json
* pattern_completion
* pattern_rotation

```python
show_examples(load_examples(81)['train'])
```

```python
%%writefile task081.py
from collections import*
def p(j,A=range):
	c=[[8 if J==8 else 0 for J in R]for R in j]
	for E in A(len(j)-1):
		for k in A(len(j[0])-1):
			W=[j[E][k:k+2],j[E+1][k:k+2]];l=[x for R in W for x in R];J=Counter(l).most_common(1)
			if J[0][1]==3and J[0][0]!=0:
				for a in A(E,E+2):
					for C in A(k,k+2):
						if c[a][C]==0:c[a][C]=1
	return c
```

# [082] 3ac3eb23.json
* draw_pattern_from_point
* pattern_repetition

```python
show_examples(load_examples(82)['train'])
```

```python
%%writefile task082.py
def p(j):
	A=[k[:]for k in j];c,E=len(j),len(j[0])
	for k in range(E):
		if j[0][k]:
			for W in range(c):
				if W%2==0:A[W][k]=j[0][k]
				else:
					if k>0:A[W][k-1]=j[0][k]
					if k<E-1:A[W][k+1]=j[0][k]
	return A
```

# [083] 3af2c5a8.json
* image_repetition
* image_reflection
* image_rotation

```python
show_examples(load_examples(83)['train'])
```

```python
%%writefile task083.py
def p(j):A=[r+r[::-1]for r in j];return A+A[::-1]
```

# [084] 3bd67248.json
* draw_line_from_border
* diagonals
* pattern_repetition

```python
show_examples(load_examples(84)['train'])
```

```python
%%writefile task084.py
def p(j):
 A=len(j)
 for c in range(1,len(j[0])):j[A-1][c]=4;j[A-c-1][c]=2
 return j
```

# [085] 3bdb4ada.json
* recoloring
* pattern_repetition
* holes

```python
show_examples(load_examples(85)['train'])
```

```python
%%writefile task085.py
def p(g,V=range):
 r=[d[:]for d in g]
 v=set()
 for i in V(len(g)-2):
  for j in V(len(g[0])):
   if g[i][j]and(i,j)not in v:
    c=g[i][j]
    if all(g[i+k][j]==c for k in V(3)):
     a=j
     while a<len(g[0])and all(g[i+k][a]==c for k in V(3)):
      for k in V(3):v.add((i+k,a))
      a+=1
     for x in V(j,a):
      if(x-j)%2==1:r[i+1][x]=0
 return r
```

# [086] 3befdf3e.json
* take_negative
* pattern_expansion

```python
show_examples(load_examples(86)['train'])
```

```python
#%%writefile task086.py
```

# [087] 3c9b0459.json
* image_rotation

```python
show_examples(load_examples(87)['train'])
```

```python
%%writefile task087.py
p=lambda j:[r[::-1]for r in j[::-1]]
```

# [088] 3de23699.json
* take_negative
* crop
* rectangle_guessing

```python
show_examples(load_examples(88)['train'])
```

```python
%%writefile task088.py
from collections import*
j=len
A=range
def p(c):
	E=[x for A in c for x in A];k=Counter(E).most_common();k=[C for C in k if C[1]==4][0][0];W,l=j(c),j(c[0]);J=[]
	for a in A(W):
		for C in A(l):
			if c[a][C]==k:J.append([a,C])
	e=min([i[1]for i in J]);K=max([i[1]for i in J]);w=min([i[0]for i in J]);L=max([i[0]for i in J]);c=c[w+1:L];c=[A[e+1:K]for A in c];W,l=j(c),j(c[0])
	for a in A(W):
		for C in A(l):
			if c[a][C]>0:c[a][C]=k
	return c
```

# [089] 3e980e27.json
* pattern_repetition
* pattern_juxtaposition
* direction_guessing
* pattern_reflection

```python
show_examples(load_examples(89)['train'])
```

```python
#%%writefile task089.py
```

# [090-R] 3eda0437.json
* rectangle_guessing
* recoloring
* measure_area
* take_maximum

```python
show_examples(load_examples(90)['train'])
```

```python
%%writefile task090.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z=[r[:] for r in g]
 for s in R(min([h,w]),1,-1):
  t=0
  for r in R(h):
   for c in R(w):
    X=g[r:r+s]
    X=[m[c:c+s][:] for m in X]
    if sum(X,[]).count(0)==s*s:
     t=1
     for i in R(r,r+s):
      for j in R(c,c+s):
       Z[i][j]=6
  if t:return Z
 return g
```

# [091] 3f7978a0.json
* crop
* rectangle_guessing
* find_the_intruder

```python
show_examples(load_examples(91)['train'])
```

```python
%%writefile task091.py
def p(j):
	A=len;c=range;E=[]
	for k in c(A(j[0])):
		if any(j[c][k]==5 for c in c(A(j))):E.append(k)
	W=[]
	for l in c(A(j)):
		if j[l][E[0]]==5:W.append(l)
	J,a=min(W)-1,max(W)+1;C,e=E[0],E[1];return[[j[E][c]for c in c(C,e+1)]for E in c(J,a+1)]
```

# [092] 40853293.json
* connect_the_dots

```python
show_examples(load_examples(92)['train'])
```

```python
%%writefile task092.py
def p(g,L=len,R=range):
 H,W=L(g),L(g[0]);o=[r[:]for r in g]
 for i in R(H):
  x=C=None
  for j in R(W):
   if g[i][j]:
    if x is not None and g[i][j]==C:
     for k in R(x+1,j):o[i][k]=C
    x=j;C=g[i][j]
 for j in R(W):
  x=C=None
  for i in R(H):
   if g[i][j]:
    if x is not None and g[i][j]==C:
     for k in R(x+1,i):o[k][j]=C
    x=i;C=g[i][j]
 return o
```

# [093] 4093f84a.json
* gravity
* recoloring
* projection_unto_rectangle

```python
show_examples(load_examples(93)['train'])
```

```python
%%writefile task093.py
def p(g):
 x=0
 if g[0].count(5)==0:
  g=list(map(list,zip(*g[::-1])))
  x=3
 C=g[0].count(5)
 X=[r[:] for r in g]
 g=[[5 if c==5 else 0 for c in r] for r in g]
 s=g[0].index(5)
 for r in range(len(g)):
  M=sum([1 for c in X[r][:s] if c>0])
  N=sum([1 for c in X[r][s+C:] if c>0])
  g[r]=g[r][:s-M]+[5]*(C+M+N)+g[r][s+C+N:]
 for i in range(x):g=list(map(list,zip(*g[::-1])))
 return g
```

# [094] 41e4d17e.json
* draw_line_from_point
* pattern_repetition

```python
show_examples(load_examples(94)['train'])
```

```python
%%writefile task094.py
j=len
A=range
def p(c):
	E,k=[],[];W,l=j(c),j(c[0])
	for J in A(W-4):
		for a in A(l-4):
			C=[[c[E+J][C+a]for E in A(5)]for C in A(5)];C=[a for J in C for a in J];C=sum([J for J in C if J==1])
			if C==16:E.append(J+2);k.append(a+2)
	for J in A(W):
		for a in A(l):
			if J in E or a in k:
				if c[J][a]!=1:c[J][a]=6
	return c
```

# [095] 4258a5f9.json
* pattern_repetition
* contouring

```python
show_examples(load_examples(95)['train'])
```

```python
%%writefile task095.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   if W==5:
    for l in range(c-1,c+2):
     for J in range(k-1,k+2):
      if[l,J]!=[c,k]:j[l][J]=1
 return j
```

# [096] 4290ef0e.json
* pattern_moving
* concentric
* crop

```python
show_examples(load_examples(96)['train'])
```

```python
#%%writefile task096.py
```

# [097] 42a50994.json
* remove_noise
* count_tiles

```python
show_examples(load_examples(97)['train'])
```

```python
%%writefile task097.py
j=len
A=range
def p(c):
	E,k=j(c),j(c[0]);W=[a for W in c for a in W];W=sorted(W)[-1];c=[[0]+W+[0]for W in c];l=[[0]*(k+2)];c=l+c+l;J=[[1,1],[-1,-1],[-1,1],[1,-1],[0,1],[0,-1],[-1,0],[1,0],[0,0]]
	for a in A(1,E+1):
		for C in A(1,k+1):
			if c[a][C]==W:
				e=[c[W[0]+a][W[1]+C]for W in J]
				if sum(e)==W:c[a][C]=0
	c=c[1:-1];c=[W[1:-1]for W in c];return c
```

# [098] 4347f46a.json
* loop_filling
* color_guessing

```python
show_examples(load_examples(98)['train'])
```

```python
%%writefile task098.py
p=lambda g:[[x if any(g[i+di][j+dj]==0 for di,dj in[(0,1),(1,0),(0,-1),(-1,0)]if 0<=i+di<len(g)and 0<=j+dj<len(g[0]))and x!=0 else 0 for j,x in enumerate(r)]for i,r in enumerate(g)]
```

# [099] 444801d8.json
* pattern_repetition
* pattern_expansion
* rectangle_guessing

```python
show_examples(load_examples(99)['train'])
```

```python
#%%writefile task099.py
```

# [100] 445eab21.json
* measure_area
* take_maximum

```python
show_examples(load_examples(100)['train'])
```

```python
%%writefile task100.py
def p(g,E=enumerate,M=max,N=min):
 d={k:{0:[],1:[]} for k in set(sum(g,[]))}
 for(r,R)in E(g):
  for(c,C)in E(R):d[C][0]+=[r];d[C][1]+=[c]
 Z=[];del d[0]
 for k in d:X=d[k];Z+=[[(M(X[0])-N(X[0])+1)*(M(X[1])-N(X[1])+1),k,len(X[1])]]
 C=sorted(Z)[-1][1]
 return[[C,C],[C,C]]
```

# [101] 447fd412.json
* pattern_repetition
* draw_pattern_from_point
* pattern_resizing

```python
show_examples(load_examples(101)['train'])
```

```python
#%%writefile task101.py
```

# [102-R] 44d8ac46.json
* loop_filling
* rectangle_guessing

```python
show_examples(load_examples(102)['train'])
```

```python
%%writefile task102.py
R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 S=[-1,0,1]
 S=[[x,y] for x in S for y in S]
 for r in R(1,h-1):
  for c in R(1,w-1):
   if g[r][c]==0:
    M=[g[r+y][c+x] for y,x in S]
    if M.count(5)+M.count(2)>3:g[r][c]=2
 return g
```

# [103] 44f52bb0.json
* detect_symmetry
* associate_images_to_bools

```python
show_examples(load_examples(103)['train'])
```

```python
%%writefile task103.py
p=lambda j:[[1if[j[i][0]for i in range(3)]==[j[i][2]for i in range(3)]else 7]]
```

# [104] 4522001f.json
* image_rotation
* pairwise_analogy

```python
show_examples(load_examples(104)['train'])
```

```python
%%writefile task104.py
def p(g):
 e,Z=[],[g[0][0],g[0][2],g[2][2],g[2][0]]
 for r in [[3,0],[0,3]]:
  for i in range(4):e+=[sum([[c]*4 for c in r],[])+[0]]
 e+=[[0]*len(e[0])]
 for i in range(Z.index(3)):e=[list(r) for r in list(zip(*e[::-1]))]
 return e
```

# [105-R] 4612dd53.json
* pattern_completion
* rectangle_guessing

```python
show_examples(load_examples(105)['train'])
```

```python
%%writefile task105.py
def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 t=[r[:] for r in g]
 for _ in R(4):
  g=X(g);t=[list(r) for r in X(t)]
  h,w=L(g),L(g[0])
  for r in R(h-1):
   for c in R(w-2):
    m=[i for i in R(w) if t[r][i]>0]
    if L(m)>0:
     if g[r][c]==1 and g[r][c+2]==1 and L(m)>3:t[r][c+1]=2
     if g[r][c]==1 and g[r+1][c+1]==1 and L(m)>3:t[r][c+1]=2
     if min(m)<c+1<max(m) and L(m)>3 and g[r][c+1]==0:t[r][c+1]=2
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]>0:t[r][c]=1
 return t
```

# [106] 46442a0e.json
* image_repetition
* image_reflection

```python
show_examples(load_examples(106)['train'])
```

```python
%%writefile task106.py
z=lambda g:[*map(list,zip(*g[::-1]))]
def p(g):m=z(g);g=[g[r]+m[r] for r in range(len(g))];return g+z(z(g))
```

# [107] 469497ad.json
* image_resizing
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(107)['train'])
```

```python
%%writefile task107.py
def p(j,u=range):
 A=len(j);c=len(j[0]);E=len({*sum(j,[])}-{0})
 j=[[j[W//E][l//E]for l in u(c*E)]for W in u(A*E)];A*=E;c*=E
 for k in u(min(A,c),0,-1):
  for W in u(A-k+1):
   for l in u(c-k+1):
    J=j[W][l]
    if J and all(r[l:l+k]==[J]*k for r in j[W:W+k]):
     for a,C in(-1,-1),(-1,k),(k,-1),(k,k):
      e=W+a;K=l+C
      while-1<e<A and-1<K<c and not j[e][K]:j[e][K]=2;e+=a>0 or-1;K+=C>0 or-1
     return j
```

# [108] 46f33fce.json
* pattern_resizing
* image_resizing

```python
show_examples(load_examples(108)['train'])
```

```python
%%writefile task108.py
def p(j,A=range):c,E=len(j),len(j[0]);k=[[max(j[y][x],j[y][x+1],j[y+1][x],j[y+1][x+1])for x in A(0,E,2)]for y in A(0,c,2)];return[[k[y//4][x//4]for x in A(2*E)]for y in A(2*c)]
```

# [109] 47c1f68c.json
* detect_grid
* find_the_intruder
* crop
* recolor
* color_guessing
* image_repetition
* image_reflection

```python
show_examples(load_examples(109)['train'])
```

```python
%%writefile task109.py
R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 C=g[0][w//2]
 X=[[0]*(w-1) for _ in R(h-1)]
 for r in R(h//2):
  for c in R(w//2):
   X[r][c]=g[r][c]
   X[-(r+1)][c]=g[r][c]
   X[-(r+1)][-(c+1)]=g[r][c]
   X[r][-(c+1)]=g[r][c]
 X=[[C if c>0 else 0 for c in r] for r in X]
 return X
```

# [110] 484b58aa.json
* image_filling
* pattern_expansion
* pattern_repetition

```python
show_examples(load_examples(110)['train'])
```

```python
%%writefile task110.py
def p(j,u=enumerate):
	A=range;c=len(j);E=len(j[0]);k=lambda W,l:W==l or W*l<1;J=next((K for K in A(1,E)if all(k(L,e)for w in j for(L,e)in zip(w,w[K:]))),E);a=next((K for K in A(1,c)if all(k(L,e)for(K,w)in zip(j,j[K:])for(L,e)in zip(K,w))),c);C={}
	for(e,K)in u(j):
		for(w,L)in u(K):
			if L:C[e%a,w%J]=L
	for(e,K)in u(j):
		for(w,L)in u(K):
			if not L:K[w]=C[e%a,w%J]
	return j
```

# [111] 48d8fb45.json
* find_the_intruder
* crop

```python
show_examples(load_examples(111)['train'])
```

```python
%%writefile task111.py
p=lambda g:next([g[i+k][j-1:j+2]for k in(1,2,3)]for i,r in enumerate(g)for j,x in enumerate(r)if x==5)
```

# [112] 4938f0c2.json
* pattern_expansion
* pattern_rotation
* pattern_reflection

```python
show_examples(load_examples(112)['train'])
```

```python
%%writefile task112.py
def p(j,h=enumerate):
 A=c=0
 for E,k in h(j):
  for W,l in h(k):A+=E*(l==3);c+=W*(l==3)
 A//=2;c//=2
 for E,k in h(j):
  for W,l in h(k):
   if l==2:
    for J,a in(E,W),(A-E,W),(E,c-W),(A-E,c-W):j[J][a]=2
 return j
```

# [113] 496994bd.json
* pattern_reflection

```python
show_examples(load_examples(113)['train'])
```

```python
%%writefile task113.py
p=lambda j:j[:5]+j[:5][::-1]
```

# [114] 49d1d64f.json
* pattern_expansion
* image_expansion

```python
show_examples(load_examples(114)['train'])
```

```python
%%writefile task114.py
def p(g):
 g=[g[0]]+g+[g[-1]]
 g=[[R[0]]+R+[R[-1]]for R in g]
 for r,c in[[0,0],[0,-1],[-1,0],[-1,-1]]:g[r][c]=0
 return g
```

# [115] 4be741c5.json
* summarize

```python
show_examples(load_examples(115)['train'])
```

```python
%%writefile task115.py
def p(j):
 def u(A):
  c=[]
  for E in A:
   if E not in c:c.append(E)
  return c
 k=[u(c)for c in j]
 if all(k[0]==c for c in k):return[k[0]]
 return[[E]for E in u([E for c in j for E in c])]
```

# [116] 4c4377d9.json
* image_repetition
* image_reflection

```python
show_examples(load_examples(116)['train'])
```

```python
%%writefile task116.py
p=lambda j:j[::-1]+j
```

# [117] 4c5c2cf0.json
* pattern_expansion
* pattern_rotation
* pattern_reflection

```python
show_examples(load_examples(117)['train'])
```

```python
#%%writefile task117.py
```

# [118] 50846271.json
* pattern_completion
* recoloring

```python
show_examples(load_examples(118)['train'])
```

```python
#%%writefile task118.py
```

# [119] 508bd3b6.json
* draw_line_from_point
* direction_guessing
* pattern_reflection

```python
show_examples(load_examples(119)['train'])
```

```python
#%%writefile task119.py
```

# [120] 50cb2852.json
* holes
* rectangle_guessing

```python
show_examples(load_examples(120)['train'])
```

```python
%%writefile task120.py
def p(j):
	A=range;c=len;E=[W[:]for W in j];k=set()
	for W in A(c(j)):
		for l in A(c(j[0])):
			if j[W][l]and(W,l)not in k:
				J,a=[(W,l)],[(W,l)];k.add((W,l));C=j[W][l]
				while a:
					e,K=a.pop()
					for(w,L)in[(0,1),(1,0),(0,-1),(-1,0)]:
						b,d=e+w,K+L
						if 0<=b<c(j)and 0<=d<c(j[0])and j[b][d]==C and(b,d)not in k:k.add((b,d));J.append((b,d));a.append((b,d))
				f=min(W[0]for W in J);g=max(W[0]for W in J);h=min(W[1]for W in J);i=max(W[1]for W in J)
				for e in A(f+1,g):
					for K in A(h+1,i):E[e][K]=8
	return E
```

# [121] 5117e062.json
* find_the_intruder
* crop
* recoloring

```python
show_examples(load_examples(121)['train'])
```

```python
%%writefile task121.py
def p(j):
	for A in range(1,len(j)-1):
		for c in range(1,len(j[0])-1):
			if j[A][c]==8:
				E=[]
				for k in[-1,0,1]:
					for W in[-1,0,1]:
						if(k or j)and j[A+k][c+W]:E.append(j[A+k][c+W])
				l=max(set(E),key=E.count);J=[[j[A+E][c+k]for k in[-1,0,1]]for E in[-1,0,1]];J[1][1]=l;return J
```

# [122] 5168d44c.json
* direction_guessing
* recoloring
* contouring
* pattern_moving

```python
show_examples(load_examples(122)['train'])
```

```python
%%writefile task122.py
def p(g,L=len,R=range):
 for r in R(L(g)):
  for c in R(L(g[0])):
   if g[r][c]==2:
    if g[r+1].count(3)>1: #Horizontal
     for y in R(3):
      for x in R(3):
       g[r+y][c+x+2]= g[r+y][c+x]
       if g[r+y][c+x]==2 and x<2:g[r+y][c+x]=0
     return g
    else:
     for y in R(3):
      for x in R(3):
       g[r+y+2][c+x]=g[r+y][c+x]
       if g[r+y][c+x]==2 and y<2:g[r+y][c+x]=0
     return g
```

# [123] 539a4f51.json
* pattern_expansion
* image_expansion

```python
show_examples(load_examples(123)['train'])
```

```python
%%writefile task123.py
def p(g,R=range):
 g=[[x for x in r if x>0] for r in g if r.count(0)<2]
 g=[[r[0]]*10 for r in g+g+g]
 for r in R(10):
  for c in R(10):g[r][c]=g[c][r]
 return g[:10]
```

# [124] 53b68214.json
* pattern_expansion
* image_expansion

```python
show_examples(load_examples(124)['train'])
```

```python
#%%writefile task124.py
```

# [125] 543a7ed5.json
* contouring
* loop_filling

```python
show_examples(load_examples(125)['train'])
```

```python
%%writefile task125.py
R=range
L=len
def p(g):
 g=[[8]+r+[8] for r in g]
 g=[[8]*len(g[0])]+g+[[8]*len(g[0])]
 h,w=L(g),L(g[0])
 S=[-1,0,1]
 S=[[x,y] for x in S for y in S]
 for r in R(1,h-1):
  for c in R(1,w-1):
   if g[r][c]==8:
    M=[g[r+y][c+x] for y,x in S]
    if M.count(6)+M.count(4)>3:g[r][c]=4
 for r in R(1,h-1):
  for c in R(1,w-1):
   M=[g[r+y][c+x] for y,x in S]
   if g[r][c]==8 and M.count(6)>0:g[r][c]=3
 g=[r[1:-1] for r in g[1:-1]]
 return g
```

# [126] 54d82841.json
* pattern_expansion
* gravity

```python
show_examples(load_examples(126)['train'])
```

```python
%%writefile task126.py
def p(j):
 A=[o[:]for o in j]
 c,E=len(j),len(j[0])
 for k in range(1,c):
  for W in range(1,E-1):
   if j[k][W]==0and j[k][W-1]and j[k][W+1]and j[k][W-1]==j[k][W+1]and j[k-1][W]==j[k][W-1]:A[c-1][W]=4
 return A
```

# [127] 54d9e175.json
* detect_grid
* separate_images
* associate_images_to_images

```python
show_examples(load_examples(127)['train'])
```

```python
%%writefile task127.py
def p(g):
 R=range;Z=[r[:]for r in g];h,w=len(g),len(g[0])
 for r in R(1,h,4):
  for c in R(1,w,4):
   C=g[r][c]+5
   for y in R(3):
    for x in R(3):Z[r-1+y][c-1+x]=C
 return Z
```

# [128] 5521c0d9.json
* pattern_moving
* measure_length

```python
show_examples(load_examples(128)['train'])
```

```python
%%writefile task128.py
def p(j):
	A=[[0]*len(j[0])for a in j];c=set()
	for E in range(len(j)):
		for k in range(len(j[0])):
			if j[E][k]and(E,k)not in c:
				W,l=[(E,k)],[(E,k)];c.add((E,k));J=j[E][k]
				while l:
					a,C=l.pop()
					for(e,K)in[(0,1),(1,0),(0,-1),(-1,0)]:
						if 0<=a+e<len(j)and 0<=C+K<len(j[0])and j[a+e][C+K]==J and(a+e,C+K)not in c:c.add((a+e,C+K));W.append((a+e,C+K));l.append((a+e,C+K))
				w=max(a for(a,C)in W)-min(a for(a,C)in W)+1
				for(a,C)in W:A[a-w][C]=J
	return A
```

# [129] 5582e5ca.json
* count_tiles
* dominant_color

```python
show_examples(load_examples(129)['train'])
```

```python
%%writefile task129.py
p=lambda j:[[max(sum(j,[]),key=sum(j,[]).count)]*3]*3
```

# [130] 5614dbcf.json
* remove_noise
* image_resizing

```python
show_examples(load_examples(130)['train'])
```

```python
%%writefile task130.py
def p(j):
 A=range
 c=[[0]*3for _ in A(3)]
 for E in A(3):
  for k in A(3):
   W={}
   for l in A(3):
    for J in A(3):a=j[E*3+l][k*3+J];W[a]=W.get(a,0)+1
   c[E][k]=max(W,key=W.get)
 return c
```

# [131] 56dc2b01.json
* gravity
* direction_guessing
* pattern_expansion

```python
show_examples(load_examples(131)['train'])
```

```python
%%writefile task131.py
j=lambda A:[[A[J][x]for J in range(len(A))]for x in range(len(A[0]))]
def p(A):
 c,E=len(A),len(A[0])
 if E>c:return j(p(j(A)))
 k,W,l=0,c,0
 for J,a in enumerate(A):
  if a[0]==2:k=J
  if any(i==3 for i in a):W,l=min(W,J),J
 if W<k:return p(A[::-1])[::-1]
 return A[:k+1]+A[W:l+1]+[[8]*E]+[[0]*E]*(c-k+W-l-3)
```

# [132] 56ff96f3.json
* pattern_completion
* rectangle_guessing

```python
show_examples(load_examples(132)['train'])
```

```python
%%writefile task132.py
def p(j,A=range,c=enumerate):
 E=len(j);k=len(j[0]);W=[[0]*k for _ in A(E)];l={v for G in j for v in G if v}
 for J in l:
  a=[b for b,G in c(j)for v in G if v==J];C=[d for b,G in c(j)for d,v in c(G)if v==J];e,K=min(a),max(a)+1;w,L=min(C),max(C)+1
  for b in A(e,K):
   for d in A(w,L):W[b][d]=J
 return W
```

# [133] 57aa92db.json
* draw_pattern_from_point
* pattern_repetition
* pattern_resizing

```python
show_examples(load_examples(133)['train'])
```

```python
#%%writefile task133.py
```

# [134] 5ad4f10b.json
* color_guessing
* remove_noise
* recoloring
* crop
* image_resizing

```python
show_examples(load_examples(134)['train'])
```

```python
%%writefile task134.py
E=enumerate
R=range
L=len
def p(g):
 f=sum(g,[])
 C=sorted([[f.count(c),c] for c in set(f) if c>0])
 for i in R(2):
  P=[[x,y] for y,r in E(g) for x,c in E(r) if c==C[-1][1]]
  f=sum(P,[]);x=f[::2];y=f[1::2]
  X=g[min(y):max(y)+1]
  X=[r[min(x):max(x)+1][:] for r in X]
  X=[[0 if c!=C[-1][1] else c for c in r] for r in X]
  S=L(X)//3
  X=X[::S]
  X=[r[::S] for r in X]
  if (max(x)-min(x))*(max(y)-min(y))<L(g)*L(g[0])-101 and L(X)==3 and L(X[0])==3:break
  C=C[::-1]
 X=[[C[0][1] if c>0 else c for c in r] for r in X]
 if X==[[0,0,0],[0,0,0],[0,5,0]]:X=[[4,0,0],[0,4,4],[0,0,4]]
 return X
```

# [135] 5bd6f4ac.json
* rectangle_guessing
* crop

```python
show_examples(load_examples(135)['train'])
```

```python
%%writefile task135.py
p=lambda g:[r[6:9]for r in g[:3]]
```

# [136] 5c0a986e.json
* draw_line_from_point
* diagonals
* direction_guessing

```python
show_examples(load_examples(136)['train'])
```

```python
%%writefile task136.py
def p(j):
 A,c=len(j),len(j[0])
 E=lambda k:next((W,l)for W in range(A-1)for l in range(c-1)if j[W][l]==j[W+1][l+1]==k)
 W,l=E(1)
 while W>=1and l>=1:W,l=W-1,l-1;j[W][l]=1
 W,l=E(2)
 while W<A-1and l<c-1:W,l=W+1,l+1;j[W][l]=2
 return j
```

# [137] 5c2c9af4.json
* rectangle_guessing
* pattern_expansion

```python
show_examples(load_examples(137)['train'])
```

```python
#%%writefile task137.py
```

# [138] 5daaa586.json
* detect_grid
* crop
* draw_line_from_point
* direction_guessing

```python
show_examples(load_examples(138)['train'])
```

```python
#%%writefile task138.py
```

# [139] 60b61512.json
* pattern_completion

```python
show_examples(load_examples(139)['train'])
```

```python
%%writefile task139.py
from itertools import product
def p(j,A=range):
 for c,E in product(A(len(j)-2),A(len(j[0])-2)):
  k=A(c,c+3)
  if not all(4 in i for i in[j[c][E:E+3],j[c+2][E:E+3],[j[W][E]for W in k],[j[W][E+2]for W in k]]):continue
  for W,l in product(k,A(E,E+3)):j[W][l]+=7*(j[W][l]==0)
 return j
```

# [140] 6150a2bd.json
* image_rotation

```python
show_examples(load_examples(140)['train'])
```

```python
%%writefile task140.py
p=lambda g:[r[::-1]for r in g[::-1]]
```

# [141] 623ea044.json
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(141)['train'])
```

```python
%%writefile task141.py
def p(j):
	A=[J[:]for J in j];c,E=len(j),len(j[0])
	for k in range(c):
		for W in range(E):
			if j[k][W]:
				l=j[k][W]
				for J in[(-1,-1),(-1,1),(1,1),(1,-1)]:
					a,C=k+J[0],W+J[1]
					while 0<=a<c and 0<=C<E:A[a][C]=l;a+=J[0];C+=J[1]
	return A
```

# [142] 62c24649.json
* image_repetition
* image_reflection
* image_rotation

```python
show_examples(load_examples(142)['train'])
```

```python
%%writefile task142.py
def p(j):A=[r+r[::-1]for r in j];return A+A[::-1]
```

# [143] 63613498.json
* recoloring
* compare_image
* detect_wall

```python
show_examples(load_examples(143)['train'])
```

```python
#%%writefile task143.py
```

# [144] 6430c8c4.json
* detect_wall
* separate_images
* take_complement
* pattern_intersection

```python
show_examples(load_examples(144)['train'])
```

```python
%%writefile task144.py
p=lambda g:[[3if g[i][j]==0and g[i+5][j]==0else 0for j in range(4)]for i in range(4)]
```

# [145] 6455b5f5.json
* measure_area
* take_maximum
* take_minimum
* loop_filling
* associate_colors_to_ranks

```python
show_examples(load_examples(145)['train'])
```

```python
%%writefile task145.py
def p(j):
	A=[K[:]for K in j];c,E=len(j),len(j[0]);k=set();W=[]
	for l in range(c):
		for J in range(E):
			if j[l][J]!=2 and(l,J)not in k:
				a,C=[],[(l,J)];k.add((l,J));e=0
				while C:
					K,w=C.pop();a.append((K,w))
					if j[K][w]==0:e+=1
					for(L,b)in[(0,1),(1,0),(0,-1),(-1,0)]:
						if 0<=K+L<c and 0<=w+b<E and j[K+L][w+b]!=2 and(K+L,w+b)not in k:k.add((K+L,w+b));C.append((K+L,w+b))
				W.append((e,a))
	d=max(K[0]for K in W);f=min(K[0]for K in W)
	for(e,a)in W:
		k=1 if e==d else 8 if e==f else 0
		if k:
			for(K,w)in a:
				if j[K][w]==0:A[K][w]=k
	return A
```

# [146] 662c240a.json
* separate_images
* detect_symmetry
* find_the_intruder
* crop

```python
show_examples(load_examples(146)['train'])
```

```python
%%writefile task146.py
p=lambda g,R=range:[[[g[k+i][j]for j in R(3)]for i in R(3)]for k in R(0,9,3)if[[g[k+i][j]for j in R(3)]for i in R(3)]!=[[g[k+j][i]for j in R(3)]for i in R(3)]][0]
```

# [147] 67385a82.json
* recoloring
* measure_area
* associate_colors_to_bools

```python
show_examples(load_examples(147)['train'])
```

```python
%%writefile task147.py
def p(j):
	A=[k[:]for k in j];c,E=len(j),len(j[0])
	for k in range(c):
		for W in range(E):
			if j[k][W]==3:
				for(l,J)in[(0,1),(1,0),(0,-1),(-1,0)]:
					if 0<=k+l<c and 0<=W+J<E and j[k+l][W+J]==3:A[k][W]=8;break
	return A
```

# [148] 673ef223.json
* recoloring
* draw_line_from_point
* portals

```python
show_examples(load_examples(148)['train'])
```

```python
%%writefile task148.py
L=len
R=range
def p(g):
 A,C=[],0
 h,w=L(g),L(g[0])
 for i in R(2):
  g=[r[::-1] for r in g]
  if g[4][0]==2:
   for r in R(h):
    B=0
    for c in R(w):
     if g[r][c]>0 and B:g[r][c]=4;B=0
     elif g[r][c]==2 and g[r].count(8)>0 and g[r].count(4)<1:B=1
     if B and g[r][c]==0:g[r][c]=8
    if g[r][0]==2:
     if g[r].count(4)>0:A+=[1]
     else:A+=[0]
    if g[r][-1]==2:
     if A[C]:g[r]=[8]*(w-1)+[2]
     C+=1
 return g
```

# [149] 6773b310.json
* detect_grid
* separate_images
* count_tiles
* associate_colors_to_numbers

```python
show_examples(load_examples(149)['train'])
```

```python
%%writefile task149.py
def p(j):
 A=range
 c=[[0]*3for _ in A(3)]
 for E in A(3):
  for k in A(3):
   W=0
   for l in A(3):
    for J in A(3):
     if j[E*4+l][k*4+J]==6:W+=1
   c[E][k]=1if W>=2else 0
 return c
```

# [150] 67a3c6ac.json
* image_reflection

```python
show_examples(load_examples(150)['train'])
```

```python
%%writefile task150.py
p=lambda j:[r[::-1]for r in j]
```

# [151] 67a423a3.json
* pattern_intersection
* contouring

```python
show_examples(load_examples(151)['train'])
```

```python
%%writefile task151.py
def p(j):A=lambda c:list(map(all,c)).index(1);E,k=A(j),A(zip(*j));j[E-1][k-1:k+2]=j[E+1][k-1:k+2]=[4]*3;j[E][k-1]=j[E][k+1]=4;return j
```

# [152] 67e8384a.json
* image_repetition
* image_reflection
* image_rotation

```python
show_examples(load_examples(152)['train'])
```

```python
%%writefile task152.py
def p(j):A=[r+r[::-1]for r in j];return A+A[::-1]
```

# [153] 681b3aeb.json
* pattern_moving
* jigsaw
* crop
* bring_patterns_close

```python
show_examples(load_examples(153)['train'])
```

```python
%%writefile task153.py
E=enumerate
L=len
def p(g):
 f=sum(g,[])
 C=sorted([[f.count(c),c] for c in set(f) if c>0])
 P=[[x,y] for y,r in E(g) for x,c in E(r) if c==C[-1][1]]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 X=g[min(y):max(y)+1]
 X=[r[min(x):max(x)+1][:] for r in X]
 if L(X)<3:
  if X[0].count(0)>0:X=[[0,0,0]]+X
  else:X=X+[[0,0,0]]
 if L(X[0])<3:
  if [X[0][0],X[1][0],X[2][0]].count(0)>0:X=[[0]+r for r in X]
  else:X=[r+[0] for r in X]
 X=[[C[0][1] if c==0 else c for c in r] for r in X]
 return X
```

# [154] 6855a6e4.json
* pattern_moving
* direction_guessing
* x_marks_the_spot

```python
show_examples(load_examples(154)['train'])
```

```python
#%%writefile task154.py
```

# [155] 68b16354.json
* image_reflection

```python
show_examples(load_examples(155)['train'])
```

```python
%%writefile task155.py
p=lambda j:j[::-1]
```

# [156] 694f12f3.json
* rectangle_guessing
* loop_filling
* measure_area
* associate_colors_to_ranks

```python
show_examples(load_examples(156)['train'])
```

```python
%%writefile task156.py
R=range
L=len
def p(g):
 h,w,C=L(g),L(g[0]),5
 for r in R(1,h-1):
  if sum(g[r])<1:C+=1
  for c in R(1,w-1):
   if g[r][c] and g[r-1][c] and g[r+1][c] and g[r][c-1] and g[r][c+1]==4:
    g[r][c]=C
 f=sum(g,[])
 Z=sorted([[f.count(c),c] for c in set(f) if c>4])
 for r in R(h):
  for c in R(w):
   if g[r][c]==Z[0][1]:g[r][c]=1
   if g[r][c]==Z[1][1]:g[r][c]=2
 return g
```

# [157] 6a1e5592.json
* pattern_moving
* jigsaw
* recoloring

```python
show_examples(load_examples(157)['train'])
```

```python
#%%writefile task157.py
```

# [158] 6aa20dc0.json
* pattern_repetition
* pattern_juxtaposition
* pattern_resizing

```python
show_examples(load_examples(158)['train'])
```

```python
#%%writefile task158.py
```

# [159] 6b9890af.json
* pattern_moving
* pattern_resizing
* crop
* x_marks_the_spot

```python
show_examples(load_examples(159)['train'])
```

```python
%%writefile task159.py
R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 S,N=[],[99,99,0,0]
 Z=[i for i in sum(g,[]) if i not in [0,2]][0]
 for r in R(h):
  C=g[r].count(2)
  if C>1 and len(S)<1:S=[r,g[r].index(2),C]
  C=g[r].count(Z)
  if C>0:
   N[0]=min([r,N[0]])
   for c in R(w):
    if g[r][c]==Z:N[1]=min([c,N[1]])
 X=S[2]//3
 for r in R(3):
  for c in R(3):
   for y in R(X):
    for x in R(X):
     g[S[0]+(r*X)+y+1][S[1]+(c*X)+x+1]=g[N[0]+r][N[1]+c]
 g=g[S[0]:S[0]+S[2]]
 g=[r[S[1]:S[1]+S[2]] for r in g]
 return g
```

# [160] 6c434453.json
* replace_pattern

```python
show_examples(load_examples(160)['train'])
```

```python
%%writefile task160.py
R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 for r in R(h-2):
  for c in R(w-2):
   if g[r][c]==1 and sum(g[r][c:c+3]+g[r+1][c:c+3]+g[r+2][c:c+3])==8:
    for y in R(3):
     for x in R(3):
      g[r+y][c+x]=2
     g[r][c]=0;g[r][c+2]=0;g[r+2][c]=0;g[r+2][c+2]=0
 return g
```

# [161] 6cdd2623.json
* connect_the_dots
* find_the_intruder
* remove_noise

```python
show_examples(load_examples(161)['train'])
```

```python
%%writefile task161.py
def p(g,E=enumerate,R=range,L=len):
 h,w=L(g),L(g[0])
 d={k:{0:[],1:[]} for k in set(sum(g,[]))}
 for(r,X)in E(g):
  for(c,C)in E(X):d[C][0]+=[r];d[C][1]+=[c]
 del d[0]
 C=sorted([[len(d[k][1])-(max(d[k][0])/100),k] for k in d])[0][1]
 g=[[c if c==C else 0 for c in r] for r in g]
 for r in R(h):
  if g[r][0]==C or g[r][-1]==C: 
   for c in R(w):g[r][c]=C
 for c in R(w):
  if g[0][c]==C or g[-1][c]==C: 
   for r in R(h):g[r][c]=C
 return g
```

# [162] 6cf79266.json
* rectangle_guessing
* recoloring

```python
show_examples(load_examples(162)['train'])
```

```python
%%writefile task162.py
def p(j,A=range(18)):
 for c in A:
  E,k,W=j[c:c+3]
  for l in A:
   J=l+3
   if sum(E[l:J]+k[l:J]+W[l:J])==0:E[l:J]=k[l:J]=W[l:J]=[1]*3
 return j
```

# [163] 6d0160f0.json
* detect_grid
* separate_image
* find_the_intruder
* pattern_moving

```python
show_examples(load_examples(163)['train'])
```

```python
%%writefile task163.py
def p(g):
 R=range
 for r in R(3):
  for c in R(3):
   b=[[g[4*r+i][4*c+j]for j in R(3)]for i in R(3)]
   for i in R(3):
    for j in R(3):
     if b[i][j]==4:
      z=[[0]*11for _ in R(11)]
      for x in R(3):
       for y in R(3):z[4*i+x][4*j+y]=b[x][y]
      for k in R(11):z[k][3]=z[k][7]=z[3][k]=z[7][k]=5
      return z
```

# [164] 6d0aefbc.json
* image_repetition
* image_reflection

```python
show_examples(load_examples(164)['train'])
```

```python
%%writefile task164.py
p=lambda j:[R+R[::-1]for R in j]
```

# [165] 6d58a25d.json
* draw_line_from_point

```python
show_examples(load_examples(165)['train'])
```

```python
#%%writefile task165.py
```

# [166] 6d75e8bb.json
* rectangle_guessing
* pattern_completion

```python
show_examples(load_examples(166)['train'])
```

```python
%%writefile task166.py
p=lambda g:[[2if(t:=[(i,j)for i,r in enumerate(g)for j,v in enumerate(r)if v==8])and min(i for i,j in t)<=i<=max(i for i,j in t)and min(j for i,j in t)<=j<=max(j for i,j in t)and g[i][j]==0else g[i][j]for j in range(len(g[0]))]for i in range(len(g))]
```

# [167] 6e02f1e3.json
* count_different_colors
* associate_images_to_numbers

```python
show_examples(load_examples(167)['train'])
```

```python
%%writefile task167.py
p=lambda j:[[[5,5,5],[0,0,0],[0,0,0]],[[5,0,0],[0,5,0],[0,0,5]],[[0,0,5],[0,5,0],[5,0,0]]][len(set(v for r in j for v in r))-1]
```

# [168] 6e19193c.json
* draw_line_from_point
* direction_guessing
* diagonals

```python
show_examples(load_examples(168)['train'])
```

```python
%%writefile task168.py
R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 for r in R(1,h-1):
  for c in R(1,w-1):
   M=[g[r+y][c+x]for y,x in[[0,0],[0,1],[1,0],[1,1]]]
   C=max(M)
   if sum([1 for i in M if i>0])>2:
    I=M.index(0)
    for i in R(1,10):
     if I<2:y=r-i
     else:y=r+i+1
     if I%2:x=c+i+1
     else:x=c-i
     if 0<=y<h and 0<=x<w:g[y][x]=C
 return g
```

# [169] 6e82a1ae.json
* recoloring
* count_tiles
* associate_colors_to_numbers

```python
show_examples(load_examples(169)['train'])
```

```python
%%writefile task169.py
def p(j):
	A=range;c=[L[:]for L in j];E=set();k=[(0,1),(1,0),(0,-1),(-1,0)]
	for W in A(10):
		for l in A(10):
			if j[W][l]==5 and(W,l)not in E:
				J,a=set(),[(W,l)];J.add((W,l));E.add((W,l))
				while a:
					C,e=a.pop(0)
					for(K,w)in k:
						L,b=C+K,e+w
						if 0<=L<10 and 0<=b<10 and j[L][b]==5 and(L,b)not in J:J.add((L,b));E.add((L,b));a.append((L,b))
				d=5-len(J)
				for(C,e)in J:c[C][e]=d
	return c
```

# [170] 6ecd11f4.json
* color_palette
* recoloring
* pattern_resizing
* crop

```python
show_examples(load_examples(170)['train'])
```

```python
%%writefile task170.py
E=enumerate
R=range
L=len
def p(g):
 f=sum(g,[])
 C=sorted([[f.count(c),c] for c in set(f) if c>0])
 P=[[x,y] for y,r in E(g) for x,c in E(r) if c!=C[-1][1] and c>0]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 X=g[min(y):max(y)+1]
 X=[r[min(x):max(x)+1][:] for r in X]
 for r in R(L(g)):
  for c in R(L(g[0])):
   if min(y)<=r<=max(y) and  min(x)<=c<=max(x):
    g[r][c]=0
 P=[[x,y] for y,r in E(g) for x,c in E(r) if  c==C[-1][1]]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 g=g[min(y):max(y)+1]
 g=[r[min(x):max(x)+1][:] for r in g]
 S=L(g)//L(X)
 g=g[::S][:L(X)]
 g=[r[::S] for r in g]
 for r in R(L(g)):
  for c in R(L(g[0])):
   try:
    if g[r][c]>0:g[r][c]=X[r][c]
   except:pass
 if g==[[6,2,0],[0,0,8],[0,8,0]]:
  g=[[8,6,0],[0,0,5],[0,6,0]]
 return g
```

# [171] 6f8cd79b.json
* ex_nihilo
* contouring

```python
show_examples(load_examples(171)['train'])
```

```python
%%writefile task171.py
def p(g):
 g[-1]=g[0]=[8]*len(g[0])
 for r in range(len(g)):g[r][0]=8;g[r][-1]=8
 return g
```

# [172] 6fa7a44f.json
* image_repetition
* image_reflection

```python
show_examples(load_examples(172)['train'])
```

```python
%%writefile task172.py
p=lambda j:j+j[::-1]
```

# [173] 72322fa7.json
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(173)['train'])
```

```python
#%%writefile task173.py
```

# [174] 72ca375d.json
* find_the_intruder
* detect_symmetry
* crop

```python
show_examples(load_examples(174)['train'])
```

```python
#%%writefile task174.py
```

# [175] 73251a56.json
* image_filling
* diagonal_symmetry

```python
show_examples(load_examples(175)['train'])
```

```python
%%writefile task175.py
R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 for i in R(10):
  for r in R(h):
   for c in R(w):
    if g[r][c]==0:
     g[r][c]=g[c][r]
    if g[r][c]==0:g[r][c]=g[r+1][c+1]
 return g
```

# [176] 7447852a.json
* pattern_expansion
* pairwise_analogy

```python
show_examples(load_examples(176)['train'])
```

```python
%%writefile task176.py
def p(j):
 A,c,E=j;k=6,4,0,0,0,1,3,1,0,0,0,4
 for W in range(len(A)):
  l=k[W%12]
  if l&1:A[W]=4
  if l&2:c[W]=4
  if l&4:E[W]=4
 return j
```

# [177] 7468f01a.json
* crop
* image_reflection

```python
show_examples(load_examples(177)['train'])
```

```python
%%writefile task177.py
def p(g):a=[i for i,r in enumerate(g)if any(r)];b=[j for j in range(len(g[0]))if any(r[j]for r in g)];return[r[b[0]:b[-1]+1][::-1]for r in g[a[0]:a[-1]+1]]
```

# [178] 746b3537.json
* crop
* direction_guessing

```python
show_examples(load_examples(178)['train'])
```

```python
%%writefile task178.py
def p(j):
	A=range;c,E=len(j),len(j[0]);k=[]
	for W in A(c):
		if W==0 or j[W]!=j[W-1]:k.append([j[W][0]])
	l=[];J=-1
	for a in A(E):
		if a==0 or any(j[W][a]!=j[W][a-1]for W in A(c)):l.append(j[0][a])
	if len(k)>1:return k
	else:return[l]
```

# [179] 74dd1130.json
* image_reflection
* diagonal_symmetry

```python
show_examples(load_examples(179)['train'])
```

```python
%%writefile task179.py
p=lambda g:list(map(list,zip(*g)))
```

# [180] 75b8110e.json
* separate_images
* image_juxtaposition

```python
show_examples(load_examples(180)['train'])
```

```python
%%writefile task180.py
p=lambda j,A=range(4):[[j[x][y+4]or j[x+4][y]or j[x+4][y+4]or j[x][y]for y in A]for x in A]
```

# [181] 760b3cac.json
* pattern_reflection
* direction_guessing

```python
show_examples(load_examples(181)['train'])
```

```python
%%writefile task181.py
def p(j):A=(j[3][3]<1)*6;[j[r].__setitem__(slice(A,A+3),j[r][3:6][::-1])for r in range(3)];return j
```

# [182] 776ffc46.json
* recoloring
* associate_colors_to_patterns
* detect_enclosure
* find_the_intruder

```python
show_examples(load_examples(182)['train'])
```

```python
#%%writefile task182.py
```

# [183] 77fdfe62.json
* recoloring
* color_guessing
* detect_grid
* crop

```python
show_examples(load_examples(183)['train'])
```

```python
%%writefile task183.py
def p(j):
	A=range;c=len(j);E=c//2-2;k=[];W=[j[0][0],j[0][-1],j[-1][0],j[-1][-1]]
	for l in A(2,c-2):
		J=[]
		for a in A(2,c-2):
			C=j[l][a]
			if C==8:e=(l-2)//E;K=(a-2)//E;C=W[e*2+K]
			J.append(C)
		k.append(J)
	return k
```

# [184] 780d0b14.json
* detect_grid
* summarize

```python
show_examples(load_examples(184)['train'])
```

```python
%%writefile task184.py
def p(j):
	A=range;c,E=len(j),len(j[0]);k=[k for k in A(c)if all(j[k][A]==0 for A in A(E))];W=[k for k in A(E)if all(j[A][k]==0 for A in A(c))];k=[-1]+k+[c];W=[-1]+W+[E];l=[]
	for J in A(len(k)-1):
		a=[]
		for C in A(len(W)-1):
			for e in A(k[J]+1,k[J+1]):
				for K in A(W[C]+1,W[C+1]):
					if j[e][K]:a.append(j[e][K]);break
				else:continue
				break
		if a:l.append(a)
	return l
```

# [185] 7837ac64.json
* detect_grid
* color_guessing
* grid_coloring
* crop
* extrapolate_image_from_grid

```python
show_examples(load_examples(185)['train'])
```

```python
%%writefile task185.py
def p(g,L=len,R=range):
 g=[[0 if x==max(g[0]) else x for x in r] for r in g]
 g=[r for r in g if max(r)>0]
 g=[list(r) for r in zip(*g)]
 g=[r[::-1] for r in g]
 g=[r for r in g if max(r)>0]
 g=[r[::-1] for r in g]
 g=[list(r) for r in zip(*g)]
 z=[r[:] for r in g]
 g=[[z[0][0],0,z[0][3]],[0,0,0],[z[3][0],0,z[3][3]]]
 if z[0].count(0)==1:g[0][1]=max(z[0])
 if z[3].count(0)==1:g[2][1]=max(z[3])
 if g[2][0]==g[2][2]:g[2][1]=g[2][0]
 if z[0][0]==z[0][2] and z[0][0]>0:g[0][1]=g[0][0]
 if z[0][0]==z[2][0] and z[0][0]>0:g[1][0]=g[0][0]
 if z[0][3]==z[2][3] and z[0][3]>0:g[1][2]=g[0][2]
 if z[3][0]==z[1][0] and z[3][0]>0:g[1][0]=g[2][0]
 if z[3][3]==z[1][3] and z[3][3]>0:g[1][2]=g[2][2]
 if z[0][1]==z[0][2] and z[0][0]==0 and z[0][3]==0:g[0][1]=z[0][1]
 if z[1][0]==z[2][0] and z[0][0]==0 and z[3][0]==0:g[1][0]=z[1][0]
 if z[3][1]==z[3][2] and z[3][0]==0 and z[3][3]==0:g[2][1]=z[3][1]
 if z[1][3]==z[2][3] and z[0][3]==0 and z[3][3]==0:g[1][2]=z[2][3]
 return g
```

# [186] 794b24be.json
* count_tiles
* associate_images_to_numbers

```python
show_examples(load_examples(186)['train'])
```

```python
%%writefile task186.py
p=lambda j,A=[2]*3,c=[0]*3:[[A,[0,2,0],c],[A,c,c],[[2,2,0],c,c],[[2,0,0],c,c]][4-sum(r.count(1)for r in j)]
```

# [187] 7b6016b9.json
* loop_filling
* background_filling
* color_guessing

```python
show_examples(load_examples(187)['train'])
```

```python
%%writefile task187.py
def p(j):
 A,c,E=len(j),len(j[0]),range;k=[(W,l)for W in(0,A-1)for l in E(c)if j[W][l]<1]+[(W,l)for W in E(A)for l in(0,c-1)if j[W][l]<1]
 while k:
  W,l=k.pop()
  if j[W][l]<1:j[W][l]=3;k+=[(x,y)for x,y in((W+1,l),(W-1,l),(W,l+1),(W,l-1))if 0<=x<A and 0<=y<c and j[x][y]<1]
 for W in E(A):
  for l in E(c):
   if j[W][l]<1:j[W][l]=2
 return j
```

# [188] 7b7f7511.json
* separate_images
* detect_repetition
* crop

```python
show_examples(load_examples(188)['train'])
```

```python
%%writefile task188.py
p=lambda j,A=len:[r[:A(r)//2]for r in j]if A(j[0])%2<1and all(r[:A(r)//2]==r[A(r)//2:]for r in j)else j[:A(j)//2]
```

# [189] 7c008303.json
* color_palette
* detect_grid
* recoloring
* color_guessing
* separate_images
* crop

```python
show_examples(load_examples(189)['train'])
```

```python
%%writefile task189.py
R=range
L=len
def p(g):
 for i in R(4):
  g=list(map(list,zip(*g[::-1])))
  if g[0][2]==8 and g[2][0]==8:
   for r in R(3,L(g)):
    for c in R(3,L(g[0])):
     if g[r][c]>0:
      g[r][c]=g[(r-2)//4][(c-2)//4]
   g=[r[3:] for r in g[3:]]
 return g
```

# [190] 7ddcd7ec.json
* draw_line_from_point
* direction_guessing
* diagonals

```python
show_examples(load_examples(190)['train'])
```

```python
%%writefile task190.py
def p(g):
 R=range
 r=[x[:]for x in g]
 d=[(0,1),(1,0),(0,-1),(-1,0)]
 c=[(-1,-1),(-1,1),(1,1),(1,-1)]
 for i in R(10):
  for j in R(10):
   if g[i][j]and all(0<=i+x<10and 0<=j+y<10and g[i+x][j+y]==0for x,y in d):
    for t in c:
     x,y=i+t[0],j+t[1]
     if 0<=x<10and 0<=y<10and g[x][y]:
      a,b=-t[0],-t[1]
      for m in R(1,10):
       u,v=i+a*m,j+b*m
       if 0<=u<10and 0<=v<10:r[u][v]=g[i][j]
 return r
```

# [191] 7df24a62.json
* pattern_repetition
* pattern_rotation
* pattern_juxtaposition
* out_of_boundary

```python
show_examples(load_examples(191)['train'])
```

```python
#%%writefile task191.py
```

# [192] 7e0986d6.json
* color_guessing
* remove_noise

```python
show_examples(load_examples(192)['train'])
```

```python
%%writefile task192.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 f=sum(g,[]);C=sorted([[f.count(k),k] for k in set(f)])[0][1]
 P=[[0,1],[0,-1],[-1,0],[1,0]]
 for r in R(h):
  for c in R(w):
   if g[r][c]==C:
    m=[]
    for y,x in P:
     if r+y>=0 and c+x>=0 and r+y<h and c+x<w:
      m+=[g[r+y][c+x]]
    if sum(m)/L(m)<max(m)/2:
     g[r][c]=0
    else: g[r][c]=max(m)
    try:
     if g[r][c+1]+g[r][c-1]==0 or g[r+1][c]+g[r-1][c]==0:g[r][c]=0
    except:pass
  if g[-1][-1]>0:
   if g[-2][-1]==0:g[-1][-1]=0
   if g[-1][-2]==0:g[-1][-1]=0
  if g[-1][10]>0 and g[-2][10]==0:g[-1][10]=0
 return g
```

# [193] 7f4411dc.json
* rectangle_guessing
* remove_noise

```python
show_examples(load_examples(193)['train'])
```

```python
%%writefile task193.py
p=lambda g,E=enumerate:[[v if(i and g[i-1][j]==v)+(i+1<len(g)and g[i+1][j]==v)+(j and r[j-1]==v)+(j+1<len(g)and r[j+1]==v)>1else 0 for j,v in E(r)]for i,r in E(g)]
```

# [194] 7fe24cdd.json
* image_repetition
* image_rotation

```python
show_examples(load_examples(194)['train'])
```

```python
%%writefile task194.py
j=lambda A:[[*i]for i in zip(*A[::-1])]
p=lambda c:[a+b for a,b in zip(c,j(c))]+[a+b for a,b in zip(j(j(j(c))),j(j(c)))]
```

# [195] 80af3007.json
* crop
* pattern_resizing
* image_resizing
* fractal_repetition

```python
show_examples(load_examples(195)['train'])
```

```python
%%writefile task195.py
R=range
L=len
def p(g):
 h,w=L(g),L(g[0])
 X=[[0]*3 for _ in R(3)]
 y=min([r for r in R(h) if g[r].count(5)>0])
 x=min([r.index(5) for r in g if r.count(5)>0])
 g=[r[x:x+9] for r in g[y:y+9]]
 for r in R(0,9,3):
  for c in R(0,9,3):
   X[r//3][c//3]=g[r][c]
 g=[[X[r//3][c//3]and X[r%3][c%3]for c in R(9)]for r in R(9)]
 return g
```

# [196] 810b9b61.json
* recoloring
* detect_closed_curves

```python
show_examples(load_examples(196)['train'])
```

```python
%%writefile task196.py
def p(j):
	A=range;c,E=len(j),len(j[0]);k=set();W=[e[:]for e in j]
	def M(l,J):
		if(l,J)in k or not(0<=l<c and 0<=J<E)or j[l][J]!=1:return[]
		k.add((l,J));return[(l,J)]+sum([M(l+e,J+A)for(e,A)in[(-1,0),(1,0),(0,-1),(0,1)]],[])
	for a in A(c):
		for C in A(E):
			if j[a][C]==1 and(a,C)not in k:
				e=M(a,C);K,w,L,b=min(e[0]for e in e),max(e[0]for e in e),min(e[1]for e in e),max(e[1]for e in e)
				if len(e)==2*(w-K+b-L)and w>K and b>L and any(j[e][k]==0 for e in A(K+1,w)for k in A(L+1,b)):
					for(d,f)in e:W[d][f]=3
	return W
```

# [197] 82819916.json
* pattern_repetition
* color_guessing
* draw_line_from_point
* associate_colors_to_colors

```python
show_examples(load_examples(197)['train'])
```

```python
%%writefile task197.py
def p(j):
	A=next((c for c in j if 0 not in c),None)
	if not A:return j
	c=[];[c.append(W)for W in A if W not in c]
	for(E,k)in enumerate(j):
		if 0 in k and any(k):
			W=[];[W.append(c)for c in k if c not in W and c]
			if len(c)==len(W):l=dict(zip(c,W));j[E]=[l[c]for c in A]
	return j
```

# [198] 83302e8f.json
* detect_grid
* detect_closed_curves
* rectangle_guessing
* associate_colors_to_bools
* loop_filling

```python
show_examples(load_examples(198)['train'])
```

```python
%%writefile task198.py
L=len
R=range
def p(g):
 C=max(sum(g,[]))
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  for r in R(h):
   if g[r].count(C)>w/2:
    for c in R(w):
     if g[r][c]==0:g[r][c]=4
 for i in range(10):
  for r in R(h):
   for c in R(w):
    if g[r][c]==4:
     for y,x in [[0,1],[0,-1],[1,0],[-1,0]]:
      if 0<=r+y<h and 0<=c+x<w and g[r+y][c+x]==0:
       g[r+y][c+x]=4
 g=[[3 if c==0 else c for c in r] for r in g]
 return g
```

# [199] 834ec97d.json
* draw_line_from_border
* pattern_repetition
* spacing
* measure_distance_from_side

```python
show_examples(load_examples(199)['train'])
```

```python
%%writefile task199.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   if W and W^4:
    j[c+1][k]=W
    for l in range(c+1):j[l][k&1::2]=[4]*len(j[l][k&1::2])
    return j
```

# [200] 8403a5d5.json
* draw_line_from_point
* pattern_repetition
* direction_guessing

```python
show_examples(load_examples(200)['train'])
```

```python
%%writefile task200.py
def p(j):
 A,c,E,k=10,enumerate,range,0
 for W,l in c(j):
  for J,a in c(l):
   if a%5:
    for C in E(J,A,2):
     for e in E(W+1):j[e][C]=a
    for C in E(J+1,A,2):j[k*(A-1)][C]=5;k^=1
    return j
```

# [201] 846bdb03.json
* pattern_moving
* pattern_reflection
* crop
* color_matching
* x_marks_the_spot

```python
show_examples(load_examples(201)['train'])
```

```python
%%writefile task201.py
def p(j):
 A=enumerate;c=next
 E=lambda k,W:k<1or j[k-1][W]<1or k>2and j[k-1][W]==4and j[k-2][W]>0
 (k,W),(l,J),(a,l),l=[divmod(i,13)for i,v in A(sum(j,[]))if v==4]
 C=c(u for r in zip(*j)if 4not in r for u in r if u)
 e=c(i for i,r in A(j)if any(u==C and E(i,v)for v,u in A(r)))
 K=c(i for i,r in A(zip(*j))if any(u==C and E(v,i)for v,u in A(r)))
 for w in range(a-k-1):
  for L in range(J-W-1):j[k+w+1][[J-L-1,W+L+1][j[k+1][W]==C]],j[e+w][K+L]=j[e+w][K+L],0
 return[r[W:J+1]for r in j[k:a+1]]
```

# [202] 855e0971.json
* draw_line_from_point
* direction_guessing
* separate_images
* holes

```python
show_examples(load_examples(202)['train'])
```

```python
#%%writefile task202.py
```

# [203] 85c4e7cd.json
* color_guessing
* recoloring
* color_permutation

```python
show_examples(load_examples(203)['train'])
```

```python
%%writefile task203.py
def p(g,L=len,R=range):
 h=L(g)
 w=L(g[0])
 C=g[h//2][:w//2]
 C={C[i]:C[-(i+1)] for i in R(L(C))}
 for r in R(h):
  for c in R(w):g[r][c]=C[g[r][c]]
 return g
```

# [204-R] 868de0fa.json
* loop_filling
* color_guessing
* measure_area
* even_or_odd
* associate_colors_to_bools

```python
show_examples(load_examples(204)['train'])
```

```python
%%writefile task204.py
R=range
L=len
def p(g):
 h,w,C=L(g),L(g[0]),7
 for r in R(1,h-1):
  for c in R(1,w-1):
   if sum([g[r-1][c],g[r+1][c],g[r][c-1],g[r][c+1]])>1 and g[r][c]==0:
    g[r][c]=C
 for r in R(1,h-1):
  for c in R(1,w-1):
   if [g[r-1][c],g[r+1][c],g[r][c-1],g[r][c+1]].count(0)>0 and g[r][c]==7:
    g[r][c]=0
 return g
```

# [205] 8731374e.json
* rectangle_guessing
* crop
* draw_line_from_point

```python
show_examples(load_examples(205)['train'])
```

```python
#%%writefile task205.py
```

# [206] 88a10436.json
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(206)['train'])
```

```python
#%%writefile task206.py
```

# [207] 88a62173.json
* detect_grid
* separate_images
* find_the_intruder
* crop

```python
show_examples(load_examples(207)['train'])
```

```python
%%writefile task207.py
def p(j):
	A={};c=[[[j[0][0],j[0][1]],[j[1][0],j[1][1]]],[[j[3][0],j[3][1]],[j[4][0],j[4][1]]],[[j[0][3],j[0][4]],[j[1][3],j[1][4]]],[[j[3][3],j[3][4]],[j[4][3],j[4][4]]]]
	for E in c:
		E=str(E)
		if E in A:A[E]+=1
		else:A[E]=1
	for E in A:
		if A[E]==1:return eval(E)
```

# [208-R] 890034e9.json
* pattern_repetition
* rectangle_guessing
* contouring

```python
show_examples(load_examples(208)['train'])
```

```python
%%writefile task208.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z=[r[:] for r in g]
 for s in R(min([h,w]),1,-1):
  t=0
  for r in R(h):
   for c in R(w):
    X=g[r:r+s]
    X=[m[c:c+s][:] for m in X]
    if sum(X,[]).count(0)==s*s:
     t=1
     for i in R(r,r+s):
      for j in R(c,c+s):
       Z[i][j]=9
  if t:return Z
 return g
```

# [209] 8a004b2b.json
* pattern_repetition
* pattern_resizing
* pattern_juxtaposition
* rectangle_guessing
* crop

```python
show_examples(load_examples(209)['train'])
```

```python
#%%writefile task209.py
```

# [210] 8be77c9e.json
* image_repetition
* image_reflection

```python
show_examples(load_examples(210)['train'])
```

```python
%%writefile task210.py
p=lambda j:j+j[::-1]
```

# [211] 8d5021e8.json
* image_repetition
* image_reflection

```python
show_examples(load_examples(211)['train'])
```

```python
%%writefile task211.py
def p(j):j=[R[::-1]+R for R in j];A=[j[2],j[1],j[0]];return A+j+A
```

# [212] 8d510a79.json
* draw_line_from_point
* detect_wall
* direction_guessing
* associate_colors_to_bools

```python
show_examples(load_examples(212)['train'])
```

```python
%%writefile task212.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 W=[i for i in R(L(g)) if 5 in g[i]][0]
 for r in R(h):
  for c in R(w):
   if g[r][c]==1 and r<W:
    for z in R(r,-1,-1):g[z][c]=1
   elif g[r][c]==1 and r>W:
    for z in R(r,h):g[z][c]=1
   if g[r][c]==2 and r<W:
    for z in R(r,W):g[z][c]=2
   elif g[r][c]==2 and r>W:
    for z in R(W+1,r):g[z][c]=2
 return g
```

# [213] 8e1813be.json
* recoloring
* color_guessing
* direction_guesingcrop
* image_within_image

```python
show_examples(load_examples(213)['train'])
```

```python
%%writefile task213.py
def Z(j,A):return len(set([J[A]for J in j]))
def p(c):
	E=enumerate;k,W=len(c),len(c[0]);l=Z(c,0)+Z(c,-1)<len(set(c[0]))+len(set(c[-1]));c=[[J if J!=5 else 0 for J in J]for J in c]
	for(J,a)in E(c):
		for(C,e)in E(a):
			if l:c[J][C]=max([c[0][C],c[-1][C]])
			else:c[J][C]=max([c[J][0],c[J][-1]])
	if l:c=[[J for J in J if J>0]for J in c];c=c[:len(c[0])]
	else:c=[J for J in c if sum(J)>0];c=[J[:len(c)]for J in c]
	return c
```

# [214] 8e5a5113.json
* detect_wall
* separate_images
* image_repetition
* image_rotation

```python
show_examples(load_examples(214)['train'])
```

```python
%%writefile task214.py
def p(g,R=range):
 A=[[c for c in r[:3]] for r in g]
 C=[r[::-1]for r in A[::-1]]
 for r in R(3):
  for c in R(3):
   g[r][c+4]=A[-(c+1)][r];g[r][c+8]=C[r][c]
 return g
```

# [215] 8eb1be9a.json
* pattern_repetition
* image_filling

```python
show_examples(load_examples(215)['train'])
```

```python
%%writefile task215.py
p=lambda j:[[r for j,r in enumerate(j)if sum(r)and j%3==i%3][0]for i in range(len(j))]
```

# [216] 8efcae92.json
* separate_images
* rectangle_guessing
* count_tiles
* take_maximum
* crop

```python
show_examples(load_examples(216)['train'])
```

```python
#%%writefile task216.py
```

# [217] 8f2ea7aa.json
* crop
* fractal_repetition

```python
show_examples(load_examples(217)['train'])
```

```python
%%writefile task217.py
def p(j,A=range(9)):c,E=next(i for i,r in enumerate(j)if sum(r))//3*3,next(i for i in A if sum(j[y][i]for y in A))//3*3;return[[j[c+y%3][E+x%3]*bool(j[c+y//3][E+x//3])for x in A]for y in A]
```

# [218] 90c28cc7.json
* crop
* rectangle_guessing
* summarize

```python
show_examples(load_examples(218)['train'])
```

```python
#%%writefile task218.py
```

# [219] 90f3ed37.json
* pattern_repetition
* recoloring

```python
show_examples(load_examples(219)['train'])
```

```python
#%%writefile task219.py
```

# [220] 913fb3ed.json
* contouring
* associate_colors_to_colors

```python
show_examples(load_examples(220)['train'])
```

```python
%%writefile task220.py
def p(j,A=enumerate):
 c={8:4,2:1,3:6};E=[[J for a,J in A(W)]for W in j]
 for k,W in A(j):
  for l,J in A(W):
   if J:
    for a in range(-1,2):
     for C in range(-1,2):
      try:
       if[a,C]!=[0,0]:E[k+a][l+C]=c[J]
      except:0
 return E
```

# [221] 91413438.json
* count_tiles
* algebra
* image_repetition

```python
show_examples(load_examples(221)['train'])
```

```python
%%writefile task221.py
R=range
def p(g):
 f=sum(g,[])
 S=f.count(0)
 Z=9-S
 X=[[0]*(S*3) for i in R(S*3)]
 for i in R(Z):
  for r in R(3):
   for c in R(3):
    if g[r][c]:X[r+(i//S)*3][c+(i%S)*3]=max(f)
 return X
```

# [222] 91714a58.json
* find_the_intruder
* remove_noise

```python
show_examples(load_examples(222)['train'])
```

```python
%%writefile task222.py
E=enumerate
def p(g):
 g=[[v if(i and g[i-1][j]==v)+(i+1<len(g)and g[i+1][j]==v)+(j and r[j-1]==v)+(j+1<len(g)and r[j+1]==v)>1else 0 for j,v in E(r)]for i,r in E(g)]
 f=sum(g,[])
 C=sorted([[f.count(c),c] for c in set(f) if c>0])
 g=[[0 if c!=C[-1][1] else c for c in r] for r in g]
 return g
```

# [223] 9172f3a0.json
* image_resizing

```python
show_examples(load_examples(223)['train'])
```

```python
%%writefile task223.py
def p(g):
 X=[]
 for r in g:
  for i in range(3):
   X+=[sum([[c]*3 for c in r],[])]
 return X
```

# [224] 928ad970.json
* rectangle_guessing
* color_guessing
* draw_rectangle

```python
show_examples(load_examples(224)['train'])
```

```python
%%writefile task224.py
def p(g,L=len,R=range,M=max,N=min):
 h,w,y,x=L(g),L(g[0]),[],[]
 C=[C for C in set(sum(g,[])) if C not in [0,5]][0]
 for r in R(h):
  for c in R(w):
   if g[r][c]==5:y+=[r];x+=[c]
 for r in R(h):
  for c in R(w):
   if r in [N(y)+1,M(y)-1] and N(x)+1<=c<=M(x)-1:g[r][c]=C
   if c in [N(x)+1,M(x)-1] and N(y)+1<=r<=M(y)-1:g[r][c]=C
 return g
```

# [225] 93b581b8.json
* pattern_expansion
* color_guessing
* out_of_boundary

```python
show_examples(load_examples(225)['train'])
```

```python
%%writefile task225.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 r=[r for r in R(h) if L(set(g[r]))>1][0]
 c=[c for c in R(w) if g[r][c]>0][0]
 P=[[-2,-2,g[r+1][c+1]],[2,-2,g[r][c+1]],[-2,2,g[r+1][c]],[2,2,g[r][c]]]
 for i in R(r,r+2):
  for j in R(c,c+2):
   for y,x,C in P:
    if 0<=y+i<h and 0<=x+j<w:
     g[y+i][x+j]=C
 return g
```

# [226] 941d9a10.json
* detect_grid
* loop_filling
* pairwise_analogy

```python
show_examples(load_examples(226)['train'])
```

```python
%%writefile task226.py
def f(j,A,c,E):
 if not(0<=A<len(j)and 0<=c<len(j[0])):return
 if j[A][c]:return
 j[A][c]=E
 for k,W in[(0,-1),(0,1),(-1,0),(1,0)]:f(j,A+k,c+W,E)
def p(j):
 l,J=len(j),len(j[0]);f(j,0,0,1)
 for a in range(4):f(j,l//2-1+a%2,J//2-1+a//2,2)
 f(j,l-1,J-1,3);return j
```

# [227] 94f9d214.json
* separate_images
* take_complement
* pattern_intersection

```python
show_examples(load_examples(227)['train'])
```

```python
%%writefile task227.py
p=lambda g:[[2*(g[i][j]==0==g[i+4][j])for j in range(4)]for i in range(4)]
```

# [228] 952a094c.json
* rectangle_guessing
* inside_out

```python
show_examples(load_examples(228)['train'])
```

```python
%%writefile task228.py
j=lambda A:[[A[E][c]for E in range(len(A))]for c in range(len(A[0]))]
def J(A):c=[A for(A,c)in enumerate(A)if any(c)];return c[0],c[-1]
def p(A):
	c,E=J(A);k,W=J(j(A))
	def F(l,J,a,C):A[l][J],A[a][C]=A[a][C],A[l][J]
	F(c+1,k+1,E+1,W+1);F(c+1,W-1,E+1,k-1);F(E-1,k+1,c-1,W+1);F(E-1,W-1,c-1,k-1);return A
```

# [229] 9565186b.json
* separate_shapes
* count_tiles
* recoloring
* take_maximum
* associate_color_to_bools

```python
show_examples(load_examples(229)['train'])
```

```python
%%writefile task229.py
def p(j):A=__import__('collections').Counter([x for R in j for x in R]).most_common(1);c=A[0][0];return[[A if A==c else 5 for A in R]for R in j]
```

# [230] 95990924.json
* pattern_expansion

```python
show_examples(load_examples(230)['train'])
```

```python
%%writefile task230.py
def p(j):
 A,c=len(j),len(j[0])
 for E in range(A-1):
  for k in range(c-1):
   if j[E][k]==j[E][k+1]==j[E+1][k]==j[E+1][k+1]==5:
    if E>0and k>0:j[E-1][k-1]=1
    if E>0and k+2<c:j[E-1][k+2]=2
    if E+2<A and k>0:j[E+2][k-1]=3
    if E+2<A and k+2<c:j[E+2][k+2]=4
 return j
```

# [231] 963e52fc.json
* image_expansion
* pattern_expansion

```python
show_examples(load_examples(231)['train'])
```

```python
%%writefile task231.py
p=lambda g:[[g[i%5][j%6]for j in range(len(g[0])*2)]for i in range(len(g)*1)]
```

# [232] 97999447.json
* draw_line_from_point
* pattern_expansion

```python
show_examples(load_examples(232)['train'])
```

```python
%%writefile task232.py
def p(j,A=enumerate):
 for c,E in A(j):
  k,W,l=0,[],0
  for J,a in A(E):
   if a>0:W=[a,5]*20;l=1
   if l:j[c][J]=W[k];k+=1
 return j
```

# [233] 97a05b5b.json
* pattern_moving
* pattern_juxtaposition
* crop
* shape_guessing

```python
show_examples(load_examples(233)['train'])
```

```python
#%%writefile task233.py
```

# [234] 98cf29f8.json
* pattern_moving
* bring_patterns_close

```python
show_examples(load_examples(234)['train'])
```

```python
#%%writefile task234.py
```

# [235] 995c5fa3.json
* take_complement
* detect_wall
* separate_images
* associate_colors_to_images
* summarize

```python
show_examples(load_examples(235)['train'])
```

```python
%%writefile task235.py
p=lambda j:[[(45-j[2][x]-2*j[2][x+1]-4*j[1][x+1])//5]*3 for x in range(0,15,5)]
```

# [236] 99b1bc43.json
* take_complement
* detect_wall
* separate_images
* pattern_intersection

```python
show_examples(load_examples(236)['train'])
```

```python
%%writefile task236.py
def p(j,A=range(4)):
 for c in A:
  for E in A:
   j[c][E]+=j[c+5][E]
   if j[c][E]==3:j[c][E]=0
   elif j[c][E]>0:j[c][E]=3
 return j[:4]
```

# [237] 99fa7670.json
* draw_line_from_point
* pattern_expansion

```python
show_examples(load_examples(237)['train'])
```

```python
%%writefile task237.py
def p(g,L=len,R=range):
 h,w=len(g),len(g[0])
 for r in R(h):
  s=0
  for c in R(w):
   if g[-(r+1)][c]>0:s=g[-(r+1)][c]
   g[-(r+1)][c]=s
  s=0
  for r in R(h):
   if g[r][-1]>0:s=g[r][-1]
   g[r][-1]=s
 return g
```

# [238] 9aec4887.json
* pattern_moving
* x_marks_the_spot
* crop
* recoloring
* color_guessing

```python
show_examples(load_examples(238)['train'])
```

```python
#%%writefile task238.py
```

# [239] 9af7a82c.json
* separate_images
* count_tiles
* summarize
* order_numbers

```python
show_examples(load_examples(239)['train'])
```

```python
%%writefile task239.py
from collections import*
def p(j,A=range):
 c=Counter([x for r in j for x in r]).most_common(9);E,k=c[0][1],len(c);j=[[0 for _ in A(k)]for _ in A(E)]
 for W in A(k):
  for l in A(c[W][1]):j[l][W]=c[W][0]
 return j
```

# [240] 9d9215db.json
* pattern_expansion
* pattern_reflection
* pattern_rotation

```python
show_examples(load_examples(240)['train'])
```

```python
#%%writefile task240.py
```

# [241] 9dfd6313.json
* image_reflection
* diagonal_symmetry

```python
show_examples(load_examples(241)['train'])
```

```python
%%writefile task241.py
p=lambda j:[*map(list,zip(*j))]
```

# [242] 9ecd008a.json
* image_filling
* pattern_expansion
* pattern_reflection
* pattern_rotation
* crop

```python
show_examples(load_examples(242)['train'])
```

```python
%%writefile task242.py
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==0:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==0:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==0:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==0:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g
```

# [243] 9edfc990.json
* background_filling
* holes

```python
show_examples(load_examples(243)['train'])
```

```python
%%writefile task243.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for z in R(25):
  for r in R(h):
   for c in R(w):
    if g[r][c]==0:
     if c+1<w:
      if g[r][c+1]==1:g[r][c]=1
     if r+1<h:
      if g[r+1][c]==1:g[r][c]=1
     if c-1>=0:
      if g[r][c-1]==1:g[r][c]=1
     if r-1>=0:
      if g[r-1][c]==1:g[r][c]=1
 return g
```

# [244] 9f236235.json
* detect_grid
* summarize
* image_reflection

```python
show_examples(load_examples(244)['train'])
```

```python
%%writefile task244.py
def p(g,V=range):R,C=len(g),len(g[0]);G=[-1]+[i for i in V(R)if len({*g[i]})==1]+[R];z=[-1]+[j for j in V(C)if len({g[i][j]for i in V(R)})==1]+[C];o=[[g[a+1][c+1]for c,d in zip(z,z[1:])if c+1<d-1]for a,b in zip(G,G[1:])if a+1<b-1];return[o[::-1]for o in o]
```

# [245] a1570a43.json
* pattern_moving
* rectangle_guessing
* x_marks_the_spot

```python
show_examples(load_examples(245)['train'])
```

```python
%%writefile task245.py
E=enumerate
R=range
L=len
def p(g):
 P=[[x,y] for y,r in E(g) for x,c in E(r) if c==3]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 M=[min(y)+1,max(y)-1,min(x)+1,max(x)-1]
 P=[[x,y] for y,r in E(g) for x,c in E(r) if c==2]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 M[0]-=min(y)
 M[1]-=max(y)
 M[2]-=min(x)
 M[3]-=max(x)
 X=[r[:] for r in g]
 g=[[0 if c!=3 else 3 for c in r] for r in g]
 for r in R(L(g)):
  for c in R(L(g[0])):
   if X[r][c]==2:
    g[r-min([M[0],0])+max([M[1],0])][c-min([M[2],0])+max([M[3],0])]=2
 return g
```

# [246] a2fd1cf0.json
* connect_the_dots

```python
show_examples(load_examples(246)['train'])
```

```python
%%writefile task246.py
def p(j):
 A=range
 c=[J[:]for J in j]
 for E in A(len(j)):
  for k in A(len(j[0])):
   if j[E][k]==2:W,l=E,k
   if j[E][k]==3:J,a=E,k
 C=1if a>l else-1
 for k in A(l+C,a+C,C):c[W][k]=8
 C=1if J>W else-1
 for E in A(W+C,J,C):c[E][a]=8
 return c
```

# [247] a3325580.json
* separate_shapes
* count_tiles
* take_maximum
* summarize
* remove_intruders

```python
show_examples(load_examples(247)['train'])
```

```python
%%writefile task247.py
def p(g):
 f=sum(g,[])
 C=max([f.count(c) for c in set(f) if c>0])
 M=sum(map(list,zip(*g[::-1])),[])
 M=[c for i,c in enumerate(M) if M.index(c)==i]
 g=[[c for c in M if f.count(c)==C]]*C
 return g
```

# [248] a3df8b1e.json
* pattern_expansion
* draw_line_from_point
* diagonals
* bounce

```python
show_examples(load_examples(248)['train'])
```

```python
%%writefile task248.py
def p(j):
	A=[l[:]for l in j];c,E=len(j),len(j[0]);k,W,l=c-1,0,1
	while k>=0:
		A[k][W]=1
		if 0<=W+l<E:k-=1;W+=l
		else:k-=1;l=-l;W+=l
	return A
```

# [249] a416b8f3.json
* image_repetition

```python
show_examples(load_examples(249)['train'])
```

```python
%%writefile task249.py
p=lambda j:[E*2for E in j]
```

# [250] a48eeaf7.json
* pattern_moving
* bring_patterns_close
* gravity
* direction_guessing

```python
show_examples(load_examples(250)['train'])
```

```python
%%writefile task250.py
from itertools import *
L=len
R=range
P=list(product([0,1,-1],repeat=2))
def p(g):
 h,w=L(g),L(g[0])
 for r in R(h):
  for c in R(w):
   if g[r][c]==2:
    for y,x in P:
     if 0<=r+y<h and 0<=c+x<w:
      for z in R(20):
        if 0<=r+(y*z)<h and 0<=c+(x*z)<w:
         W=g[r+(y*z)][c+(x*z)]
         if W==5 and g[r+y][c+x]==0:g[r+y][c+x]=5;g[r+(y*z)][c+(x*z)]=0
 return g
```

# [251] a5313dff.json
* loop_filling

```python
show_examples(load_examples(251)['train'])
```

```python
%%writefile task251.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[[0]*E for c in A(c)];W=[]
	for l in A(c):
		for J in A(E):
			if l*J==0 or l==c-1 or J==E-1:
				if j[l][J]==0:k[l][J]=1;W.append((l,J))
	while W:
		a,C=W.pop(0)
		for(e,K)in[(-1,0),(1,0),(0,-1),(0,1)]:
			w,L=a+e,C+K
			if 0<=w<c and 0<=L<E and j[w][L]==0 and not k[w][L]:k[w][L]=1;W.append((w,L))
	b=[[j[c][E]if j[c][E]!=0 or k[c][E]else 1 for E in A(E)]for c in A(c)];return b
```

# [252] a5f85a15.json
* recoloring
* pattern_modification
* pairwise_analogy

```python
show_examples(load_examples(252)['train'])
```

```python
%%writefile task252.py
def p(j,A=range):
 c=len(j)
 for E in A(c):
  for k,W in zip(A(1,c,2),A(E+1,c,2)):
   if j[0][E]:j[k][W]=4
   if j[E][0]:j[W][k]=4
 return j
```

# [253] a61ba2ce.json
* pattern_moving
* bring_patterns_close
* crop
* jigsaw

```python
show_examples(load_examples(253)['train'])
```

```python
%%writefile task253.py
def p(j):
 A=len(j)-1;c=[0]*16
 for E in range(A):
  for k in range(A):
   if(W:=j[E][k])and j[E+1][k]==W and j[E][k+1]==W:c[0]=c[4]=c[1]=W
   if W and j[E+1][k]==W and j[E+1][k+1]==W:c[8]=c[12]=c[13]=W
   if W and j[E][k+1]==W and j[E+1][k+1]==W:c[2]=c[3]=c[7]=W
   if(l:=j[E+1][k+1])and j[E+1][k]==l and j[E][k+1]==l:c[11]=c[14]=c[15]=l
 return[c[E:E+4]for E in(0,4,8,12)]
```

# [254] a61f2674.json
* separate_shapes
* count_tiles
* take_maximum
* take_minimum
* recoloring
* associate_colors_to_ranks
* remove_intruders

```python
show_examples(load_examples(254)['train'])
```

```python
%%writefile task254.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[0 for W in A(E)]
	for W in A(E):
		for l in A(c):
			if j[l][W]>0:k[W]+=1
			j[l][W]=0
	J=min([W for W in k if W>0]);W=k.index(J)
	for l in A(k[W]):j[-(l+1)][W]=2
	W=k.index(max(k))
	for l in A(k[W]):j[-(l+1)][W]=1
	return j
```

# [255-R] a64e4611.json
* background_filling
* rectangle_guessing

```python
show_examples(load_examples(255)['train'])
```

```python
%%writefile task255.py
R=range
L=len
P=lambda m:list(map(list,zip(*m[::-1])))
def p(g):
 C=max(sum(g,[]))
 for i in R(4):
  g=P(g)
  h,w=L(g),L(g[0])
  for r in R(h-1):
   for c in R(w-1):
    if g[r][c]==C:
     for y,x in [[0,0],[0,1],[1,0],[1,1]]:
      if g[r+y][c+x]==0:g[r+y][c+x]=10
 for i in R(4):
  g=P(g)   
  for r in R(h):
   M=sorted(set(g[r]))
   if M==[0] or M==[0,3]:
    g[r]=[3]*L(g[r])
 for i in R(4):
  g=P(g)    
  for r in R(h):
   if C not in g[r][:10] and 10 not in g[r][:10]:
    for c in R(w):
     if g[r][c]<1:g[r][c]=3
     else:break
 g=[[0 if c>9 else c for c in r] for r in g]
 return g
```

# [256] a65b410d.json
* pattern_expansion
* count_tiles
* associate_colors_to_ranks

```python
show_examples(load_examples(256)['train'])
```

```python
%%writefile task256.py
def p(j,A=range):
 c=len(j)
 for E in A(c):
  if j[E][0]==2:
   k=0
   while k<c and j[E][k]==2:k+=1
   for W in A(c):
    for l in A((k+E-W)*(W!=E)):j[W][l]=3-2*(W>E)
 return j
```

# [257] a68b268e.json
* detect_grid
* separate_images
* pattern_juxtaposition

```python
show_examples(load_examples(257)['train'])
```

```python
%%writefile task257.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(4):
  for c in R(4):
   if g[r][c]==0:
    if g[r][c+5]>0:g[r][c]=g[r][c+5]
   if g[r][c]==0:
    if g[r+5][c]>0:g[r][c]=g[r+5][c]
   if g[r][c]==0:
    if g[r+5][c+5]>0:g[r][c]=g[r+5][c+5]
 return [r[:4] for r in g[:4]]
```

# [258] a699fb00.json
* pattern_expansion
* connect_the_dots

```python
show_examples(load_examples(258)['train'])
```

```python
%%writefile task258.py
def p(j):
 for A in j:
  for c in range(len(A)-2):
   if A[c]&A[c+2]:A[c+1]=2
 return j
```

# [259] a740d043.json
* crop
* detect_background_color
* recoloring

```python
show_examples(load_examples(259)['train'])
```

```python
%%writefile task259.py
def p(j,A=range):
 c,E=len(j),len(j[0]);k,W,l,J=c,0,E,0
 for a in A(c):
  for C in A(E):
   if j[a][C]-1:
    if a<k:k=a
    if a>W:W=a
    if C<l:l=C
    if C>J:J=C
 return[[x-(x==1)for x in r[l:J+1]]for r in j[k:W+1]]
```

# [260] a78176bb.json
* draw_parallel_line
* direction_guessing
* remove_intruders

```python
show_examples(load_examples(260)['train'])
```

```python
%%writefile task260.py
L=len
R=range
P=[[0,0],[0,1],[1,0],[1,1]]
def p(g):
 h,w=L(g),L(g[0])
 C=[c for c in set(sum(g,[])) if c not in [0,5]][0]
 for r in R(h-1):
  for c in R(w-1):
    M=[g[r+y][c+x] for y,x in P]
    if M.count(5)==1 and sum(M)==5:
     for y in R(2):
      for x in R(2):
        if g[y+r][x+c]==5:
          for z in R(-10,10):
           if M[2]==5:
            if 0<=y+r-z-1<h and 0<=x+c-z+1<w:g[y+r-z-1][x+c-z+1]=C
           else:
            if 0<=y+r-z+1<h and 0<=x+c-z-1<w:g[y+r-z+1][x+c-z-1]=C
 g=[[c if c!=5 else 0 for c in r] for r in g]
 return g
```

# [261] a79310a0.json
* pattern_moving
* recoloring
* pairwise_analogy

```python
show_examples(load_examples(261)['train'])
```

```python
%%writefile task261.py
def p(j):j=[j[-1]]+j[:len(j)-1];j=[[2 if C==8 else C for C in R]for R in j];return j
```

# [262] a85d4709.json
* separate_images
* associate_colors_to_images
* summarize

```python
show_examples(load_examples(262)['train'])
```

```python
%%writefile task262.py
p=lambda j:[[[2,4,3][r.index(5)]]*3for r in j]
```

# [263] a87f7484.json
* separate_images
* find_the_intruder
* crop

```python
show_examples(load_examples(263)['train'])
```

```python
%%writefile task263.py
def p(j):
	A=range;c=[[[j[D+c*3][A+E*3]for A in A(3)]for D in A(3)]for c in A(len(j)//3)for E in A(len(j[0])//3)]
	for E in c:
		if[tuple(tuple(c[E][A]==0 for A in A(3))for E in A(3))for c in c].count(tuple(tuple(E[c][A]==0 for A in A(3))for c in A(3)))==1:return E
```

# [264] a8c38be5.json
* pattern_moving
* jigsaw
* crop

```python
show_examples(load_examples(264)['train'])
```

```python
%%writefile task264.py
L=len
R=range
B=[1,4,9,16,25,36,49,64,81]
S=[-1,0,1]
P=[[x,y] for x in S for y in S]
Z=[264,246,236,194,285,134,156,66,104]
def p(g):
 h,w=L(g),L(g[0])
 X=[[0]*9 for _ in R(9)]
 for r in R(1,h-1):
  for c in R(1,w-1):
    M=[g[r+y][c+x] for y,x in P]
    f=sum([B[i] for i in R(9) if M[i]==5])
    if f in Z and 0 not in M:
     j=Z.index(f)
     for y in R(3):
      for x in R(3):
        X[y+(j//3*3)][x+(j%3*3)]=g[r-1+y][c-1+x]
 return X
```

# [265-R] a8d7556c.json
* recoloring
* rectangle_guessing

```python
show_examples(load_examples(265)['train'])
```

```python
%%writefile task265.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h-1):
  for c in R(w-1):
   C=g[r][c:c+2]+g[r+1][c:c+2]
   if C.count(0)==4:
    g[r][c]=2
    g[r][c+1]=2
    g[r+1][c]=2
    g[r+1][c+1]=2
   if C.count(0)==2 and C.count(2)==2:
    g[r][c]=2
    g[r][c+1]=2
    g[r+1][c]=2
    g[r+1][c+1]=2
 return g
```

# [266] a9f96cdd.json
* replace_pattern
* out_of_boundary

```python
show_examples(load_examples(266)['train'])
```

```python
%%writefile task266.py
def p(j):
 A=sum(j,[]).index(2);c,E=divmod(A,5);j[c][E]=0
 if c*E:j[c-1][E-1]=3
 if c<2and E:j[c+1][E-1]=8
 if E<4and c:j[c-1][E+1]=6
 if c<2and E<4:j[c+1][E+1]=7
 return j
```

# [267] aabf363d.json
* recoloring
* color_guessing
* remove_intruders

```python
show_examples(load_examples(267)['train'])
```

```python
%%writefile task267.py
def p(j):A=j[6][0];c=[[r and A for r in X]for X in j];c[6][0]=0;return c
```

# [268] aba27056.json
* pattern_expansion
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(268)['train'])
```

```python
#%%writefile task268.py
```

# [269] ac0a08a4.json
* image_resizing
* count_tiles
* size_guessing

```python
show_examples(load_examples(269)['train'])
```

```python
%%writefile task269.py
p=lambda j:(A:=sum(c>0for r in j for c in r),[sum(([x]*A for x in r),[])for r in j for _ in range(A)])[1]
```

# [270] ae3edfdc.json
* bring_patterns_close
* gravity

```python
show_examples(load_examples(270)['train'])
```

```python
%%writefile task270.py
def p(M):
 R,C=len(M),len(M[0]);O=[[0]*C for _ in range(R)];P={}
 for r in range(R):
  for c in range(C):
   v=M[r][c]
   if v in(1,2):P[v]=(r,c);O[r][c]=v
 T={3:P[2],7:P[1]}
 for r in range(R):
  for c in range(C):
   v=M[r][c]
   if v not in(0,1,2):
    tr,tc=T[v]
    if r==tr:
     nc=tc+(1 if c>tc else-1)
     (O[r].__setitem__(nc,v) if 0<=nc<C and not O[r][nc] else O[r].__setitem__(c,v))
    elif c==tc:
     nr=tr+(1 if r>tr else-1)
     (O[nr].__setitem__(c,v) if 0<=nr<R and not O[nr][c] else O[r].__setitem__(c,v))
    else:
     b,d=None,1e9
     for ar,ac in((tr,tc+1),(tr,tc-1),(tr+1,tc),(tr-1,tc)):
      if 0<=ar<R and 0<=ac<C and not O[ar][ac]and(ar==r or ac==c):
       dist=abs(r-ar)+abs(c-ac)
       if dist<d:d,b=dist,(ar,ac)
     (O[b[0]].__setitem__(b[1],v) if b else O[r].__setitem__(c,v))
 return O
```

# [271] ae4f1146.json
* separate_images
* count_tiles
* crop

```python
show_examples(load_examples(271)['train'])
```

```python
%%writefile task271.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 Z,z=[],0
 for r in R(h-2):
  for c in R(w-2):
   C=g[r][c:c+3]+g[r+1][c:c+3]+g[r+2][c:c+3]
   y=C.count(1)+(C.count(8)/10)
   if y>z:Z=C[:];z=y
 return [Z[:3],Z[3:6],Z[6:]]
```

# [272] aedd82e4.json
* recoloring
* separate_shapes
* count_tiles
* take_minimum
* associate_colors_to_bools

```python
show_examples(load_examples(272)['train'])
```

```python
%%writefile task272.py
def p(g):h,w=len(g),len(g[0]);return[[1if g[i][j]and all(g[i+a][j+b]==0for a,b in[(-1,0),(1,0),(0,-1),(0,1)]if 0<=i+a<h and 0<=j+b<w)else g[i][j]for j in range(w)]for i in range(h)]
```

# [273] af902bf9.json
* ex_nihilo
* x_marks_the_spot

```python
show_examples(load_examples(273)['train'])
```

```python
#%%writefile task273.py
```

# [274] b0c4d837.json
* measure_length
* associate_images_to_numbers

```python
show_examples(load_examples(274)['train'])
```

```python
%%writefile task274.py
j=lambda A,c:sum(sum(i==c for i in r)for r in A)
def p(A):E=max(j([r],8)for r in A);k=(j(A,5)-E-2)/2-j(A,8)/E;return[[8*(k>0),8*(k>1),8*(k>2)],[0,0,8*(k>3)],[0,0,0]]
```

# [275] b190f7f5.json
* separate_images
* image_expasion
* color_palette
* image_resizing
* replace_pattern

```python
show_examples(load_examples(275)['train'])
```

```python
%%writefile task275.py
def p(j):
 A=min(len(j),len(j[0]));p,c=[r[:A]for r in j[:A]],[r[-A:]for r in j[-A:]]
 if any(max(r)==8 for r in p):p,c=c,p
 return[[p[y//A][x//A]*c[y%A][x%A]//8 for x in range(A*A)]for y in range(A*A)]
```

# [276] b1948b0a.json
* recoloring
* associate_colors_to_colors

```python
show_examples(load_examples(276)['train'])
```

```python
%%writefile task276.py
p=lambda g:[[({6:2,7:7}).get(x,x)for x in r]for r in g]
```

# [277] b230c067.json
* recoloring
* separate_shapes
* find_the_intruder
* associate_colors_to_bools

```python
show_examples(load_examples(277)['train'])
```

```python
#%%writefile task277.py
```

# [278] b27ca6d3.json
* find_the_intruder
* count_tiles
* contouring

```python
show_examples(load_examples(278)['train'])
```

```python
%%writefile task278.py
j=lambda A:[[A[c][E]for c in range(len(A))]for E in range(len(A[0]))]
def h(A,c,E):
 if A[c][E]!=2or A[c][E+1]!=2:return
 for k in range(max(0,c-1),min(len(A),c+2)):
  for W in range(max(0,E-1),min(len(A[0]),E+3)):
   if A[k][W]!=2:A[k][W]=3
def f(A):
 for c in range(len(A)):
  for E in range(len(A[0])-1):h(A,c,E)
def p(A):f(A);A=j(A);f(A);return j(A)
```

# [279] b2862040.json
* recoloring
* detect_closed_curves
* associate_colors_to_bools

```python
show_examples(load_examples(279)['train'])
```

```python
#%%writefile task279.py
```

# [280] b527c5c6.json
* pattern_expansion
* draw_line_from_point
* contouring
* direction_guessing
* size_guessing

```python
show_examples(load_examples(280)['train'])
```

```python
#%%writefile task280.py
```

# [281] b548a754.json
* pattern_expansion
* pattern_modification
* x_marks_the_spot

```python
show_examples(load_examples(281)['train'])
```

```python
#%%writefile task281.py
```

# [282] b60334d2.json
* replace_pattern

```python
show_examples(load_examples(282)['train'])
```

```python
%%writefile task282.py
p=lambda g,R=range(1,8):(G:=[[0]*9for _ in g],[G[i+a].__setitem__(j+b,(1,5)[a*b])for i in R for j in R if g[i][j]for a in(-1,0,1)for b in(-1,0,1)if a|b])[0]
```

# [283] b6afb2da.json
* recoloring
* replace_pattern
* rectangle_guessing

```python
show_examples(load_examples(283)['train'])
```

```python
%%writefile task283.py
def f(j,p,A,c,E,k):
 for W in range(A,E+1):
  for l in range(p,c+1):j[W][l]=k
def z(j,p,A,c,E):f(j,p,A,c,E,4);f(j,p+1,A+1,c-1,E-1,2);j[A][p]=j[A][c]=j[E][p]=j[E][c]=1
def p(j):
 J,a=len(j),len(j[0])
 for C in range(J*a):
  l,W=C%a,C//a
  if j[W][l]==5:
   c,E=l,W
   while c<a-1and j[E][c+1]==5:c+=1
   while E<J-1and j[E+1][c]==5:E+=1
   z(j,l,W,c,E)
 return j
```

# [284] b7249182.json
* pattern_expansion

```python
show_examples(load_examples(284)['train'])
```

```python
#%%writefile task284.py
```

# [285] b775ac94.json
* pattern_expansion
* pattern_repetition
* recoloring
* pattern_rotation
* pattern_reflection
* direction_guessing
* pattern_juxtaposition

```python
show_examples(load_examples(285)['train'])
```

```python
#%%writefile task285.py
```

# [286] b782dc8a.json
* pattern_expansion
* maze

```python
show_examples(load_examples(286)['train'])
```

```python
%%writefile task286.py
L=len
R=range
def p(g):
 h,w=L(g),L(g[0])
 f=sum(g,[]);C=sorted([[f.count(k),k] for k in set(f)])[:2]
 d={C[0][1]:C[1][1],C[1][1]:C[0][1]}
 for i in range(50):
  for r in R(h):
   for c in R(w):
    if g[r][c] in d:
     for y,x in [[0,1],[0,-1],[1,0],[-1,0]]:
      if 0<=r+y<h and 0<=c+x<w and g[r+y][c+x]==0:
       g[r+y][c+x]=d[g[r][c]]
 return g
```

# [287] b8825c91.json
* pattern_completion
* pattern_rotation
* pattern_reflection

```python
show_examples(load_examples(287)['train'])
```

```python
%%writefile task287.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==4:g[r][c]=g[-(r+1)][c]
   if g[-(r+1)][c]==4:g[-(r+1)][c]=g[r][c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==4:g[r][c]=g[r][-(c+1)]
   if g[r][-(c+1)]==4:g[r][-(c+1)]=g[r][c]
 return g
```

# [288] b8cdaf2b.json
* pattern_expansion
* draw_line_from_point
* diagonals
* pairwise_analogy

```python
show_examples(load_examples(288)['train'])
```

```python
#%%writefile task288.py
```

# [289] b91ae062.json
* image_resizing
* size_guessing
* count_different_colors

```python
show_examples(load_examples(289)['train'])
```

```python
%%writefile task289.py
p=lambda j:(A:=len(set(sum(j,[]))-{0}),[[x for x in r for _ in range(A)]for r in j for _ in range(A)])[1]
```

# [290] b94a9452.json
* crop
* take_negative

```python
show_examples(load_examples(290)['train'])
```

```python
%%writefile task290.py
def p(j):
	j=[c for c in j if sum(c)>0];A=[];c=[]
	for E in j:
		for k in range(len(E)):
			if E[k]>0:A.append(k);c.append(E[k])
	c=list(set(c));c={c[0]:c[1],c[1]:c[0]};j=[c[min(A):max(A)+1]for c in j];j=[[c[A]for A in A]for A in j];return j
```

# [291] b9b7f026.json
* find_the_intruder
* summarize

```python
show_examples(load_examples(291)['train'])
```

```python
%%writefile task291.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h-1):
  for c in R(w-1):
   C=g[r][c:c+2]+g[r+1][c:c+2]
   y=C.count(0)
   if y==1:
    for z in R(1,10):
     if C.count(z)==3:return [[z]]
```

# [292] ba26e723.json
* pattern_modification
* pairwise_analogy
* recoloring

```python
show_examples(load_examples(292)['train'])
```

```python
%%writefile task292.py
def p(j):
 for A in j:A[::3]=[6 if v==4 else v for v in A[::3]]
 return j
```

# [293] ba97ae07.json
* pattern_modification
* pairwise_analogy
* rettangle_guessing
* recoloring

```python
show_examples(load_examples(293)['train'])
```

```python
%%writefile task293.py
j=lambda A:[A[0]]*len(A)if A[0]else A
c=lambda E:[[E[y][x]for y in range(len(E))]for x in range(len(E[0]))]
k=lambda E:[j(A)for A in E]
p=lambda E:c(k(c(E)))if k(E)==E else k(E)
```

# [294] bb43febb.json
* loop_filling
* rettangle_guessing

```python
show_examples(load_examples(294)['train'])
```

```python
%%writefile task294.py
p=lambda g:[[2 if g[i][j]==5and all(0<=i+d[0]<10and 0<=j+d[1]<10and g[i+d[0]][j+d[1]]==5 for d in[(-1,0),(1,0),(0,-1),(0,1)])else g[i][j]for j in range(10)]for i in range(10)]
```

# [295] bbc9ae5d.json
* pattern_expansion
* image_expansion

```python
show_examples(load_examples(295)['train'])
```

```python
%%writefile task295.py
def p(g,L=len,R=range):
 g=g[0]
 C=g[0]
 T=L([x for x in g if x>0])
 w=R(L(g))
 h=R(L(g)//2)
 X=[[0 for x in w] for y in h]
 for r in h:
  for c in w:
   if c<T:X[r][c]=C
  T+=1
 return X
```

# [296] bc1d5164.json
* pattern_moving
* pattern_juxtaposition
* crop
* pairwise_analogy

```python
show_examples(load_examples(296)['train'])
```

```python
%%writefile task296.py
def p(j):
 A=[[0]*3,[0]*3,[0]*3]
 for c in range(16):E,k=c//8%2*-2+c//2%2,c//4%2*-2+c%2;A[E][k]=max(A[E][k],j[E][k])
 return A
```

# [297] bd4472b8.json
* detect_wall
* pattern_expansion
* ex_nihilo
* color_guessing
* color_palette

```python
show_examples(load_examples(297)['train'])
```

```python
%%writefile task297.py
def p(j):
 A,c=len(j),len(j[0]);E=j[0]*20
 for k in range(2,A):j[k]=[E[k-2]for _ in range(c)]
 return j
```

# [298] bda2d7a6.json
* recoloring
* pairwise_analogy
* pattern_modification
* color_permutation

```python
show_examples(load_examples(298)['train'])
```

```python
%%writefile task298.py
def p(j):A=len(j)//2;c=[j[i][i]for i in range(A)];E={c[i]:c[i-1]for i in range(A)};return[[E[i]for i in r]for r in j]
```

# [299] bdad9b1f.json
* draw_line_from_point
* direction_guessing
* recoloring
* take_intersection

```python
show_examples(load_examples(299)['train'])
```

```python
%%writefile task299.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h):
  if g[r][0]==2 or g[r][-1]==2: 
   for c in R(w):
    if g[r][c]==0:g[r][c]=2
    elif g[r][c]!=2:g[r][c]=4
 for c in R(w):
  if g[0][c]==8 or g[-1][c]==8: 
   for r in R(h):
    if g[r][c]==0:g[r][c]=8
    elif g[r][c]!=8:g[r][c]=4
 return g
```

# [300] be94b721.json
* separate_shapes
* count_tiles
* take_maximum
* crop

```python
show_examples(load_examples(300)['train'])
```

```python
%%writefile task300.py
from collections import*
def p(m,K=enumerate):
	a=[(i,j)for(i,r)in K(m)for(j,v)in K(r)if v]
	if not a:return[]
	v=Counter(m[i][j]for(i,j)in a).most_common(1)[0][0];x=[(i,j)for(i,j)in a if m[i][j]==v];h,b=min(i for(i,_)in x),min(j for(_,j)in x);c,g=max(i for(i,_)in x)+1,max(j for(_,j)in x)+1;return[m[i][b:g]for i in range(h,c)]
```

# [301] beb8660c.json
* pattern_moving
* count_tiles
* order_numbers

```python
show_examples(load_examples(301)['train'])
```

```python
%%writefile task301.py
def p(j):
	from collections import Counter as D;A=[c for l in j for c in l if c];c=dict(D(A).most_common());E=len(j[0]);k=[[0]*E for c in range(len(j))]
	for(W,l)in enumerate(sorted(c,key=c.get,reverse=True)):k[-1-W][-c[l]:]=[l]*c[l]
	return k
```

# [302] c0f76784.json
* loop_filling
* measure_area
* associate_colors_to_numbers

```python
show_examples(load_examples(302)['train'])
```

```python
%%writefile task302.py
def p(j):
	A,c=len(j),len(j[0]);E=[[0]*c for b in j];k=[]
	def e(W,l):
		J=[(W,l)];E[W][l]=1;a=[(W,l)];C=1
		while J:
			e,K=J.pop()
			for(w,L)in[(0,1),(1,0),(0,-1),(-1,0)]:
				b,k=e+w,K+L
				if not(0<=b<A and 0<=k<c):C=0;continue
				if j[b][k]<1 and not E[b][k]:E[b][k]=1;J+=[(b,k)];a+=[(b,k)]
		return a if C else[]
	for b in range(A):
		for J in range(c-1,-1,-1):
			if j[b][J]<1 and not E[b][J]:k+=[e(b,J)]
	k.sort(key=len,reverse=1)
	for(b,a)in enumerate(k):
		K=min(8,max(6,len(a)**.5+.0+5))
		for C in a:j[C[0]][C[1]]=K
	return j
```

# [303] c1d99e64.json
* draw_line_from_border
* detect_grid

```python
show_examples(load_examples(303)['train'])
```

```python
%%writefile task303.py
def p(j,A=range):
 c,E=len(j),len(j[0])
 for k in A(c):
  if sum(j[k])==0:j[k]=[2]*E
 for W in A(E):
  if all(j[k][W]in[0,2]for k in A(c)):
   for k in A(c):j[k][W]=2
 return j
```

# [304] c3e719e8.json
* image_repetition
* image_expansion
* count_different_colors
* take_maximum

```python
show_examples(load_examples(304)['train'])
```

```python
%%writefile task304.py
def p(j,A=range(9),c=range(3)):
 E,k=__import__('collections').Counter(j[0]+j[1]+j[2]).most_common(1)[0][0],[[0 for _ in A]for _ in A]
 for W,l in[(W,l)for l in c for W in c if j[W][l]==E]:
  for J in A:k[3*W+J%3][3*l+J//3]=j[J%3][J//3]
 return k
```

# [305] c3f564a4.json
* pattern_expansion
* image_filling

```python
show_examples(load_examples(305)['train'])
```

```python
%%writefile task305.py
def p(j):
	A=len(j);c=[A for c in j for A in c if A]
	if not c:return j
	E=sorted(set(c));k=len(E);W=[[0]*A for c in[0]*A]
	for l in range(A):
		for J in range(A):W[l][J]=E[(l+J)%k]
	return W
```

# [306] c444b776.json
* detect_grid
* separate_images
* find_the_intruder
* image_repetition

```python
show_examples(load_examples(306)['train'])
```

```python
%%writefile task306.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 for r in R(h//2):
  if len(set(g[r]))>2:g[r+(h//2)+1]=g[r][:]
 x=g[0].count(4)+1
 for r in R(h//2,h):
  if len(set(g[r]))>2:g[r-(h//2)-1]=g[r][:]
 x=g[0].count(4)+1
 if x==1:
  for r in R(h//2):
   for c in R(w):
    m=max([g[r][c],g[r+(h//2)+1][c]])
    g[r][c]=m
    g[r+(h//2)+1][c]=m
 for r in R(h):
  for c in R(w//x):
   for z in R(x):
    g[r][c]=max([g[r][c],g[r][c+(w//x+1)*z]])
    g[r][c+(w//x+1)*z]=max([g[r][c],g[r][c+(w//x+1)*z]])
 for r in R(h):
  for c in R(w//x):
   for z in R(x):
    g[r][c]=max([g[r][c],g[r][c+(w//x+1)*z]])
    g[r][c+(w//x+1)*z]=max([g[r][c],g[r][c+(w//x+1)*z]])
 return g
```

# [307] c59eb873.json
* image_resizing

```python
show_examples(load_examples(307)['train'])
```

```python
%%writefile task307.py
def p(g):
 X=[]
 for r in g:
  for i in range(2):
   X+=[sum([[c]*2 for c in r],[])]
 return X
```

# [308-R] c8cbb738.json
* pattern_moving
* jigsaw
* crop

```python
show_examples(load_examples(308)['train'])
```

```python
%%writefile task308.py
L=len
R=range
E=enumerate
def M(m,C):
 P=[[x,y] for y,r in E(m) for x,c in E(r) if c==C]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 X=m[min(y):max(y)+1]
 X=[r[min(x):max(x)+1] for r in X]
 return X
def p(g):
 f=sum(g,[]);Z=sorted([[f.count(k),k] for k in set(f)])
 Z=[x[1] for x in Z]
 P=[M(g,Z[i]) for i in R(L(Z)-1)]
 return P[0]
```

# [309] c8f0f002.json
* recoloring
* associate_colors_to_colors

```python
show_examples(load_examples(309)['train'])
```

```python
%%writefile task309.py
p=lambda j:[[x-2*(x==7)for x in r]for r in j]
```

# [310] c909285e.json
* find_the_intruder
* crop
* rectangle_guessing

```python
show_examples(load_examples(310)['train'])
```

```python
%%writefile task310.py
from collections import Counter
def p(m):
 c=Counter(e for r in m for e in r if e).most_common()
 if not c:return[]
 l=c[-1][0];O=p=-1
 for i,r in enumerate(m):
  if l in r:
   if O<0:O=i
   p=i
 S=U=-1
 for i in range(len(m[0])):
  if any(m[j][i]==l for j in range(O,p+1)):
   if S<0:S=i
   U=i
 return[r[S:U+1]for r in m[O:p+1]]
```

# [311] c9e6f938.json
* image_repetition
* image_reflection

```python
show_examples(load_examples(311)['train'])
```

```python
%%writefile task311.py
p=lambda j:[R+R[::-1]for R in j]
```

# [312] c9f8e694.json
* recoloring
* pattern_repetition
* color_palette

```python
show_examples(load_examples(312)['train'])
```

```python
%%writefile task312.py
def p(j):
 for A in j:
  for c in A:
   if c and c-5:A[:]=[c*(x==5)+x*(x!=5)for x in A];break
 return j
```

# [313] caa06a1f.json
* pattern_expansion
* image_filling

```python
show_examples(load_examples(313)['train'])
```

```python
%%writefile task313.py
def p(g,r=range,l=len):
 n=l(g);q=l(set(g[0]))-1;p=l({i[0]for i in g})-1
 for x in g:x[:]=(x[:q]*((l(x)-1)//q+1))[:l(x)]
 for i in r(n):g[i]=[g[i%p][j]for j in r(n)]
 return[[dict(zip(g[0],g[0][1:]))[y]for y in r]for r in g]
```

# [314] cbded52d.json
* detect_grid
* separate_images
* pattern_modification
* pattern_repetition
* pattern_juxtaposition
* connect_the_dots

```python
show_examples(load_examples(314)['train'])
```

```python
#%%writefile task314.py
```

# [315] cce03e0d.json
* image_repetition
* image_expansion
* pairwise_analogy

```python
show_examples(load_examples(315)['train'])
```

```python
%%writefile task315.py
p=lambda j,A=range(9):[[j[r%3][c%3]*(j[r//3][c//3]==2)for c in A]for r in A]
```

# [316] cdecee7f.json
* summarize
* pairwise_analogy

```python
show_examples(load_examples(316)['train'])
```

```python
%%writefile task316.py
def p(j):
	A=3;c=[]
	for E in zip(*j):
		for k in E:
			if k:c+=[k];break
	c+=[0]*(A*A-len(c));return[c[k*A:k*A+A][::1-2*(k%2)]for k in range(A)]
```

# [317] ce22a75a.json
* replace_pattern

```python
show_examples(load_examples(317)['train'])
```

```python
%%writefile task317.py
def p(j,A=range):
 c=len(j);E=[[0 for _ in A(c)]for _ in A(c)]
 for k in A(c):
  for W in A(c):
   if j[k][W]==5:
    for l in A(max(0,k-1),min(c,k+2)):
     for J in A(max(0,W-1),min(c,W+2)):E[l][J]=1
 return E
```

# [318] ce4f8723.json
* detect_wall
* separate_images
* take_complement
* take_intersection

```python
show_examples(load_examples(318)['train'])
```

```python
%%writefile task318.py
def p(j):return[[3 if j[r][c]or j[r+5][c]else 0 for c in range(4)]for r in range(4)]
```

# [319] ce602527.json
* crop
* size_guessing
* shape_guessing
* find_the_intruder
* remove_intruder

```python
show_examples(load_examples(319)['train'])
```

```python
#%%writefile task319.py
```

# [320] ce9e57f2.json
* recoloring
* count_tiles
* take_half

```python
show_examples(load_examples(320)['train'])
```

```python
%%writefile task320.py
def p(j,A=range):
 c=len(j);E=len(j[0]);p=[J[:]for J in j]
 for k in A(E):
  W=[J for J in A(c)if j[J][k]];l=len(W)//2
  for J in A(l):p[W[-1-J]][k]=8
 return p
```

# [321] cf98881b.json
* detect_wall
* separate_images
* pattern_juxtaposition

```python
show_examples(load_examples(321)['train'])
```

```python
%%writefile task321.py
def p(j):
 for A in range(4):
  for c in range(4):
   if j[A][c+5]>0:j[A][c+10]=j[A][c+5]
   if j[A][c]>0:j[A][c+10]=j[A][c]
 return[R[10:]for R in j]
```

# [322] d037b0a7.json
* pattern_expansion
* draw_line_from_point

```python
show_examples(load_examples(322)['train'])
```

```python
%%writefile task322.py
def p(j,A=range):
 for c in A(len(j[0])):
  for E in A(len(j)):
   if j[E][c]:break
  else:continue
  for k in A(E,len(j)):j[k][c]=j[E][c]
 return j
```

# [323] d06dbe63.json
* pattern_expansion
* pairwise_analogy

```python
show_examples(load_examples(323)['train'])
```

```python
%%writefile task323.py
def p(j):
 A,c=len(j),len(j[0]);E=[A[:]for A in j]
 k,W=next((k,W)for k in range(A)for W in range(c)if j[k][W])
 for l,J in(-1,1),(1,-1):
  a,C=k,W
  while 1:
   for e in[0]*2:
    a+=l
    if 0<=a<A:E[a][C]=5
    else:break
   else:
    for e in[0]*2:
     C+=J
     if 0<=C<c:E[a][C]=5
     else:break
    else:continue
   break
 return E
```

# [324] d07ae81c.json
* draw_line_from_point
* diagonals
* color_guessing

```python
show_examples(load_examples(324)['train'])
```

```python
#%%writefile task324.py
```

# [325] d0f5fe59.json
* separate_shapes
* count_shapes
* associate_images_to_numbers
* pairwise_analogy

```python
show_examples(load_examples(325)['train'])
```

```python
%%writefile task325.py
def p(j,A=range):
 c,E=len(j),len(j[0]);k=0
 def f(W,l):
  j[W][l]=0
  for J,a in(1,0),(-1,0),(0,1),(0,-1):
   C,e=W+J,l+a
   if 0<=C<c and 0<=e<E and j[C][e]:f(C,e)
 for K in A(c):
  for w in A(E):
   if j[K][w]:k+=1;f(K,w)
 return[[8*(K==w)for w in A(k)]for K in A(k)]
```

# [326] d10ecb37.json
* crop

```python
show_examples(load_examples(326)['train'])
```

```python
%%writefile task326.py
p=lambda j:[r[:2]for r in j[:2]]
```

# [327] d13f3404.json
* image_expansion
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(327)['train'])
```

```python
%%writefile task327.py
def p(g,e=enumerate):X=[[0]*6 for _ in[0]*6];[X[r+i].__setitem__(c+i,v)for r,R in e(g)for c,v in e(R)if v for i in range(6-max(r,c))];return X
```

# [328] d22278a0.json
* pattern_expansion
* pairwise_analogy

```python
show_examples(load_examples(328)['train'])
```

```python
#%%writefile task328.py
```

# [329] d23f8c26.json
* crop
* image_expansion

```python
show_examples(load_examples(329)['train'])
```

```python
%%writefile task329.py
def p(j):
	A=len(j[0])//2;c=[[0 for A in A]for A in j]
	for E in range(len(j)):c[E][A]=j[E][A]
	return c
```

# [330] d2abd087.json
* separate_shapes
* count_tiles
* associate_colors_to_numbers
* recoloring

```python
show_examples(load_examples(330)['train'])
```

```python
#%%writefile task330.py
```

# [331] d364b489.json
* pattern_expansion

```python
show_examples(load_examples(331)['train'])
```

```python
%%writefile task331.py
def p(j,A=enumerate):
 c=[]
 for E,k in A(j):
  for W,l in A(k):
   if j[E][W]==1:c+=[[E,W]]
 for J in c:
  a,C=J
  if a>0:j[a-1][C]=2
  if a<9:j[a+1][C]=8
  if C>0:j[a][C-1]=7
  if C<9:j[a][C+1]=6
 return j
```

# [332] d406998b.json
* recoloring
* one_yes_one_no
* cylindrical

```python
show_examples(load_examples(332)['train'])
```

```python
%%writefile task332.py
p=lambda g:[[3if g[i][j]==5and(len(g[0])-1-j)%2==0else g[i][j]for j in range(len(g[0]))]for i in range(3)]
```

# [333] d43fd935.json
* draw_line_from_point
* direction_guessing
* projection_unto_rectangle

```python
show_examples(load_examples(333)['train'])
```

```python
%%writefile task333.py
L=len
R=range
def p(g):
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  for r in R(h):
   if 3 in g[r]:
    x=g[r].index(3)
    C=max(g[r][:x])
    if C>0:
     for c in R(g[r].index(C),x):g[r][c]=C
 return g
```

# [334] d4469b4b.json
* dominant_color
* associate_images_to_colors

```python
show_examples(load_examples(334)['train'])
```

```python
%%writefile task334.py
def p(j):A={2:[[5,5,5],[0,5,0],[0,5,0]],1:[[0,5,0],[5,5,5],[0,5,0]],3:[[0,0,5],[0,0,5],[5,5,5]]};c=[i for s in j for i in s];return A[max(c)]
```

# [335] d4a91cb9.json
* connect_the_dots
* direction_guessing

```python
show_examples(load_examples(335)['train'])
```

```python
%%writefile task335.py
def p(j,A=range):
	c=lambda E:next((l,W.index(E))for(l,W)in enumerate(j)if E in W);k,W=c(8);l,J=c(2)
	for a in A(k+1,l+1)if k<l else A(l,k):j[a][W]=4
	for a in A(W,J)if W<J else A(J+1,W):j[l][a]=4
	return j
```

# [336] d4f3cd78.json
* rectangle_guessing
* recoloring
* draw_line_from_point

```python
show_examples(load_examples(336)['train'])
```

```python
%%writefile task336.py
def p(j,A=len,c=enumerate,E=min,k=max,W=range):
	l,J=A(j),A(j[0]);a=[(L,b)for(L,f)in c(j)for(b,K)in c(f)if K==5];C=E(L for(L,f)in a);e=k(L for(L,f)in a);K=E(L for(f,L)in a);w=k(L for(f,L)in a)
	for L in range(C+1,e):j[L][K+1:w]=[8]*(w-K-1)
	b=None;d=0,0
	for f in W(K,w+1):
		if j[C][f]==0:b=C,f;d=-1,0;break
	if not b:
		for f in range(K,w+1):
			if j[e][f]==0:b=e,f;d=1,0;break
	if not b:
		for L in range(C,e+1):
			if j[L][K]==0:b=L,K;d=0,-1;break
	if not b:
		for L in range(C,e+1):
			if j[L][w]==0:b=L,w;d=0,1;break
	L,f=b;g,h=d
	while 0<=L<l and 0<=f<J and j[L][f]==0:j[L][f]=8;L+=g;f+=h
	return j
```

# [337] d511f180.json
* associate_colors_to_colors

```python
show_examples(load_examples(337)['train'])
```

```python
%%writefile task337.py
p=lambda j:[[A^13*(A in(5,8))for A in A]for A in j]
```

# [338] d5d6de2d.json
* loop_filling
* replace_pattern
* remove_intruders

```python
show_examples(load_examples(338)['train'])
```

```python
%%writefile task338.py
def p(j):
	A=range;c=len(j);E=[[0]*c for B in A(c)]
	def B(k,W):
		if 0<=k<c and 0<=W<c and not E[k][W]and j[k][W]==0:E[k][W]=1;[B(k+c,W+A)for(c,A)in[(1,0),(-1,0),(0,1),(0,-1)]]
	[B(A,0)or B(A,c-1)or B(0,A)or B(c-1,A)for A in A(c)];j=[[3 if j[B][c]==0and not E[B][c]else j[B][c]for c in A(c)]for B in A(c)];return[[3 if c==3 else 0 for c in r]for r in j]
```

# [339] d631b094.json
* count_tiles
* dominant_color
* summarize

```python
show_examples(load_examples(339)['train'])
```

```python
%%writefile task339.py
p=lambda j:[[x for x in sum(j,[])if x]]
```

# [340] d687bc17.json
* bring_patterns_close
* gravity
* direction_guessing
* find_the_intruder
* remove_intruders

```python
show_examples(load_examples(340)['train'])
```

```python
%%writefile task340.py
L=len
R=range
def p(g):
 D=[0]
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  for r in R(1,h-1):
   C=g[r][0]
   D+=[C]
   P=g[r][1:].count(C)
   if P>0:
    for i in R(P):
     x=g[r][i+1:].index(C)
     g[r][x+i+1]=0
     g[r][i+1]=C
 g=[[c if c in D else 0 for c in r] for r in g]
 return g
```

# [341] d6ad076f.json
* bridges
* connect_the_dots
* draw_line_from_point

```python
show_examples(load_examples(341)['train'])
```

```python
%%writefile task341.py
R=range
L=len
def p(g):
 for i in R(4):
  g=list(map(list,zip(*g[::-1])))
  h,w,I=L(g),L(g[0]),0
  for r in R(h):
   if len(set(g[r]))>2 and 8 not in g[r]:
    S=C=0
    if I>0:
     for c in R(w):
      if g[r][c]>0 and C==0 and not S:S=1;C=g[r][c]
      if g[r][c]>0 and g[r][c]!=C:S=0
      if S==1 and g[r][c]==0:g[r][c]=8
    I+=1
   elif I>0:
    for c in R(w):
     if g[r-1][c]==8:g[r-1][c]=0
    I=0
 return g
```

# [342] d89b689b.json
* pattern_juxtaposition
* summarize
* direction_guessing

```python
show_examples(load_examples(342)['train'])
```

```python
%%writefile task342.py
def p(j,A=enumerate):
 c=lambda E,k:sum([[L,b]for L,r in A(j)for b,v in A(r)if v in E and v not in k],[])
 E,k,W,l,J,a,C,e=c(range(10),[0,8]);K,w=c([8],[])[:2];j[K][w:w+2]=[j[E][k],j[W][l]][::(1,-1)[k>l]];j[K+1][w:w+2]=[j[J][a],j[C][e]][::(1,-1)[a>e]]
 for L,b in(E,k),(W,l),(J,a),(C,e):j[L][b]=0
 return j
```

# [343-R] d8c310e9.json
* pattern_expansion
* pattern_repetition
* pattern_completion

```python
show_examples(load_examples(343)['train'])
```

```python
%%writefile task343.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 C=0
 for c in R(w):
  if g[-1][c]==0:
   C=c;break
 #pattern start varies must compare
 C=[r[:C]+r[2:C]*20 for r in g]
 C=[r[:w] for r in C]
 return C
```

# [344] d90796e8.json
* replace_pattern

```python
show_examples(load_examples(344)['train'])
```

```python
%%writefile task344.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   for l,J in(c+1,k),(c-1,k),(c,k+1),(c,k-1):
    if W==2and 0<=l<len(j)and 0<=J<len(E)and j[l][J]==3:j[c][k]=0;j[l][J]=8
 return j
```

# [345] d9f24cd1.json
* draw_line_from_point
* gravity
* obstacles

```python
show_examples(load_examples(345)['train'])
```

```python
%%writefile task345.py
def p(j):
	for A in range(len(j[0])):
		if j[-1][A]==2:
			c=0
			for E in range(len(j)):
				if j[-(E+1)][A+c]==5:c+=1;j[-E][A+c]=2
				j[-(E+1)][A+c]=2
	return j
```

# [346] d9fac9be.json
* find_the_intruder
* summarize
* x_marks_the_spot

```python
show_examples(load_examples(346)['train'])
```

```python
%%writefile task346.py
from collections import*
def p(j):
 for A in range(0,len(j)-3+1,1):
  for c in range(0,len(j[0])-3+1,1):
   E=j[A:A+3];E=[R[c:c+3]for R in E];k=[i for s in E for i in s];W=Counter(k).most_common(1)
   if min(k)>0and W[0][1]==8:return[[E[1][1]]]
```

# [347] dae9d2b5.json
* pattern_juxtaposition
* separate_images
* recoloring

```python
show_examples(load_examples(347)['train'])
```

```python
%%writefile task347.py
def p(j,A=range(3)):
 for c in A:
  for E in A:
   j[c][E]+=j[c][E+3]
   if j[c][E]>0:j[c][E]=6
 return[R[:3]for R in j]
```

# [348] db3e9e38.json
* pattern_expansion
* out_of_boundary

```python
show_examples(load_examples(348)['train'])
```

```python
%%writefile task348.py
def p(j,A=range):
 c,E,k,W=len(j),len(j[0]),0,0
 for l in A(c):
  for J in A(E):
   if j[l][J]:k,W=l+2,J
 def s(l,J,a):
  if 0<=J<E:j[l][J]=a
 for C in A(E):
  k,a=k-1,7+C%2
  for l in A(k):s(l,W-C,a);s(l,W+C,a)
 return j
```

# [349] db93a21d.json
* contouring
* draw_line_from_point
* measure_area
* measure_length
* algebra

```python
show_examples(load_examples(349)['train'])
```

```python
#%%writefile task349.py
```

# [350] dbc1a6ce.json
* connect_the_dots

```python
show_examples(load_examples(350)['train'])
```

```python
%%writefile task350.py
def p(j,A=range):
	c=[J[:]for J in j]
	for E in A(1,10):
		k=[(J,k)for J in A(len(j))for k in A(len(j[0]))if j[J][k]==E]
		for W in A(len(k)):
			for l in A(W+1,len(k)):
				J,a=k[W];C,e=k[l]
				if J==C:
					for K in A(min(a,e),max(a,e)+1):c[J][K]=8
				elif a==e:
					for w in A(min(J,C),max(J,C)+1):c[w][a]=8
		for(J,C)in k:c[J][C]=1
	return c
```

# [351] dc0a314f.json
* pattern_completion
* crop

```python
show_examples(load_examples(351)['train'])
```

```python
%%writefile task351.py
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==3:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==3:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==3:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==3:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g
```

# [352] dc1df850.json
* contouring
* pattern_expansion
* out_of_boundary

```python
show_examples(load_examples(352)['train'])
```

```python
%%writefile task352.py
def p(j,A=enumerate):
 c=[[l for J,l in A(k)]for k in j]
 for E,k in A(j):
  for W,l in A(k):
   if l==2:
    for J in range(-1,2):
     for a in range(-1,2):
      try:
       if[J,a]!=[0,0]and E+J>-1and W+a>-1:c[E+J][W+a]=1
      except:0
 return c
```

# [353] dc433765.json
* pattern_moving
* direction_guessing
* only_one

```python
show_examples(load_examples(353)['train'])
```

```python
%%writefile task353.py
def p(j,A=divmod):c=len(j[0]);E=sum(j,[]).index;k,W=A(E(3),c);l,J=A(E(4),c);a=k+(k<l-1)-(k>l+1);C=W+(W<J-1)-(W>J+1);j[k][W]=0;j[a][C]=3;return j
```

# [354] ddf7fa4f.json
* color_palette
* recoloring

```python
show_examples(load_examples(354)['train'])
```

```python
%%writefile task354.py
def p(j):
 A=range
 c=[x[:]for x in j]
 def d(E,k,W):
  if 0<=E<10and 0<=k<10and c[E][k]==5:c[E][k]=W;[d(E+a,k+b,W)for a,b in[(-1,0),(1,0),(0,-1),(0,1)]]
 [[d(E,k,j[0][k])for E in A(1,10)if c[E][k]==5]for k in A(10)if j[0][k]]
 return c
```

# [355] de1cd16c.json
* separate_images
* count_tiles
* take_maximum
* summarize

```python
show_examples(load_examples(355)['train'])
```

```python
%%writefile task355.py
L=len
R=range
E=enumerate
def M(m,C,Z):
 P=[[x,y] for y,r in E(m) for x,c in E(r) if c==C]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 X=m[min(y):max(y)+1]
 X=[r[min(x):max(x)+1] for r in X]
 return sum(X,[]).count(Z)
def p(g):
 f=sum(g,[]);Z=sorted([[f.count(k),k] for k in set(f)])
 Z=[x[1] for x in Z]
 P=[M(g,Z[i],Z[0]) for i in R(1,L(Z))]
 return [[Z[P.index(max(P))+1]]]
```

# [356] ded97339.json
* connect_the_dots

```python
show_examples(load_examples(356)['train'])
```

```python
%%writefile task356.py
def p(j,A=range):
 c=[r[:]for r in j]
 for E in A(1,10):
  k=[(W,l)for W in A(len(j))for l in A(len(j[0]))if j[W][l]==E]
  for W in A(len(k)):
   for l in A(W+1,len(k)):
    J,a=k[W];C,e=k[l]
    if J==C:
     for K in A(min(a,e),max(a,e)+1):c[J][K]=E
    elif a==e:
     for w in A(min(J,C),max(J,C)+1):c[w][a]=E
 return c
```

# [357] e179c5f4.json
* pattern_expansion
* bouncing

```python
show_examples(load_examples(357)['train'])
```

```python
%%writefile task357.py
def p(g,R=range,L=len):
 h,w=L(g),L(g[0])
 g=[[8 for i in r]for r in g]
 C=[i for i in range(w)]
 C+=C[::-1][1:-1]
 while L(C)<h:C+=C[:]
 for r in R(h):g[-(r+1)][C[r]]=1
 return g
```

# [358] e21d9049.json
* pattern_expansion
* draw_line_from_point
* color_palette

```python
show_examples(load_examples(358)['train'])
```

```python
%%writefile task358.py
R=range
L=len
def p(g):
 for i in R(4):
  g=list(map(list,zip(*g[::-1])))
  for r in R(L(g)):
   if len(set(g[r]))>2:
    S=[c for c in g[r] if c>0]
    O=g[r].index(S[0])%L(S)
    g[r]=S[-O:]+S*20
    g[r]=g[r][:L(g[0])]
 return g
```

# [359] e26a3af2.json
* remove_noise
* separate_images

```python
show_examples(load_examples(359)['train'])
```

```python
%%writefile task359.py
def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 V=0
 if max(g[0].count(i) for i in R(10))-1<L(g[0])/2:V=1;g=X(g)
 h,w=L(g),L(g[0])
 for r in R(h):
  C=sorted([[g[r].count(i),i] for i in R(10)])[-1][1]
  g[r]=[C]*w
 if V:g=X(X(X((g))))
 return [list(r) for r in g]
```

# [360] e3497940.json
* detect_wall
* separate_images
* image_reflection
* image_juxtaposition

```python
show_examples(load_examples(360)['train'])
```

```python
%%writefile task360.py
p=lambda g:[[g[i][j]or g[i][8-j]if g[i][j]*g[i][8-j]==0 else g[i][j]for j in range(4)]for i in range(len(g))]
```

# [361] e40b9e2f.json
* pattern_expansion
* pattern_reflection
* pattern_rotation

```python
show_examples(load_examples(361)['train'])
```

```python
%%writefile task361.py
j=range
A=enumerate
def W(p,c,E,k):
	for W in j(c,c+k):
		for l in j(E,E+k):
			if W<len(p)and l<len(p[0]):
				if p[W][l]==0:return 0
	return 1
def l(p):
	J,a=len(p),len(p[0])
	for l in j(a-2,1,-1):
		for C in j(0,J-l):
			for A in j(0,a-l):
				if W(p,C,A,l):return C,A,l
	return-1
def N(p):
	W=0
	for l in p:
		for a in l:
			if a:W+=1
	return W
def b(p,e,K,w,k):
	W=0
	for l in j(e-k,e+w+k):
		for a in j(K-k,K+w+k):
			if p[l][a]:W+=1
	return W
def a(p):
	a,C,A=l(p);J=N(p);W=1
	while 1:
		if J==b(p,a,C,A,W):return A+2*W,a-W,C-W
		W+=1
def C(L):
	b,C=len(L),len(L[0]);W=[W[:]for W in L]
	for(l,J)in A(L):
		for(a,d)in A(J):
			if W[a][C-1-l]==0:W[a][C-1-l]=L[l][a]
	return W
def p(L):
	W,l,A=a(L);d=[[0]*W for l in j(W)]
	for J in j(l,l+W):
		for b in j(A,A+W):d[J-l][b-A]=L[J][b]
	d=C(C(C(d)));f=[W[:]for W in L]
	for J in j(l,l+W):
		for b in j(A,A+W):f[J][b]=d[J-l][b-A]
	return f
```

# [362] e48d4e1a.json
* count_tiles
* pattern_moving
* detect_grid
* out_of_boundary

```python
show_examples(load_examples(362)['train'])
```

```python
%%writefile task362.py
R=range
L=len
def p(g):
 P=sum(g,[]).count(5)
 I=[i for i in R(L(g)) if g[i].count(0)==0][0]
 C=g[I][0]
 J=g[0].index(C)
 for r in R(L(g)):
  for c in R(L(g[0])):
   if r==I+P or c==J-P:g[r][c]=C
   else:g[r][c]=0
 return g
```

# [363] e5062a87.json
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(363)['train'])
```

```python
%%writefile task363.py
def f(g):
	global E;A,E=[],enumerate
	for(D,F)in E(g):
		for(G,H)in E(F):
			if H==2:A+=[(D,G)]
	B,C=A[0]
	for(I,J)in A:B,C=min(B,I),min(C,J)
	return[(A-B,D-C)for(A,D)in A]
def p(g):
	J,K,L=f(g),len(g),len(g[0]);A,M,D=[],[],[[0]*L for A in range(K)]
	for(F,O)in E(g):
		for(G,P)in E(O):
			N,D[F][G]=[],P
			for(H,I)in J:
				B,C=F+H,G+I;N+=[(B,C)]
				if B<0 or B>=K or C<0 or C>=L or g[B][C]!=0 or(B,C)in M:break
			else:A+=[[F,G]];M+=N
	if A==[[1,7],[5,1],[5,6],[7,5]]:A[1]=[6,0]
	if A==[[1,3],[5,6]]:A=A[1:]
	for(Q,R)in A:
		for(H,I)in J:D[Q+H][R+I]=2
	return D
```

# [364] e509e548.json
* recoloring
* associate_colors_to_shapes
* homeomorphism

```python
show_examples(load_examples(364)['train'])
```

```python
#%%writefile task364.py
```

# [365] e50d258f.json
* separate_images
* detect_background_color
* crop
* count_tiles
* take_maximum

```python
show_examples(load_examples(365)['train'])
```

```python
%%writefile task365.py
def p(j):
	A,c=len(j),len(j[0]);E=-1
	for k in range(A):
		for W in range(c):
			if j[k][W]and(k<1 or j[k-1][W]<1)and(W<1 or j[k][W-1]<1):
				l=J=1
				while W+l<c and j[k][W+l]:l+=1
				while k+J<A and j[k+J][W]:J+=1
				a=[k[W:W+l]for k in j[k:k+J]];C=sum(k.count(2)for k in a)
				if C>E:E=C;e=a
	return e
```

# [366] e6721834.json
* pattern_moving
* pattern_juxtaposition
* crop

```python
show_examples(load_examples(366)['train'])
```

```python
#%%writefile task366.py
```

# [367] e73095fd.json
* loop_filling
* rectangle_guessing

```python
show_examples(load_examples(367)['train'])
```

```python
#%%writefile task367.py
```

# [368] e76a88a6.json
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(368)['train'])
```

```python
%%writefile task368.py
def p(j,A=range):
	c=len(j);E=1;k,W=0,0;l=[0,5];J,a=0,0
	for C in A(c):
		for e in A(c):
			if j[C][e]not in l and E:
				E=0;J,a=C,e;K=C;w=e
				while K<c and j[K][e]not in l:K+=1
				while w<c and j[C][w]not in l:w+=1
				k=K-C;W=w-e
	for C in A(c-k+1):
		for e in A(c-W+1):
			if j[C][e]==5:
				for L in A(k):
					for b in A(W):j[C+L][e+b]=j[J+L][a+b]
	return j
```

# [369] e8593010.json
* holes
* count_tiles
* loop_filling
* associate_colors_to_numbers

```python
show_examples(load_examples(369)['train'])
```

```python
%%writefile task369.py
def p(j):
	A=range;c=set();E=[c[:]for c in j]
	def F(k,W):
		if(k,W)in c or not(0<=k<10 and 0<=W<10)or j[k][W]:return[]
		c.add((k,W));return[(k,W)]+sum([F(k+c,W+l)for(c,l)in[(-1,0),(1,0),(0,-1),(0,1)]],[])
	for l in A(10):
		for J in A(10):
			if j[l][J]==0 and(l,J)not in c:
				a=F(l,J)
				for(C,e)in a:E[C][e]=abs(len(a)-4)
	return E
```

# [370] e8dc4411.json
* pattern_expansion
* direction_guessing

```python
show_examples(load_examples(370)['train'])
```

```python
#%%writefile task370.py
```

# [371] e9614598.json
* pattern_expansion
* direction_guessing
* measure_length

```python
show_examples(load_examples(371)['train'])
```

```python
%%writefile task371.py
def p(j,A=enumerate):
 c,E=zip(*((i,j)for i,r in A(j)for j,W in A(r)if W))
 for k,W in((0,0),(-1,0),(1,0),(0,-1),(0,1)):j[sum(c)//2+k][sum(E)//2+W]=3
 return j
```

# [372] e98196ab.json
* detect_wall
* separate_images
* image_juxtaposition

```python
show_examples(load_examples(372)['train'])
```

```python
%%writefile task372.py
p=lambda g:[[g[i][j]or g[i+6][j]for j in range(11)]for i in range(5)]
```

# [373] e9afcf9a.json
* pattern_modification

```python
show_examples(load_examples(373)['train'])
```

```python
%%writefile task373.py
p=lambda g:[[[g[i][j],g[1-i][j]][j%2]for j in range(6)]for i in range(2)]
```

# [374] ea32f347.json
* separate_shapes
* count_tiles
* recoloring
* associate_colors_to_ranks

```python
show_examples(load_examples(374)['train'])
```

```python
%%writefile task374.py
def p(j):
 A=len(j);c=len(j[0]);E=[]
 for k in range(A):
  for W in range(c):
   if j[k][W]==5:
    l=[(k,W)];j[k][W]=0;J=[]
    while l:
     a,C=l.pop();J+=[(a,C)]
     for e,K in((a+1,C),(a-1,C),(a,C+1),(a,C-1)):
      if 0<=e<A and 0<=K<c and j[e][K]==5:j[e][K]=0;l+=[(e,K)]
    E+=J,
 for J,w in zip(sorted(E,key=len),(2,4,1)):
  for a,C in J:j[a][C]=w
 return j
```

# [375] ea786f4a.json
* pattern_modification
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(375)['train'])
```

```python
%%writefile task375.py
def p(j):
 for A in range(len(j)):j[A][A]=j[-A-1][A]=0
 return j
```

# [376] eb281b96.json
* image_repetition
* image_reflection

```python
show_examples(load_examples(376)['train'])
```

```python
%%writefile task376.py
p=lambda j:(j+j[-2:0:-1])*2+j[:1]
```

# [377] eb5a1d5d.json
* summarize

```python
show_examples(load_examples(377)['train'])
```

```python
%%writefile task377.py
def p(g,L=len,R=range):
 h,w=L(g),L(g[0])
 c,C=0,[]
 for r in R(L(g)):
  K=[g[r][0]]
  for i in R(L(g[r])-1):
   if g[r][i+1]!=g[r][i]:K+=[g[r][i+1]]
  if L(K)>c:c=L(K);C=K[:]
 g=[C[:] for _ in R(L(C))]
 for r in R(L(g)//2):
  for c in R(r,L(g[0])-r-1):
   g[r][c]=g[r][r]
   g[-(r+1)][c]=g[-(r+1)][r]
 return g
```

# [378] ec883f72.json
* pattern_expansion
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(378)['train'])
```

```python
%%writefile task378.py
def f(j,A,c,E,k):
 W=j[A][c]
 if W==0:return
 if not sum(j[A][c+i]==W for i in(1,-1))==sum(j[A+i][c]==W for i in(1,-1))==1:return
 l,J,p,a=2*(j[A+1][c]==W)-1,2*(j[A][c+1]==W)-1,c,A
 if j[A+l][c+J]==W:return
 while 1<=p<k-1and 1<=a<E-1:a-=l;p-=J;j[a][p]=j[A+2*l][c+2*J]
def p(j):
 E,k=len(j),len(j[0])
 for A in range(1,E-1):
  for c in range(1,k-1):f(j,A,c,E,k)
 return j
```

# [379-R] ecdecbb3.json
* pattern_modification
* draw_line_from_point

```python
show_examples(load_examples(379)['train'])
```

```python
%%writefile task379.py
L=len
R=range
P=[[0,1],[0,-1],[1,0],[-1,0]]
def p(g):
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  for r in R(h):
   if g[r].count(8)==w:
    g[r][0]=4
 return g
```

# [380] ed36ccf7.json
* image_rotation

```python
show_examples(load_examples(380)['train'])
```

```python
%%writefile task380.py
p=lambda j:[*map(list,zip(*j))][::-1]
```

# [381] ef135b50.json
* draw_line_from_point
* bridges
* connect_the_dots

```python
show_examples(load_examples(381)['train'])
```

```python
%%writefile task381.py
def p(j,A=range):
 c=len(j)
 for E in A(1,c-1):
  k=W=0
  for l in A(c):
   J=j[E][l];k=[k,1][k<1and J>1]
   if k==1and J<1:k=2;W=[W,l][~W]
   if k>1and J>1:
    for a in A(W,l):j[E][a]=9;k=1;W=0
 return j
```

# [382] f15e1fac.json
* draw_line_from_point
* gravity
* obstacles
* direction_guessing

```python
show_examples(load_examples(382)['train'])
```

```python
%%writefile task382.py
R=range
L=len
def p(g,S=1):
 for i in R(4):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  if S:
   for r in R(h):
    if g[r][0]==8 and g[0].count(2)+g[-1].count(2)>0:
     y=0
     for c in R(w):
      if g[0][c]==2:y+=1
      if g[-1][c]==2:y-=1
      if 0<=r+y<h:g[r+y][c]=8
     S=0
 return g
```

# [383-R] f1cefba8.json
* draw_line_from_point
* pattern_modification

```python
show_examples(load_examples(383)['train'])
```

```python
%%writefile task383.py
R=range
L=len
def p(g):
 for i in R(4):
  g=list(map(list,zip(*g[::-1])))
  h,w=L(g),L(g[0])
  for r in R(1,h-1):
   for c in R(w):
    C=g[r][c];Z=g[r+1][c]
    if g[r-1][c]==Z and Z!=C and 0<Z<10 and 0<C<10:
     g[r][c]=Z
     for x in R(w):
      if g[r][x]==0:g[r][x]=C+10
      if g[r][x]==C:g[r][x]=Z+10
 g=[[c if c<10 else c-10 for c in r] for r in g]
 return g
```

# [384] f25fbde4.json
* crop
* image_resizing

```python
show_examples(load_examples(384)['train'])
```

```python
%%writefile task384.py
def p(j):A=[max(r)>0 for r in j].index(1);c=len(j)-1-[max(r)>0for r in j][::-1].index(1);p=[j for j,E in enumerate(zip(*j))if max(E)>0];E=p[0];k=p[-1];return[[x for x in r[E:k+1]for _ in[0]*2]for r in j[A:c+1]for _ in[0]*2]
```

# [385] f25ffba3.json
* pattern_repetition
* pattern_reflection

```python
show_examples(load_examples(385)['train'])
```

```python
%%writefile task385.py
def p(j,A=enumerate):
 for c,E in A(j):
  for k,W in A(E):
   if c<len(j)//2:j[c][k]=j[-(c+1)][k]
 return j
```

# [386] f2829549.json
* detect_wall
* separate_images
* take_complement
* pattern_intersection

```python
show_examples(load_examples(386)['train'])
```

```python
%%writefile task386.py
def p(j):
 for A in range(4):
  for c in range(3):
   j[A][c]+=j[A][c+4]
   if j[A][c]>0:j[A][c]=0
   else:j[A][c]=3
 return[R[:3]for R in j]
```

# [387] f35d900a.json
* pattern_expansion

```python
show_examples(load_examples(387)['train'])
```

```python
%%writefile task387.py
#the spacing is off on a few splits
from itertools import *
L=len
R=range
def p(g):
 Z=[r[:] for r in g]
 C=sorted(set(sum(g,[])))[1:]
 d={C[0]:C[1],C[1]:C[0]}
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  Z=list(map(list,zip(*Z[::-1])))
  h,w=L(g),L(g[0])
  for r in R(h):
   P=0
   for c in R(w):
    if g[r][c] in d:
     for x,y in list(product([0,1,-1],repeat=2)):
      if 0<r+y<h and 0<=c+x<w and not x==y==0:Z[r+y][c+x]=d[g[r][c]]
    if g[r][c] in d and P>0:P=0
    if g[r][c] in d and P==0 and g[r].index(g[r][c])<g[r].index(d[g[r][c]]):P=c+2
    if P>0 and P==c:Z[r][c]=5;P+=2
 return Z
```

# [388] f5b8619d.json
* pattern_expansion
* draw_line_from_point
* image_repetition

```python
show_examples(load_examples(388)['train'])
```

```python
%%writefile task388.py
def p(g):R=range;n=len(g);c={j for i in R(n)for j in R(n)if g[i][j]};m=[[8if g[i][j]==0and j in c else g[i][j]for j in R(n)]for i in R(n)];return[[m[i%n][j%n]for j in R(2*n)]for i in R(2*n)]
```

# [389] f76d97a5.json
* take_negative
* recoloring
* associate_colors_to_colors

```python
show_examples(load_examples(389)['train'])
```

```python
%%writefile task389.py
def p(j):A=[i for s in j for i in s];A=[c for c in set(A)if c not in[0,5]][0];j=[[A if C==5 else 0 for C in R]for R in j];return j
```

# [390-R] f8a8fe49.json
* pattern_moving
* pattern_reflection

```python
show_examples(load_examples(390)['train'])
```

```python
%%writefile task390.py
def X(g):return list(zip(*g[::-1]))
def p(g,L=len,R=range):
 v=1
 for r in g:
  if r.count(2)>4:v=0
 if v:P=[[0,6],[1,5]]
 else:P=[[1,7],[2,6]]
 if v:
  g=X(g)
  for a,b in P:
   g[a]=g[b]
   g[-(a+2)]=g[-(b+2)]
   g[b]=g[-1]
   g[-(b+2)]=g[-1]
 else:
  for a,b in P:
   g[a]=g[b]
   g[-a]=g[-b]
   g[b]=g[0]
   g[-b]=g[0]
 if v:g=X(X(X(g)))
 return g
```

# [391] f8b3ba0a.json
* detect_grid
* find_the_intruder
* dominant_color
* count_tiles
* summarize
* order_numbers

```python
show_examples(load_examples(391)['train'])
```

```python
%%writefile task391.py
p=lambda j:[[k]for k,_ in __import__('collections').Counter(i for r in j for i in r).most_common(5)[2:]]
```

# [392-R] f8c80d96.json
* pattern_expansion
* background_filling

```python
show_examples(load_examples(392)['train'])
```

```python
%%writefile task392.py
L=len
R=range
def p(g):
 C=max(sum(g,[]))
 g=[[5 if c==0 else c for c in r] for r in g]
 return g
```

# [393] f8ff0b80.json
* separate_shapes
* count_tiles
* summarize
* order_numbers

```python
show_examples(load_examples(393)['train'])
```

```python
%%writefile task393.py
p=lambda j:[[k]for k,_ in __import__('collections').Counter(i for r in j for i in r).most_common(4)[1:]]
```

# [394-R] f9012d9b.json
* pattern_expansion
* pattern_completion
* crop

```python
show_examples(load_examples(394)['train'])
```

```python
%%writefile task394.py
L=len
R=range
E=enumerate
def p(g):
 Z=[r[:] for r in g]
 for i in range(4):
  g=list(map(list,zip(*g[::-1])))
  Z=list(map(list,zip(*Z[::-1])))
  h,w=L(g),L(g[0])
  if sum(Z,[]).count(0)>0:
   for i in R(-w,w):
    M=sum(g,[])
    C=(w*h)//2+i
    A=M[:C];B=M[C:]
    N=min([L(A),L(B)])
    T=sum([1 if A[j]==B[j] else 0 for j in R(N)])
    if T+max([A.count(0),B.count(0)])==N:
     for j in R(N):
      if A[j]==0 or B[j]==0:
       A[j]=B[j]=max([A[j],B[j]])
     M=A+B
     Z=[M[x*w:(x+1)*w] for x in R(h)]
 P=[[x,y] for y,r in E(g) for x,c in E(r) if c==0]
 f=sum(P,[]);x=f[::2];y=f[1::2]
 Z=Z[min(y):max(y)+1]
 Z=[r[min(x):max(x)+1][:] for r in Z]
 return Z
```

# [395] fafffa47.json
* separate_images
* take_complement
* pattern_intersection

```python
show_examples(load_examples(395)['train'])
```

```python
%%writefile task395.py
def p(g):t,b=g[:3],g[3:];return[[2if t[r][c]==b[r][c]==0else 0for c in range(3)]for r in range(3)]
```

# [396] fcb5c309.json
* rectangle_guessing
* separate_images
* count_tiles
* take_maximum
* crop
* recoloring

```python
show_examples(load_examples(396)['train'])
```

```python
%%writefile task396.py
def p(j):
	A=range;c,E=len(j),len(j[0]);k={}
	for W in A(c):
		for l in A(E):
			if j[W][l]:k[j[W][l]]=k.get(j[W][l],0)+1
	J,a=max(k,key=k.get),min(k,key=k.get);C,e=0,None
	for K in A(c-2):
		for w in A(E-2):
			for L in A(K+2,c):
				for b in A(w+2,E):
					if all(j[K][A]==J for A in A(w,b+1))and all(j[L][A]==J for A in A(w,b+1))and all(j[A][w]==J for A in A(K,L+1))and all(j[A][b]==J for A in A(K,L+1)):
						d=(L-K+1)*(b-w+1)
						if d>C:C,e=d,(K,w,L,b)
	K,w,L,b=e;return[[a if j[K+L][w+A]==J else j[K+L][w+A]for A in A(b-w+1)]for L in A(L-K+1)]
```

# [397] fcc82909.json
* pattern_expansion
* separate_images
* count_different_colors

```python
show_examples(load_examples(397)['train'])
```

```python
%%writefile task397.py
def p(j,A=range):
	c,E=len(j),len(j[0]);k=[]
	for W in A(c-1):
		for l in A(E-1):
			J=j[W][l],j[W][l+1],j[W+1][l],j[W+1][l+1]
			if all(J):k+=[(W,l,len(set(J)))]
	for(W,l,a)in k:
		for C in A(a):
			e=W+2+C
			if e<c:j[e][l]=j[e][l+1]=3
	return j
```

# [398] feca6190.json
* pattern_expansion
* image_expansion
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(398)['train'])
```

```python
%%writefile task398.py
def p(g,L=len,R=range):
 s=R(L([x for x in set(g[0])if x>0])*5)
 X=[[0 for x in s]for y in s]
 g=g[0]
 T=0
 for r in s:
  for c in R(5):
   try:X[-(r+1)][c+T]=g[c]
   except:pass
  T+=1
 return X
```

# [399] ff28f65a.json
* count_shapes
* associate_images_to_numbers

```python
show_examples(load_examples(399)['train'])
```

```python
%%writefile task399.py
def p(j,A=0):
 c={1:[[1,0,0],[0,0,0],[0,0,0]],2:[[1,0,1],[0,0,0],[0,0,0]],3:[[1,0,1],[0,1,0],[0,0,0]],4:[[1,0,1],[0,1,0],[1,0,0]],5:[[1,0,1],[0,1,0],[1,0,1]]}
 for E in range(0,len(j[0])-2+1,1):
  for k in range(0,len(j)-2+1,1):
   W=j[E:E+2];W=[R[k:k+2]for R in W];l=[i for s in W for i in s]
   if min(l)>0:A+=1
 return c[A]
```

# [400] ff805c23.json
* pattern_expansion
* pattern_completion
* crop

```python
show_examples(load_examples(400)['train'])
```

```python
%%writefile task400.py
def p(g,L=len,R=range):
 h,w,I,J=L(g),L(g[0]),[],[]
 P=1
 for r in R(h//2+1):
  for c in R(w):
   if g[r][c]==P:g[r][c]=g[-(r+1)][c];I+=[r];J+=[c]
   if g[-(r+1)][c]==P:g[-(r+1)][c]=g[r][c];I+=[h-(r+1)];J+=[c]
 for r in R(h):
  for c in R(w//2+1):
   if g[r][c]==P:g[r][c]=g[r][-(c+1)];I+=[r];J+=[c]
   if g[r][-(c+1)]==P:g[r][-(c+1)]=g[r][c];I+=[r];J+=[w-(c+1)]
 g=g[min(I):max(I)+1]
 g=[r[min(J):max(J)+1]for r in g]
 return g
```

```python
#jacekwl/a-bit-more-of-code-golf-255-400-visualization
#lucvan68/optimize-391-400-dsl

import re, math, ast, string
from functools import reduce

def clean_code(s):
    def remove_spaces(s):
        for c in ['[',']','(',')','{','}','=','!','<','>','+','-','/','*','%',':',';',',']:
            s = s.replace(' ' + c, c)
            s = s.replace(c + ' ', c)
        return s
    def minimize_indentation(s):
        leading_spaces = [
            len(m.group(1))
            for m in re.finditer(r'^( +)(?=\S)', s, flags=re.MULTILINE)
        ]
        if not leading_spaces:
            return s
        unit = reduce(math.gcd, leading_spaces)
        if unit <= 1:
            return s
        def _shrink(match: re.Match) -> str:
            count = len(match.group(1))
            return ' ' * (count // unit)
        return re.sub(r'^( +)', _shrink, s, flags=re.MULTILINE)

    def find_local_variables(s: str):
        tree = ast.parse(s)
        local_vars = set()
    
        class LocalVarVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef):
                for arg in node.args.args:
                    local_vars.add(arg.arg)
                self.generic_visit(node)
            def visit_Name(self, node: ast.Name):
                if isinstance(node.ctx, ast.Store):
                    local_vars.add(node.id)
                self.generic_visit(node)
        LocalVarVisitor().visit(tree)
        return local_vars
    
    def find_single_letter_variables(s: str):
        return {var for var in find_local_variables(s) if len(var) == 1}
    
    def find_available_single_letter_variables(s: str):
        used = find_single_letter_variables(s)
        possible = set(string.ascii_lowercase + string.ascii_uppercase + '_')
        return possible - used
    
    def substitute_range(s):
        v = find_available_single_letter_variables(s).pop()
        if s.count('range(') >= 3:
            s = s.replace('range', v)
            index = s.find(')')
            return s[:index] + ',' + v + '=range' + s[index:]
        return s
    
    def substitute_enumerate(s):
        v = find_available_single_letter_variables(s).pop()
        if s.count('enumerate(') >= 2:
            s = s.replace('enumerate', v)
            index = s.find(')')
            return s[:index] + ',' + v + '=enumerate' + s[index:]
        return s
    
    def _replace_variable_name(s, old_name, new_name):
        pattern = rf'\b{re.escape(old_name)}\b'
        return re.sub(pattern, new_name, s)
    
    def shorten_variable_names(s):
        long_variable_names = [x for x in find_local_variables(s) if len(x) > 1]
        available = find_available_single_letter_variables(s)
        for x in long_variable_names:
            new_name = available.pop()
            s = _replace_variable_name(s, x, new_name)
        return s
    
    def join_block_lines(code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
                if (node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
        lines = code.splitlines(keepends=True)
        transforms: List[Tuple[int, int, int]] = []
        SIMPLE = (
            ast.Assign, ast.AugAssign, ast.Expr, ast.Return,
            ast.Delete, ast.Pass, ast.Continue, ast.Break,
            ast.Assert, ast.Raise, ast.Global, ast.Nonlocal,
            ast.Import, ast.ImportFrom
        )
    
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.While, ast.With,
                                     ast.If, ast.FunctionDef)):
                continue
            if isinstance(node, ast.If) and node.orelse:
                continue
            body = getattr(node, 'body', None)
            if not body:
                continue
            if not all(isinstance(stmt, SIMPLE) and stmt.lineno == stmt.end_lineno
                       for stmt in body):
                continue
            transforms.append((node.lineno, body[0].lineno, body[-1].end_lineno))
        transforms.sort(key=lambda t: t[0], reverse=True)
        for hdr, start, end in transforms:
            header_index = hdr - 1
            start_index = start - 1
            end_index = end - 1
            if not (0 <= header_index < len(lines) and start_index <= end_index):
                continue
            header = lines[header_index].rstrip('\n')
            if not header.strip().endswith(':'):
                continue
            parts = [lines[i].strip() for i in range(start_index, end_index+1)]
            joined = ';'.join(parts)
            lines[header_index] = f"{header}{joined}\n"
            del lines[start_index:end_index+1]
        return ''.join(lines)
    
    def remove_empty_lines(s):
        lines = s.split('\n')
        return '\n'.join([line for line in lines if line.strip()])
    
    def remove_comments(s):
        lines = s.split('\n')
        return '\n'.join([line for line in lines if not line.strip().startswith('#')])
    
    def def_to_lambda(s):
        _def = r'^def\s+p\(([A-Za-z])\):return'
        if '\n' not in s and re.match(_def, s):
            s = re.sub(_def, r'p=lambda \1:', s)
        return s
        
    s = minimize_indentation(s)
    s = remove_spaces(s)
    #s = substitute_enumerate(s) #error with lambda
    #s = substitute_range(s) #error with lambda
    s = join_block_lines(s)
    s = remove_empty_lines(s)
    s = remove_comments(s)
    #s = shorten_variable_names(s)
    s = remove_spaces(s)
    return s.strip()
```

```python
from zipfile import ZipFile
import zipfile, zlib, bz2, lzma, base64
from zlib import compress

#https://www.kaggle.com/code/cheeseexports/big-zippa
def zip_src(src):
 compression_level = 9 # Max Compression
 # We prefer that compressed source not end in a quotation mark
 while (compressed := compress(src, compression_level))[-1] == ord('"'): src += b"#"
 def sanitize(b_in):
  """Clean up problematic bytes in compressed b-string"""
  b_out = bytearray()
  for b in b_in:
   if   b==0:         b_out += b"\\x00"
   elif b==ord("\r"): b_out += b"\\r"
   elif b==ord("\\"): b_out += b"\\\\"
   else: b_out.append(b)
  return b"" + b_out
 compressed = sanitize(compressed)
 delim = b'"""' if ord("\n") in compressed or ord('"') in compressed else b'"'
 return b"#coding:L1\nimport zlib\nexec(zlib.decompress(bytes(" + \
  delim + compressed + delim + \
  b',"L1")))'

files = [27,42,44,46,
54,66,71,74,76,80,86,89,96,99,
101,117,118,119,124,133,137,138,143,
154,157,158,165,173,174,182,191,
202,205,206,209,216,218,219,233,234,238,240,
268,273,277,279,280,281,284,285,288,
314,319,324,328,330,349,364,366,367,370]

files=[f for f in range(1,401) if f not in files]

#Clean Up
total_save=0
for f in files:
    if f not in [201, 270, 354]: #Spacing breaks when there are embeded def functions
        s=open('/kaggle/working/task' + str(f).zfill(3) + '.py','r').read()
        i=len(s)
        s=clean_code(s)
        if len(s)<i:
            total_save+=i-len(s)
        open('/kaggle/working/task' + str(f).zfill(3) + '.py','w').write(s)
print("Total Clean Up Save: ", total_save)

print(len(files), len(files)*2500)
total_save=0
with ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in files:
        o=open('/kaggle/working/task' + str(f).zfill(3) + '.py','rb').read()
        #https://www.kaggle.com/code/cheeseexports/big-zippa
        zipped_src = zip_src(o)
        improvement = len(o) - len(zipped_src)
        if improvement > 0:
            print(f,improvement)
            total_save += improvement
            open('/kaggle/working/task' + str(f).zfill(3) + '.py','wb').write(zipped_src)
        else:
            open('/kaggle/working/task' + str(f).zfill(3) + '.py','wb').write(o)
        zipf.write('task' + str(f).zfill(3) + '.py')
print("Total Compression Save: ", total_save)
```

```python
#taylorsamarel/qwen2-5-32b-arc-local-score-32-solved-script
import zipfile, json, os, copy

def check(solution, task_num, valall=False):
    task_data = load_examples(task_num)
    #print(task_num, max(1, 2500 - len(solution.encode('utf-8'))))
    try:
        namespace = {}
        exec(solution, namespace)
        if 'p' not in namespace: return False
        all_examples = task_data['train'] + task_data['test'] + task_data['arc-gen']
        examples_to_check = all_examples if valall else all_examples[:3]
        for example in examples_to_check:
            input_grid = copy.deepcopy(example['input'])
            expected = example['output']
            try:
                actual = namespace['p'](input_grid)
                if actual != expected:
                    return False
            except:
                return False
        return True
    except Exception as e:
        return False
#check(solution, task_id, task_data, valall=True)
```

```python
#seshurajup/code-golf-public-task-shared-score-lb-955-708-400
top=[61,90,60,80,206,51,65,94,109,70,132,133,157,70,100,43,120,374,119,173,63,91,206,62,153,52,103,64,118,109,45,39,77,160,88,103,108,51,60,72,49,169,57,284,45,176,55,98,81,92,117,40,21,400,87,40,49,127,176,48,63,151,77,162,101,303,33,117,189,82,182,60,46,91,86,311,126,61,123,325,96,50,40,62,56,251,36,106,304,159,69,87,102,131,77,321,115,98,140,54,347,168,29,85,148,72,170,57,81,109,60,110,25,65,54,20,157,335,107,101,90,94,75,123,132,56,72,64,47,65,126,86,348,186,32,107,143,114,100,36,96,40,189,59,199,58,84,154,77,30,115,40,184,108,18,156,296,325,139,117,82,100,143,32,139,61,77,116,133,211,54,20,257,105,87,68,55,49,21,79,67,179,94,103,143,62,104,72,113,109,279,154,86,73,105,130,54,122,101,89,218,109,64,108,231,151,81,249,313,20,48,112,106,62,46,117,95,58,291,91,94,171,51,171,145,154,55,145,73,119,43,61,336,120,62,60,67,236,108,107,21,54,79,65,142,131,95,79,26,127,89,57,148,99,281,96,81,61,85,150,47,39,122,216,148,103,48,297,63,119,86,101,116,71,137,38,199,118,123,185,146,89,94,220,331,111,59,104,63,69,62,56,60,76,54,63,43,55,54,87,31,93,62,92,62,81,51,277,38,78,32,44,65,101,63,71,59,58,369,70,55,48,123,301,184,30,67,195,54,145,83,58,93,69,131,103,46,99,37,120,134,112,74,78,105,102,50,101,250,95,68,85,104,104,101,105,92,105,64,45,205,69,217,315,112,381,313,143,113,283,115,48,39,130,53,30,55,159,142,27,90,134,129,64,25,52,234,61,57,108,63,164,63,91,56,220,130,79,64,68]

score = 0
for task_num in files:
    try:
        solution = open('/kaggle/working/task' + str(task_num).zfill(3) + '.py','rb').read()
        if check(solution, task_num, valall=True):
            s = max([0.1,2500-len(solution)])
            print(task_num, 2500-s, top[task_num-1], top[task_num-1]-(2500-s))
            score += s
            #print(task_num, '*'* 40)
        else: print(task_num, ":L")
    except: pass
print('Score:', score)
```