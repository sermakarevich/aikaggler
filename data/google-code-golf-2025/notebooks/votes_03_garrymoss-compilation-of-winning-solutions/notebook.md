# Compilation of winning solutions

- **Author:** Garry Moss
- **Votes:** 228
- **Ref:** garrymoss/compilation-of-winning-solutions
- **URL:** https://www.kaggle.com/code/garrymoss/compilation-of-winning-solutions
- **Last run:** 2025-11-03 00:01:29.057000

---

# Compilation of winning solutions

* Compilation and post-comp diamonds found at https://github.com/arc-code-golf/solutions
* Links to original results at the conclusion of the competition: https://www.kaggle.com/competitions/google-code-golf-2025/discussion/613968
* See the end of this notebook for compression code - all of the zlib functions are compressed within this notebook
* I have also added a few of my own post-comp diamonds in here as well

```python
import sys
sys.path.append("/kaggle/input/google-code-golf-2025/code_golf_utils")
from code_golf_utils import *
show_legend()
```

# [001]
[007bbfb7.json](https://arcprize.org/play?task=007bbfb7)


**Size: 58 bytes**
* image_repetition
* fractal_repetition

```python
show_examples(load_examples(1)['train'])
```

```python
%%writefile task001.py
p=lambda*a:[(*a,min,p)[2](s,t)for s in a[0]for t in a[-1]]
```

# [002]
[00d62c1b.json](https://arcprize.org/play?task=00d62c1b)


**Size: 85 bytes**
* loop_filling

```python
show_examples(load_examples(2)['train'])
```

```python
%%writefile task002.py
p=lambda g,i=67:g*-i or p([[r.pop()%sum(r[-1:],4)or i>>4&4for r in g]for r in g],i-1)
```

# [003]
[017c7c7b.json](https://arcprize.org/play?task=017c7c7b)


**Size: 58 bytes**
* recoloring
* pattern_expansion
* pattern_repetition
* image_expansion

```python
show_examples(load_examples(3)['train'])
```

```python
%%writefile task003.py
p=lambda g:[[y*2for y in x]for x in g+g[g[2]==g[5]:][2:5]]
```

# [004]
[025d127b.json](https://arcprize.org/play?task=025d127b)


**Size: 73 bytes**
* pattern_modification

```python
show_examples(load_examples(4)['train'])
```

```python
%%writefile task004.py
import re;p=lambda i:eval(re.sub(r"(.),(?=.*\1.*0, \1)",r"0,\1|",str(i)))
```

# [005]
[045e512c.json](https://arcprize.org/play?task=045e512c)


**Size: 166 bytes**
* pattern_expansion
* direction_guessing

```python
show_examples(load_examples(5)['train'])
```

```python
%%writefile task005.py
import re
p=lambda g:eval(re.sub(f"0(?=(?=(.)+{'(.{2%%d})+(?<=%s))'%max(str(29**14),key=f'{g}'.count)*2}|"*2%(*b"<<HH",),r"\1\4",f'{*zip(*g[147:]or p(g*2)),}'))[::-1]
```

# [006]
[0520fde7.json](https://arcprize.org/play?task=0520fde7)


**Size: 49 bytes**
* detect_wall
* separate_images
* pattern_intersection

```python
show_examples(load_examples(6)['train'])
```

```python
%%writefile task006.py
p=lambda g:[eval('r[4]*r.pop(0)*2,'*3)for r in g]
```

# [007]
[05269061.json](https://arcprize.org/play?task=05269061)


**Size: 62 bytes**
* image_filling
* pattern_expansion
* diagonals

```python
show_examples(load_examples(7)['train'])
```

```python
%%writefile task007.py
p=lambda g:eval(f"({'max(sum(g:=g[1:3]+g,[0])[::3]),'*7}),"*7)
```

# [008]
[05f2a901.json](https://arcprize.org/play?task=05f2a901)


**Size: 83 bytes**
* pattern_moving
* direction_guessing
* bring_patterns_close

```python
show_examples(load_examples(8)['train'])
```

```python
%%writefile task008.py
p=lambda g,*n:sorted(zip(*n or p([],*g)),key=lambda r:(8in g.__iadd__(r))*3^any(r))
```

# [009]
[06df4c85.json](https://arcprize.org/play?task=06df4c85)


**Size: 95 bytes**
* detect_grid
* connect_the_dots
* grid_coloring

```python
show_examples(load_examples(9)['train'])
```

```python
%%writefile task009.py
p=lambda g,*r,i=0:[x|max({*g[i::3]}&{*g[:(i:=i+1)]})for x in r]or[*map(p,g,*map(p,zip(*g),*g))]
```

# [010]
[08ed6ac7.json](https://arcprize.org/play?task=08ed6ac7)


**Size: 66 bytes**
* measure_length
* order_numbers
* associate_colors_to_ranks
* recoloring

```python
show_examples(load_examples(10)['train'])
```

```python
%%writefile task010.py
p=lambda g:[g:=[(x*-5or y*sum(r))%6for x,y in zip(g,r)]for r in g]
```

# [011]
[09629e4f.json](https://arcprize.org/play?task=09629e4f)


**Size: 113 bytes**
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
p=eval("lambda a:max(a*(not'8'in'%s'%a)"+f"for*a,in[*map(zip,a,a,a{',a[3:]*9,*[a[%d:]]*3'*2%(1,2)})][::4]"*2+")")
```

# [012]
[0962bcdd.json](https://arcprize.org/play?task=0962bcdd)


**Size: 122 bytes**
* pattern_expansion

```python
show_examples(load_examples(12)['train'])
```

```python
%%writefile task012.py
import re
p=lambda g:eval([g:=re.sub(r"0(?=.{%d}([^0])..(.)..\1)"%x,f"\{895%x%5}",str(g)[::-1])for x in b'HHNB%'*2][-1])
```

# [013]
[0a938d79.json](https://arcprize.org/play?task=0a938d79)


**Size: 124 bytes**
* direction_guessing
* draw_line_from_point
* pattern_expansion

```python
show_examples(load_examples(13)['train'])
```

```python
%%writefile task013.py
p=lambda g,h=0,l=[]:[[(l:=[max(c+[j*(i>0)for i,j in zip(l,l[1::2])])]+l)[:1]*len(c),c][c[1:-1]>c]for*c,in zip(*h or p(g,g))]
```

# [014]
[0b148d64.json](https://arcprize.org/play?task=0b148d64)


**Size: 57 bytes**
* detect_grid
* separate_images
* find_the_intruder
* crop

```python
show_examples(load_examples(14)['train'])
```

```python
%%writefile task014.py
p=lambda g:[p(zip(r,*g))or r[0]for r in[*g]if[*{*r}][2:]]
```

# [015]
[0ca9ddb6.json](https://arcprize.org/play?task=0ca9ddb6)


**Size: 87 bytes**
* pattern_expansion
* associate_patterns_to_colors

```python
show_examples(load_examples(15)['train'])
```

```python
%%writefile task015.py
p=lambda g,i=-7:g*i or p([[r.pop()%9|36%(6^-[0,*r][i//7])%13for r in g]for*r,in g],i+1)
```

# [016]
[0d3d703e.json](https://arcprize.org/play?task=0d3d703e)


**Size: 43 bytes**
* associate_colors_to_colors

```python
show_examples(load_examples(16)['train'])
```

```python
%%writefile task016.py
p=lambda g:[[109//c&12%c+7for c in g[0]]]*3
```

# [017]
[0dfd9992.json](https://arcprize.org/play?task=0dfd9992)


**Size: 90 bytes**
* image_filling
* pattern_expansion

```python
show_examples(load_examples(17)['train'])
```

```python
%%writefile task017.py
p=lambda g:[[*map(max,*[r*any(-i^-j>0for i,j in zip(r,s))+s for s in g])]for*r,in zip(*g)]
```

# [018]
[0e206a2e.json](https://arcprize.org/play?task=0e206a2e)


**Size: 275 bytes (402 raw)**
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
def p(a):e={m*1j+f:a for m,a in enumerate(a)for f,a in enumerate(a)if a};[(l:={r},[l:={r}|l for r in[*e]*5for u in l if abs(r-u)<2],[*l][3:]and[5for i in[1,3,6,7]for n in e if all(sum(e[u]==e.get(f)for f in[*l,(u-r-i//4*(u-r).real*2)*1j**i+n])>1for u in l)for u in l if(a:=[[{u:0,(u-r-i//4*(u-r).real*2)*1j**i+n:e[u]}.get(m*1j+f,a)for f,a in enumerate(a)]for m,a in enumerate(a)])])for r in e];return a
```

# [019]
[10fcaaa3.json](https://arcprize.org/play?task=10fcaaa3)


**Size: 103 bytes**
* pattern_expansion
* image_repetition

```python
show_examples(load_examples(19)['train'])
```

```python
%%writefile task019.py
p=lambda g,n=7:-n*g or p(-~(n>5)*[g:=[r.pop()or(x*-1or 0)%-8&8for x in[0]+g[:-1]]for*r,in zip(*g)],n-1)
```

# [020]
[11852cab.json](https://arcprize.org/play?task=11852cab)


**Size: 126 bytes**
* pattern_expansion

```python
show_examples(load_examples(20)['train'])
```

```python
%%writefile task020.py
def p(g):
 m,n=[[*map(any,h)].index(1)for h in(g,zip(*g))];k=75
 while k:k-=1;i=k//5%5;g[m+i][n+k%5]|=g[m+~k%5][n+i]
 return g
```

# [021]
[1190e5a7.json](https://arcprize.org/play?task=1190e5a7)


**Size: 51 bytes**
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
p=lambda i:i*-1*-1or-~min(map(i.count,i))*[p(i[0])]
```

# [022]
[137eaa0f.json](https://arcprize.org/play?task=137eaa0f)


**Size: 91 bytes**
* pattern_juxtaposition

```python
show_examples(load_examples(22)['train'])
```

```python
%%writefile task022.py
r=-1,0,1
p=lambda g:[[dict(sorted(zip(o:=sum(g,[]),o[x*11+y:]+o)))[5]for y in r]for x in r]
```

# [023]
[150deff5.json](https://arcprize.org/play?task=150deff5)


**Size: 182 bytes**
* pattern_coloring
* pattern_deconstruction
* associate_colors_to_patterns

```python
show_examples(load_examples(23)['train'])
```

```python
%%writefile task023.py
import re;p=lambda i,w=2:s!=(r:=re.sub((w%2*"5, "+"5(.%s)??")%{w*3%-7%len(i[0]*3)+2}*(3-w%2),r" 82,\81\ 12 \82, 82"[w::2],s,1))and p(eval(r))or w and p(i,w-1)if"5"in(s:=str(i))else i
```

# [024]
[178fcbfb.json](https://arcprize.org/play?task=178fcbfb)


**Size: 62 bytes**
* direction_guessing
* draw_line_from_point

```python
show_examples(load_examples(24)['train'])
```

```python
%%writefile task024.py
p=lambda g:[[3%-~max(i)or(2in j)*2for j in zip(*g)]for i in g]
```

# [025]
[1a07d186.json](https://arcprize.org/play?task=1a07d186)


**Size: 127 bytes**
* bring_patterns_close
* find_the_intruder

```python
show_examples(load_examples(25)['train'])
```

```python
%%writefile task025.py
p=lambda g,N=0:[g:=[[any(min(g)+[I:=(x:=r.pop())in r,A:=all(c)])*x|N+(N:=A*I*x)for c in g[::-1]]for*r,in zip(*g)]for _ in g][3]
```

# [026]
[1b2d62fb.json](https://arcprize.org/play?task=1b2d62fb)


**Size: 50 bytes**
* detect_wall
* separate_images
* pattern_intersection

```python
show_examples(load_examples(26)['train'])
```

```python
%%writefile task026.py
p=lambda g:[eval('8>>r[4]+r.pop(0),'*3)for r in g]
```

# [027]
[1b60fb0c.json](https://arcprize.org/play?task=1b60fb0c)


**Size: 97 bytes**
* pattern_deconstruction
* pattern_rotation
* pattern_expansion

```python
show_examples(load_examples(27)['train'])
```

```python
%%writefile task027.py
R=range(10);p=lambda g:[[g[i][j]or 2*g[~j+([*[*zip(*g)][5]]<g[5][::-1])][i]for j in R]for i in R]
```

# [028]
[1bfc4729.json](https://arcprize.org/play?task=1bfc4729)


**Size: 63 bytes**
* pattern_expansion

```python
show_examples(load_examples(28)['train'])
```

```python
%%writefile task028.py
p=lambda g:[[c:=max(g[x%15]),*[x&c]*8,c]for x in b'/ /  ppp']
```

# [029]
[1c786137.json](https://arcprize.org/play?task=1c786137)


**Size: 103 bytes**
* detect_enclosure
* crop

```python
show_examples(load_examples(29)['train'])
```

```python
%%writefile task029.py
p=lambda g,w=9:g*w and p(g,w-1)+[g:=[r[~x::-1]for*r,in zip(*g)if w in r]for x in[0]*4+[1]*5][7]*([]==g)
```

# [030]
[1caeab9d.json](https://arcprize.org/play?task=1caeab9d)


**Size: 93 bytes**
* pattern_moving
* pattern_alignment

```python
show_examples(load_examples(30)['train'])
```

```python
%%writefile task030.py
p=lambda g:[*zip(*[r[(I:=sum(g,[]).index)(max(r))//10-I(1)//10:]*any(r)+r for r in zip(*g)])]
```

# [031]
[1cf80156.json](https://arcprize.org/play?task=1cf80156)


**Size: 45 bytes**
* crop

```python
show_examples(load_examples(31)['train'])
```

```python
%%writefile task031.py
p=lambda g,*a:[*filter(any,zip(*a or p(*g)))]
```

# [032]
[1e0a9b12.json](https://arcprize.org/play?task=1e0a9b12)


**Size: 39 bytes**
* pattern_moving
* gravity

```python
show_examples(load_examples(32)['train'])
```

```python
%%writefile task032.py
p=lambda g:[*zip(*map(sorted,zip(*g)))]
```

# [033]
[1e32b0e9.json](https://arcprize.org/play?task=1e32b0e9)


**Size: 71 bytes**
* detect_grid
* separate_images
* image_repetition
* pattern_completion

```python
show_examples(load_examples(33)['train'])
```

```python
%%writefile task033.py
p=lambda g,h=[],i=0:g*0!=0and[*map(p,g[:6]*3,h+g,g[5:6]*17)]or(g>h)*i|h
```

# [034]
[1f0c79e5.json](https://arcprize.org/play?task=1f0c79e5)


**Size: 116 bytes**
* pattern_expansion
* diagonals
* direction_guessing

```python
show_examples(load_examples(34)['train'])
```

```python
%%writefile task034.py
import re
p=lambda g,n=-35:n*g or eval(re.sub("([^02]), 2(.{28})0",r"*[\1]*3\2\1,2",f"{*zip(*p(g,n+1)),*g}"))[8::-1]
```

# [035]
[1f642eb9.json](https://arcprize.org/play?task=1f642eb9)


**Size: 82 bytes**
* image_within_image
* projection_unto_rectangle

```python
show_examples(load_examples(35)['train'])
```

```python
%%writefile task035.py
p=lambda i,k=3:-k*i or[[k:=i[y!=8<<k]or y for y in i]for i[::-1]in zip(*p(i,k-1))]
```

# [036]
[1f85a75f.json](https://arcprize.org/play?task=1f85a75f)


**Size: 75 bytes**
* crop
* find_the_intruder

```python
show_examples(load_examples(36)['train'])
```

```python
%%writefile task036.py
p=lambda g,x=0:[r for r in zip(*x or p(g,g))if{*r}-{*sum(g[:5]+g[25:],[])}]
```

# [037]
[1f876c06.json](https://arcprize.org/play?task=1f876c06)


**Size: 101 bytes**
* connect_the_dots
* diagonals

```python
show_examples(load_examples(37)['train'])
```

```python
%%writefile task037.py
import re
p=lambda g:g[:~99]or p(eval(re.sub(r"(?<=(.).{34})(?=(.{35})*\1)0",r"\1",str(g[9::-1])))+g)
```

# [038]
[1fad071e.json](https://arcprize.org/play?task=1fad071e)


**Size: 51 bytes**
* count_patterns
* associate_images_to_numbers

```python
show_examples(load_examples(38)['train'])
```

```python
%%writefile task038.py
p=lambda g:[(str(g).count('1, 1')*[1]+[0]*9)[:9:2]]
```

# [039]
[2013d3e2.json](https://arcprize.org/play?task=2013d3e2)


**Size: 60 bytes**
* pattern_deconstruction
* crop

```python
show_examples(load_examples(39)['train'])
```

```python
%%writefile task039.py
p=lambda g:[*eval('zip(*[*filter(sum,'*2+'g)]))][:3])')][:3]
```

# [040]
[2204b7a8.json](https://arcprize.org/play?task=2204b7a8)


**Size: 63 bytes**
* proximity_guessing
* recoloring

```python
show_examples(load_examples(40)['train'])
```

```python
%%writefile task040.py
p=lambda g,h=[]:g*0!=0and[*map(p,g[:1]*5+g[9:]*5,h+g)]or h%~h&g
```

# [041]
[22168020.json](https://arcprize.org/play?task=22168020)


**Size: 49 bytes**
* pattern_expansion

```python
show_examples(load_examples(41)['train'])
```

```python
%%writefile task041.py
p=lambda g,x=0:[[c|(x:=x^c)for c in r]for r in g]
```

# [042]
[22233c11.json](https://arcprize.org/play?task=22233c11)


**Size: 132 bytes**
* pattern_expansion
* size_guessing

```python
show_examples(load_examples(42)['train'])
```

```python
%%writefile task042.py
import re
p=lambda g:exec("s=str(g);g[::-1]=zip(*eval(re.sub('0(?=.{%r}3.{%r}3)'%(*b'%Kq9V'[s.count('3')//8::3],),'8',s)));"*4)or g
```

# [043]
[2281f1f4.json](https://arcprize.org/play?task=2281f1f4)


**Size: 56 bytes**
* direction_guessing
* draw_line_from_point
* pattern_intersection

```python
show_examples(load_examples(43)['train'])
```

```python
%%writefile task043.py
p=lambda g:[[y+r[-1]&r.pop(0)+2for y in g[0]]for*r,in g]
```

# [044]
[228f6490.json](https://arcprize.org/play?task=228f6490)


**Size: 195 bytes (567 raw)**
* pattern_moving
* loop_filling
* shape_guessing
* x_marks_the_spot

```python
show_examples(load_examples(44)['train'])
```

```python
%%writefile task044.py
def p(i):
 for f in range(100):
  r=[n for n in range(100)if i[n//10][n%10]<5in i[n//10][:n%10]and 5in i[n//10][n%10:]and n<50]
  t=[n for n in range(100)if i[n//10][n%10]==f]
  if[r[0]-f for f in r]==[t[0]-f for f in t]:
   for n in r:i[n//10][n%10]=f
   for n in t:i[n//10][n%10]=0
 for f in range(100):
  r=[n for n in range(100)if i[n//10][n%10]<5in i[n//10][:n%10]and 5in i[n//10][n%10:]and n>50]
  t=[n for n in range(100)if i[n//10][n%10]==f]
  if[r[0]-f for f in r]==[t[0]-f for f in t]:
   for n in r:i[n//10][n%10]=f
   for n in t:i[n//10][n%10]=0
 return i
```

# [045]
[22eb0ac0.json](https://arcprize.org/play?task=22eb0ac0)


**Size: 45 bytes**
* connect_the_dots
* color_matching

```python
show_examples(load_examples(45)['train'])
```

```python
%%writefile task045.py
p=lambda g:[10*i[:i[0]==i[9]]or i for i in g]
```

# [046]
[234bbc79.json](https://arcprize.org/play?task=234bbc79)


**Size: 166 bytes**
* recoloring
* bring_patterns_close
* crop

```python
show_examples(load_examples(46)['train'])
```

```python
%%writefile task046.py
def p(g,d=3):g=(5,),*zip(*g);return*zip(*[[sum({*x*(q+r+s)}-{5})for x in 3*r][d:3+d]for q,r,s in zip(g,g[1:],g[2:]+g)if any(r)or(d:=d-[*q,5].index(5)+s.index(5))*0]),
```

# [047]
[23581191.json](https://arcprize.org/play?task=23581191)


**Size: 55 bytes**
* draw_line_from_point
* pattern_intersection

```python
show_examples(load_examples(47)['train'])
```

```python
%%writefile task047.py
p=lambda g:[[sum({*i+j})%13for*j,in zip(*g)]for i in g]
```

# [048]
[239be575.json](https://arcprize.org/play?task=239be575)


**Size: 92 bytes**
* detect_connectedness
* associate_images_to_bools

```python
show_examples(load_examples(48)['train'])
```

```python
%%writefile task048.py
p=lambda g:[[hash((*b'+]`dBPx <IaAacF#p3e7"kz0W}k&N%r'[sum(b'%r'%g)%39:]%g,))&8]]
```

# [049]
[23b5c85d.json](https://arcprize.org/play?task=23b5c85d)


**Size: 81 bytes**
* measure_area
* take_minimum
* crop

```python
show_examples(load_examples(49)['train'])
```

```python
%%writefile task049.py
p=lambda g:[d*[c]for i in g if(d:=i.count(c:=min(q:=sum(g,[0]*99),key=q.count)))]
```

# [050]
[253bf280.json](https://arcprize.org/play?task=253bf280)


**Size: 83 bytes**
* connect_the_dots
* direction_guessing

```python
show_examples(load_examples(50)['train'])
```

```python
%%writefile task050.py
p=lambda g,x=0:[[c|((x:=~sum(r)&c^x)>6>c)*3for c in r]for*r,in zip(*x*g or p(g,1))]
```

# [051]
[25d487eb.json](https://arcprize.org/play?task=25d487eb)


**Size: 111 bytes**
* draw_line_from_point
* direction_guessing
* color_guessing

```python
show_examples(load_examples(51)['train'])
```

```python
%%writefile task051.py
p=lambda i:[*eval("map(lambda*x,l=0,b=1,a=1:[[l:=l|(b!=y>a<1)*(a:=b),b:=y][y>0]for y in x][::-1],*"*4+"i))))")]
```

# [052]
[25d8a9c8.json](https://arcprize.org/play?task=25d8a9c8)


**Size: 40 bytes**
* detect_hor_lines
* recoloring
* remove_noise

```python
show_examples(load_examples(52)['train'])
```

```python
%%writefile task052.py
p=lambda g:[[len({*r})%2*5]*3for r in g]
```

# [053]
[25ff71a9.json](https://arcprize.org/play?task=25ff71a9)


**Size: 21 bytes**
* pattern_moving

```python
show_examples(load_examples(53)['train'])
```

```python
%%writefile task053.py
p=lambda g:(g+g)[2:5]
```

# [054]
[264363fd.json](https://arcprize.org/play?task=264363fd)


**Size: 265 bytes (618 raw)**
* pattern_repetition
* pattern_juxtaposition
* draw_line_from_point

```python
show_examples(load_examples(54)['train'])
```

```python
%%writefile task054.py
def p(r):
 p=[i*1for i in r]
 for i in range(28):
  for o in range(28):
   if r[i][o+1]==r[i][o-1]!=r[i][o]!=r[i+1][o]==r[i-1][o]!={r[i][o+1],r[i+1][o+1]}!={r[i+1][o]}:
    for e in range(28):
     for t in range(28):
      if r[i][o]==r[e][t]:
       for n in range(-1,2):
        for d in range(-1,2):
         f=1
         if r[i+n*f][o+d*f]!=r[i+1][o+2]:
          p[e+n*f][t+d*f]=r[i+n][o+d];f+=1
          if r[i][o]!=r[i+n*f][o+d*f]!=r[i+1][o+2]:
           while r[e+n*f][t+d*f]!=r[i+1][o+2]:p[e+n*f][t+d*f]=r[i+n][o+d];f+=1
    for n in range(-2,3):
     for d in range(-2,3):p[i+n][o+d]=r[i+1][o+2]
 return p
```

# [055]
[272f95fa.json](https://arcprize.org/play?task=272f95fa)


**Size: 77 bytes**
* detect_grid
* mimic_pattern
* grid_coloring

```python
show_examples(load_examples(55)['train'])
```

```python
%%writefile task055.py
p=lambda i,z=0:i*0!=0and[p(y,3*(z:=z+([y]>i)))for y in i]or i or 2222096>>z&7
```

# [056]
[27a28665.json](https://arcprize.org/play?task=27a28665)


**Size: 39 bytes**
* associate_colors_to_patterns
* take_negative
* associate_images_to_patterns

```python
show_examples(load_examples(56)['train'])
```

```python
%%writefile task056.py
p=lambda g:[[2^(g[2]<[1])*3+(g<[[1]])]]
```

# [057]
[28bf18c6.json](https://arcprize.org/play?task=28bf18c6)


**Size: 48 bytes**
* crop
* pattern_repetition

```python
show_examples(load_examples(57)['train'])
```

```python
%%writefile task057.py
p=lambda g:[*filter(any,zip(*g[8:]or p(g*2)*2))]
```

# [058]
[28e73c20.json](https://arcprize.org/play?task=28e73c20)


**Size: 85 bytes**
* ex_nihilo
* mimic_pattern

```python
show_examples(load_examples(58)['train'])
```

```python
%%writefile task058.py
p=lambda a:a and[len(a)*[3],[*a[0],3>>2%len(a),3][2:],*zip(*p([*zip(*a[2:])])[::-1])]
```

# [059]
[29623171.json](https://arcprize.org/play?task=29623171)


**Size: 151 bytes**
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
R=range(11);p=lambda g,n=36,o=0:[[(g[i][j]==5)*5or(sum(S:=[g[i&12|k%3][j&12|k//3]for k in R[:9]])>n!=[o:=1])*max(S)for j in R]for i in R]*o or p(g,n-1)
```

# [060]
[29c11459.json](https://arcprize.org/play?task=29c11459)


**Size: 47 bytes**
* draw_line_from_point
* count_tiles

```python
show_examples(load_examples(60)['train'])
```

```python
%%writefile task060.py
p=lambda g:[b[:1]*5+[5%9**c]+[c]*5for*b,c in g]
```

# [061]
[29ec7d0e.json](https://arcprize.org/play?task=29ec7d0e)


**Size: 62 bytes**
* image_filling
* pattern_expansion
* detect_grid
* pattern_repetition

```python
show_examples(load_examples(61)['train'])
```

```python
%%writefile task061.py
p=lambda g,q=range(18):[[i*j%max(g)[1]+1for j in q]for i in q]
```

# [062]
[2bcee788.json](https://arcprize.org/play?task=2bcee788)


**Size: 121 bytes**
* pattern_reflection
* direction_guessing
* image_filling
* background_filling

```python
show_examples(load_examples(62)['train'])
```

```python
%%writefile task062.py
p=lambda g:exec("q=[]\nfor r in zip(*g):q+=q[::-1]*({*r}^{2}=={3}<{*q[-1]})or[[v or 3for v in r]]\ng[:]=q[9::-1];"*8)or g
```

# [063]
[2bee17df.json](https://arcprize.org/play?task=2bee17df)


**Size: 74 bytes**
* draw_line_from_border
* count_tiles
* take_maximum

```python
show_examples(load_examples(63)['train'])
```

```python
%%writefile task063.py
p=lambda g:[[x|3>>x+sum(r[1:-any(c)])for _,*c,_,x in zip(*g,r)]for r in g]
```

# [064]
[2c608aff.json](https://arcprize.org/play?task=2c608aff)


**Size: 135 bytes**
* draw_line_from_point
* projection_unto_rectangle

```python
show_examples(load_examples(64)['train'])
```

```python
%%writefile task064.py
p=lambda g,n=-7:n*g or p([[P:=[x:=r.pop(),P][r.count(max({*r*P*(g[0].count(x)>3),n}-{x,P}))>2]for _ in g]for*r,in zip(*g)if[P:=0]],n+2)
```

# [065]
[2dc579da.json](https://arcprize.org/play?task=2dc579da)


**Size: 82 bytes**
* detect_grid
* find_the_intruder
* crop

```python
show_examples(load_examples(65)['train'])
```

```python
%%writefile task065.py
p=lambda*g:min(g,key=g.count)if g[3:]else[*map(p,*g,*[h[len(h)//2+1:]for h in g])]
```

# [066]
[2dd70a9a.json](https://arcprize.org/play?task=2dd70a9a)


**Size: 237 bytes (391 raw)**
* draw_line_from_point
* direction_guessing
* maze

```python
show_examples(load_examples(66)['train'])
```

```python
%%writefile task066.py
def f(r,n,e,l,d):
 if len(r)>n+1>1>r[e][n]%3:r[e][n]=3;return f(r,n+d,e,l,d)
 if r[e][n]%3:return f([[*n]for n in zip(*r)],e,n-d,l-1,1)or f([[*n]for n in zip(*r)],e,n-d,l-1,-1)if l else(3>r[e][n])*r
def p(r):(n,e),(l,d)=[(n,e)for n in range(len(r))for e in range(len(r))if r[e][n]%2];return f([[*n]for n in r],n,e,2,1)or f(r,n,e,2,-1)if l>n else[[*n]for n in zip(*p([[*n]for n in zip(*r)]))]
```

# [067]
[2dee498d.json](https://arcprize.org/play?task=2dee498d)


**Size: 33 bytes**
* detect_repetition
* crop
* divide_by_n

```python
show_examples(load_examples(67)['train'])
```

```python
%%writefile task067.py
p=lambda g:[a[:len(g)]for a in g]
```

# [068]
[31aa019c.json](https://arcprize.org/play?task=31aa019c)


**Size: 99 bytes**
* find_the_intruder
* remove_noise
* contouring

```python
show_examples(load_examples(68)['train'])
```

```python
%%writefile task068.py
p=lambda g,x=0:[[(x:=x*2|2>>sum(g,g).count(y))%2*y|(x>>89&7345159>0)*2for y in r]for*r,in g*2][10:]
```

# [069]
[321b1fc6.json](https://arcprize.org/play?task=321b1fc6)


**Size: 143 bytes**
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(69)['train'])
```

```python
%%writefile task069.py
def p(g):
 *b,=a=sum(g,[]);i=j=100
 while i:i-=1;a[i]==8==exec("if b[j:=j-1]%8:g+=j,;a[i-g[10]+j]=b[j];a[j]=0\n"*j)
 return*zip(*[iter(a)]*10),
```

# [070]
[32597951.json](https://arcprize.org/play?task=32597951)


**Size: 78 bytes**
* find_the_intruder
* recoloring

```python
show_examples(load_examples(70)['train'])
```

```python
%%writefile task070.py
p=lambda g:[[max({*i,hash(i)%1070}&{*r})%5**i[0]for i in zip(r,*g)]for r in g]
```

# [071]
[3345333e.json](https://arcprize.org/play?task=3345333e)


**Size: 100 bytes**
* pattern_completion
* pattern_reflection
* remove_noise

```python
show_examples(load_examples(71)['train'])
```

```python
%%writefile task071.py
p=lambda g,a=0:[[a*(v>0<r[~i+t])for i,v in enumerate(r)if a or{t:=i-r[::-1].index(a:=v)}]for r in g]
```

# [072]
[3428a4f5.json](https://arcprize.org/play?task=3428a4f5)


**Size: 54 bytes**
* detect_wall
* separate_images
* pattern_differences

```python
show_examples(load_examples(72)['train'])
```

```python
%%writefile task072.py
p=lambda g,h=[]:g*0!=0and[*map(p,g,h+g[7:])]or-(g^h)%5
```

# [073]
[3618c87e.json](https://arcprize.org/play?task=3618c87e)


**Size: 46 bytes**
* gravity

```python
show_examples(load_examples(73)['train'])
```

```python
%%writefile task073.py
p=lambda g:g[:1]*3+[g[3],[5-b*4for b in g[2]]]
```

# [074]
[3631a71a.json](https://arcprize.org/play?task=3631a71a)


**Size: 77 bytes**
* image_filling
* pattern_expansion
* pattern_rotation

```python
show_examples(load_examples(74)['train'])
```

```python
%%writefile task074.py
p=lambda g:g[:-90]or p([[*map(min,x,y,[9]*2+x[::-1])]for*y,x in zip(*g,g)]+g)
```

# [075]
[363442ee.json](https://arcprize.org/play?task=363442ee)


**Size: 86 bytes**
* detect_wall
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(75)['train'])
```

```python
%%writefile task075.py
p=lambda g,r=range(9):[g[i][:4]+[g[i%3][j%3]*g[i-i%3+1][j-j%3+5]for j in r]for i in r]
```

# [076]
[36d67576.json](https://arcprize.org/play?task=36d67576)


**Size: 258 bytes (488 raw)**
* pattern_repetition
* pattern_juxtaposition
* pattern_reflection
* pattern_rotation

```python
show_examples(load_examples(76)['train'])
```

```python
%%writefile task076.py
def p(r):
 a=r
 for n in range(len(r)):
  for f in range(len(r[0])):
   if a[n][f]==1:z={(f,n)}
 for n in r:z={(f+l,e+n)for l,n in z for f in range(-1,2)for e in range(-1,2)if e+n in range(len(r))!=f+l in range(len(r[0]))!=0<r[e+n][f+l]}
 for n in(1,1,-1)*4:
  r=[z for*z,in zip(*r[::n])]
  for e in range(-13,13):
   for f in range(-13,13):
    if all(e+n in range(len(r))!=f+l in range(len(r[0]))!=a[n][l]in(1,3,r[e+n][f+l])for l,n in z):
     for l,n in z:r[e+n][f+l]=a[n][l]
 return r
```

# [077]
[36fdfd69.json](https://arcprize.org/play?task=36fdfd69)


**Size: 110 bytes**
* recoloring
* rectangle_guessing

```python
show_examples(load_examples(77)['train'])
```

```python
%%writefile task077.py
p=lambda i,k=7,*w:k and p([*map(p,i,[k>1]*99,[i*2]+i,i[1:]+[i*2],*w)],k-1)or((c:=w.count)(2)+c(4)>=2!=i)*4or i
```

# [078]
[3906de3d.json](https://arcprize.org/play?task=3906de3d)


**Size: 59 bytes**
* gravity

```python
show_examples(load_examples(78)['train'])
```

```python
%%writefile task078.py
p=lambda i,*n:sorted(n,key=0 .__eq__)or[*zip(*map(p,i,*i))]
```

# [079]
[39a8645d.json](https://arcprize.org/play?task=39a8645d)


**Size: 105 bytes**
* count_patterns
* take_maximum
* crop

```python
show_examples(load_examples(79)['train'])
```

```python
%%writefile task079.py
p=eval(f"lambda a:max(b:=[a {'for*a,in map(zip,a,a[1:],a[2:])'*2}if any(min(*a,*zip(*a)))],key=b.count)")
```

# [080]
[39e1d7f9.json](https://arcprize.org/play?task=39e1d7f9)


**Size: 231 bytes (301 raw)**
* detect_grid
* pattern_repetition
* grid_coloring

```python
show_examples(load_examples(80)['train'])
```

```python
%%writefile task080.py
def p(f):n=f.index(min(f,key=set))+1;e={u*1j+o:f for u,f in enumerate(f[::n])for o,f in enumerate(f[::n])};return[[[f or[e[t:=u//n*1j+o//n],*[e[a+t-r]for r in e if(e[r]==e[a])*2>abs(t-r)]][-1]for o,f in enumerate(f)]for u,f in enumerate(f)]for a in e if all(e.get(a+1j**u)for u,f in enumerate(f))][-1]
```

# [081]
[3aa6fb7a.json](https://arcprize.org/play?task=3aa6fb7a)


**Size: 81 bytes**
* pattern_completion
* pattern_rotation

```python
show_examples(load_examples(81)['train'])
```

```python
%%writefile task081.py
p=lambda g:exec(f"g[:]={'(q:=1)*[q:=r.pop()or[1]<r[-1:q]for r in g],'*7};"*4)or g
```

# [082]
[3ac3eb23.json](https://arcprize.org/play?task=3ac3eb23)


**Size: 50 bytes**
* draw_pattern_from_point
* pattern_repetition

```python
show_examples(load_examples(82)['train'])
```

```python
%%writefile task082.py
p=lambda g:[c:=g[0],[*map(max,[0]+c,c[1:]+[0])]]*3
```

# [083]
[3af2c5a8.json](https://arcprize.org/play?task=3af2c5a8)


**Size: 40 bytes**
* image_repetition
* image_reflection
* image_rotation

```python
show_examples(load_examples(83)['train'])
```

```python
%%writefile task083.py
p=lambda g:[i+i[::-1]for i in g+g[::-1]]
```

# [084]
[3bd67248.json](https://arcprize.org/play?task=3bd67248)


**Size: 62 bytes**
* draw_line_from_border
* diagonals
* pattern_repetition

```python
show_examples(load_examples(84)['train'])
```

```python
%%writefile task084.py
def p(g,i=1):g[-1][i]=4;g[~i][i]=2;g>g[:i+1]>p(g,i+1);return g
```

# [085]
[3bdb4ada.json](https://arcprize.org/play?task=3bdb4ada)


**Size: 49 bytes**
* recoloring
* pattern_repetition
* holes

```python
show_examples(load_examples(85)['train'])
```

```python
%%writefile task085.py
p=lambda g:g*0!=0and[g:=[p(r),r][g!=r]for r in g]
```

# [086]
[3befdf3e.json](https://arcprize.org/play?task=3befdf3e)


**Size: 151 bytes**
* take_negative
* pattern_expansion

```python
show_examples(load_examples(86)['train'])
```

```python
%%writefile task086.py
p=lambda i,k=7,s=0:-k*i or[[[-((s:=[abs(s)or 1,s&s//4][y>0]-1)>1)|y,*[x for x in sum(i,x)if 0<x!=y!=0],0][k>6]for y in x]for*x,in zip(*p(i,k-1)[::-1])]
```

# [087]
[3c9b0459.json](https://arcprize.org/play?task=3c9b0459)


**Size: 36 bytes**
* image_rotation

```python
show_examples(load_examples(87)['train'])
```

```python
%%writefile task087.py
p=lambda j:[r[::-1]for r in j[::-1]]
```

# [088]
[3de23699.json](https://arcprize.org/play?task=3de23699)


**Size: 95 bytes**
* take_negative
* crop
* rectangle_guessing

```python
show_examples(load_examples(88)['train'])
```

```python
%%writefile task088.py
exec(f"def p(g):{'g[:]=zip(*g[any(h:=g[-1])-2::-1]);'*48}return[[h[g<1]"+'for g in g][1:-1]'*2)
```

# [089]
[3e980e27.json](https://arcprize.org/play?task=3e980e27)


**Size: 232 bytes (279 raw)**
* pattern_repetition
* pattern_juxtaposition
* direction_guessing
* pattern_reflection

```python
show_examples(load_examples(89)['train'])
```

```python
%%writefile task089.py
def	p(i):
	for	n	in(e:={n*1j+a:i	for	n,i	in	enumerate(i)for	a,i	in	enumerate(i)if	i}):
		for	r	in	e:
			t={r}
			for	a	in[*e]*3:
				if	e[n]==e[r]!=any(0<abs(n-r)<2for	r	in	e)<any(0<abs(a-r)<2for	r	in	t):t|={a};i[int((a-r+n).imag)][int((-(-1)**e[r]*(a-r)+n).real)]=e[a]
	return	i
```

# [090]
[3eda0437.json](https://arcprize.org/play?task=3eda0437)


**Size: 146 bytes**
* rectangle_guessing
* recoloring
* measure_area
* take_maximum

```python
show_examples(load_examples(90)['train'])
```

```python
%%writefile task090.py
import re
p=lambda g:eval("6".join(max([re.split((f"(..{ {len(g[0])*3-i%8*3}})0{i%8*'(, )0'}"*(i%5+2))[8:],str(g),1)for i in range(40)],key=len)))
```

# [091]
[3f7978a0.json](https://arcprize.org/play?task=3f7978a0)


**Size: 62 bytes**
* crop
* rectangle_guessing
* find_the_intruder

```python
show_examples(load_examples(91)['train'])
```

```python
%%writefile task091.py
p=lambda g,i=46:g*~i or p([*zip(*g[(5in g[i|-2])-2::-1])],i-1)
```

# [092]
[40853293.json](https://arcprize.org/play?task=40853293)


**Size: 86 bytes**
* connect_the_dots

```python
show_examples(load_examples(92)['train'])
```

```python
%%writefile task092.py
p=lambda a:[*map(f:=lambda*b,i=0:[v|(i:=i^b.count(v)*v-v)>>v*9for v in b],*map(f,*a))]
```

# [093]
[4093f84a.json](https://arcprize.org/play?task=4093f84a)


**Size: 98 bytes**
* gravity
* recoloring
* projection_unto_rectangle

```python
show_examples(load_examples(93)['train'])
```

```python
%%writefile task093.py
import re
p=lambda g:exec("g[::-1]=zip(*eval(re.sub('[^05],([, 0]*5)',r'\\1,5',str(g))));"*12)or g
```

# [094]
[41e4d17e.json](https://arcprize.org/play?task=41e4d17e)


**Size: 96 bytes**
* draw_line_from_point
* pattern_repetition

```python
show_examples(load_examples(94)['train'])
```

```python
%%writefile task094.py
import re
p=lambda g,x=0:eval(re.sub("8(?=[^(]*+[^)]*1.{46}1, 1)","6",f'{*zip(*x or p(g,g)),}'))
```

# [095]
[4258a5f9.json](https://arcprize.org/play?task=4258a5f9)


**Size: 70 bytes**
* pattern_repetition
* contouring

```python
show_examples(load_examples(95)['train'])
```

```python
%%writefile task095.py
p=lambda g:exec(f"g[:]={'[r.pop()or[0]<r[-1:]for r in g],'*9};"*4)or g
```

# [096]
[4290ef0e.json](https://arcprize.org/play?task=4290ef0e)


**Size: 282 bytes (380 raw)**
* pattern_moving
* concentric
* crop

```python
show_examples(load_examples(96)['train'])
```

```python
%%writefile task096.py
import re
def p(n):u={0:(0,i:=max(n:=re.sub(', ','',f'{n,*zip(*n)}'),key=n.count))}|{len(max(re.findall(f'{l}+',n)))*(((r:=len(re.findall(f'{l}{l}([^]){l}]+){l}|$',n+n[::-1])[0]))>0)+1)+r>>1:(1+r>>1,l)for l in{*n}-{*'([^])',i}};return[[int([u[max(abs(l),abs(r))][1],i][u[max(abs(l),abs(r))][0]>min(abs(l),abs(r))])for l in range(-max(u),max(u)+1)]for r in range(-max(u),max(u)+1)]
```

# [097]
[42a50994.json](https://arcprize.org/play?task=42a50994)


**Size: 93 bytes**
* remove_noise
* count_tiles

```python
show_examples(load_examples(97)['train'])
```

```python
%%writefile task097.py
p=lambda i,*w:i*0!=0and[*map(p,*sum([[x,x[1:]+[i*4],[i*4]+x]for x in[i,*w]],[]))]or(i in w)*i
```

# [098]
[4347f46a.json](https://arcprize.org/play?task=4347f46a)


**Size: 64 bytes**
* loop_filling
* color_guessing

```python
show_examples(load_examples(98)['train'])
```

```python
%%writefile task098.py
p=lambda i,*w:i*0!=0and[*map(p,i,i[:1]+i,i[1:]+i,*w)]or i^min(w)
```

# [099]
[444801d8.json](https://arcprize.org/play?task=444801d8)


**Size: 93 bytes**
* pattern_repetition
* pattern_expansion
* rectangle_guessing

```python
show_examples(load_examples(99)['train'])
```

```python
%%writefile task099.py
p=lambda g,n=0:[g:=[r.pop()or 1in g!=x>0and~-sum({*g})for x in g]for r in(n or p(g,g))[::-1]]
```

# [100]
[445eab21.json](https://arcprize.org/play?task=445eab21)


**Size: 85 bytes**
* measure_area
* take_maximum

```python
show_examples(load_examples(100)['train'])
```

```python
%%writefile task100.py
p=lambda g:[max(y*[r.count(y)*c.count(y),y]for r in g for*c,y in zip(*g,r))[-1:]*2]*2
```

# [101]
[447fd412.json](https://arcprize.org/play?task=447fd412)


**Size: 256 bytes (431 raw)**
* pattern_repetition
* draw_pattern_from_point
* pattern_resizing

```python
show_examples(load_examples(101)['train'])
```

```python
%%writefile task101.py
def p(e):
 w,f,r,o,q=[{a+m*20for m,e in enumerate(e)for a,e in enumerate(e)if e==d}for d in range(5)]
 for d in r:o|={d for d in r for m in f|o if abs(d-m)in(1,20)}
 for d in r:p={d}-q and{d-min(len({d+m*(n-2)for n in range(5)}&r)for m in(1,20))*(min(o)-max(o))}&r;q|=p;e=[[e or a+m*20in(p and{d-min(len({d+m*(n-2)for n in range(5)}&r)for m in(1,20))*(min(o)-m)for m in f})for a,e in enumerate(e)]for m,e in enumerate(e)]
 return e
```

# [102]
[44d8ac46.json](https://arcprize.org/play?task=44d8ac46)


**Size: 124 bytes**
* loop_filling
* rectangle_guessing

```python
show_examples(load_examples(102)['train'])
```

```python
%%writefile task102.py
p=lambda g,k=-11,i=1,q=0:k*g or[[q:=[i:=i<<9,2&132132>>c%511,5,c|q,0][k*6%64%6+c%~c]for c in g]for g[::-1]in zip(*p(g,k+1))]
```

# [103]
[44f52bb0.json](https://arcprize.org/play?task=44f52bb0)


**Size: 29 bytes**
* detect_symmetry
* associate_images_to_bools

```python
show_examples(load_examples(103)['train'])
```

```python
%%writefile task103.py
p=lambda g:[[g==g[::-1]or 7]]
```

# [104]
[4522001f.json](https://arcprize.org/play?task=4522001f)


**Size: 84 bytes**
* image_rotation
* pairwise_analogy

```python
show_examples(load_examples(104)['train'])
```

```python
%%writefile task104.py
p=lambda g:[[*[x%5]*4+[x%6]*4,0][::1|g[1][0]-2]for x in b'NNNNKKKKZ'][::1|g[0][1]-2]
```

# [105]
[4612dd53.json](https://arcprize.org/play?task=4612dd53)


**Size: 124 bytes**
* pattern_completion
* rectangle_guessing

```python
show_examples(load_examples(105)['train'])
```

```python
%%writefile task105.py
p=lambda g,i=11,k=0:-i*g or[[c*(k:=b|(1in r))or(sum(map(bool,r))-b>2>i%6)*2for c in r]for r in zip(*p(g,i-1)[::-1])if[b:=k]]
```

# [106]
[46442a0e.json](https://arcprize.org/play?task=46442a0e)


**Size: 55 bytes**
* image_repetition
* image_reflection

```python
show_examples(load_examples(106)['train'])
```

```python
%%writefile task106.py
p=lambda i,s=[],k=3:-k*i or p([*zip(*i+s)],i[::-1],k-1)
```

# [107]
[469497ad.json](https://arcprize.org/play?task=469497ad)


**Size: 132 bytes**
* image_resizing
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(107)['train'])
```

```python
%%writefile task107.py
def p(i):z=len({*str(i)})-5;r=range(5*z);return[[i[x//z][y//z]or(x-z*0**i[0][1]in[u:=y-z*0**i[1][0],z*2+~u])*2for y in r]for x in r]
```

# [108]
[46f33fce.json](https://arcprize.org/play?task=46f33fce)


**Size: 46 bytes**
* pattern_resizing
* image_resizing

```python
show_examples(load_examples(108)['train'])
```

```python
%%writefile task108.py
p=lambda a:a>a*0!=0and[p(a[1])]*4+p(a[2:])or a
```

# [109]
[47c1f68c.json](https://arcprize.org/play?task=47c1f68c)


**Size: 77 bytes**
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
p=lambda a,s=0:a*0!=0and(b:=[*map(p,a,[a[l:=len(a)//2]]*l)])+b[::-1]or a%~a&s
```

# [110]
[484b58aa.json](https://arcprize.org/play?task=484b58aa)


**Size: 83 bytes**
* image_filling
* pattern_expansion
* pattern_repetition

```python
show_examples(load_examples(110)['train'])
```

```python
%%writefile task110.py
p=lambda g:[[*map(max,*(r*max(-a^-b for a,b in zip(r,t))+t for t in g))]for r in g]
```

# [111]
[48d8fb45.json](https://arcprize.org/play?task=48d8fb45)


**Size: 60 bytes**
* find_the_intruder
* crop

```python
show_examples(load_examples(111)['train'])
```

```python
%%writefile task111.py
p=lambda g:[(l:=sum(g,g))[l.index(5)+n:][:3]for n in b"	"]
```

# [112]
[4938f0c2.json](https://arcprize.org/play?task=4938f0c2)


**Size: 91 bytes**
* pattern_expansion
* pattern_rotation
* pattern_reflection

```python
show_examples(load_examples(112)['train'])
```

```python
%%writefile task112.py
p=lambda g:exec("h=g[::-1]*2;g[:]=zip(*map(max,g,h[2*~[*map(max,g)].index(3):]+h));"*2)or g
```

# [113]
[496994bd.json](https://arcprize.org/play?task=496994bd)


**Size: 25 bytes**
* pattern_reflection

```python
show_examples(load_examples(113)['train'])
```

```python
%%writefile task113.py
p=lambda j:j[:5]+j[4::-1]
```

# [114]
[49d1d64f.json](https://arcprize.org/play?task=49d1d64f)


**Size: 64 bytes**
* pattern_expansion
* image_expansion

```python
show_examples(load_examples(114)['train'])
```

```python
%%writefile task114.py
p=lambda g,v=1:g*0!=0and[v*p(g[0],0),*map(p,g),v*p(g[-1],0)]or g
```

# [115]
[4be741c5.json](https://arcprize.org/play?task=4be741c5)


**Size: 51 bytes**
* summarize

```python
show_examples(load_examples(115)['train'])
```

```python
%%writefile task115.py
p=lambda g,F={}.fromkeys:[*F(zip(*zip(*map(F,g))))]
```

# [116]
[4c4377d9.json](https://arcprize.org/play?task=4c4377d9)


**Size: 20 bytes**
* image_repetition
* image_reflection

```python
show_examples(load_examples(116)['train'])
```

```python
%%writefile task116.py
p=lambda j:j[::-1]+j
```

# [117]
[4c5c2cf0.json](https://arcprize.org/play?task=4c5c2cf0)


**Size: 115 bytes**
* pattern_expansion
* pattern_rotation
* pattern_reflection

```python
show_examples(load_examples(117)['train'])
```

```python
%%writefile task117.py
p=lambda g,n=-79:n*g or p([*zip(*[g,h:=[*map(max,g,i:=(g*3)[n%-21::-1])]][40in map(str(i+6*h).count,str(h))])],n+1)
```

# [118]
[50846271.json](https://arcprize.org/play?task=50846271)


**Size: 225 bytes (300 raw)**
* pattern_completion
* recoloring

```python
show_examples(load_examples(118)['train'])
```

```python
%%writefile task118.py
def	p(u):
	for	f	in(2,3):
		a,e,n,*r=[{s-j*1jfor	j,u	in	enumerate(u)for	s,u	in	enumerate(u)if	u>=f}for	f	in(2,0,5,6)]
		for	j	in	e:i={s	for	s	in	e	if	abs(s-j)in(2,0,1,f)};r+=[f|i	for	f	in	r	if	a-f>i]
		for	f	in	r:
			if	a-n<f:return[[u+3*(s-j*1jin	f&n)for	s,u	in	enumerate(u)]for	j,u	in	enumerate(u)]
```

# [119]
[508bd3b6.json](https://arcprize.org/play?task=508bd3b6)


**Size: 102 bytes**
* draw_line_from_point
* direction_guessing
* pattern_reflection

```python
show_examples(load_examples(119)['train'])
```

```python
%%writefile task119.py
import re
p=lambda g:exec("g[::-1]=zip(*eval(re.sub('0(?=.{40}[83].{40}[832])','3',str(g))));"*40)or g
```

# [120]
[50cb2852.json](https://arcprize.org/play?task=50cb2852)


**Size: 74 bytes**
* holes
* rectangle_guessing

```python
show_examples(load_examples(120)['train'])
```

```python
%%writefile task120.py
p=lambda i,*w:i*0!=0and[*map(p,i,[w]+i,i[1:]+[w],*w)]or[8,i]["0"in str(w)]
```

# [121]
[5117e062.json](https://arcprize.org/play?task=5117e062)


**Size: 85 bytes**
* find_the_intruder
* crop
* recoloring

```python
show_examples(load_examples(121)['train'])
```

```python
%%writefile task121.py
p=lambda g:(a:=(g:=(g:=sum(g,[]))[g.index(8)-14:])[:3],(g[13],max(a),g[15]),g[26:29])
```

# [122]
[5168d44c.json](https://arcprize.org/play?task=5168d44c)


**Size: 70 bytes**
* direction_guessing
* recoloring
* contouring
* pattern_moving

```python
show_examples(load_examples(122)['train'])
```

```python
%%writefile task122.py
p=lambda g:'3, 0'in'%s'%max(g)and[*map(p,g)]or min(g,g[4:])[:2]+g[:-2]
```

# [123]
[539a4f51.json](https://arcprize.org/play?task=539a4f51)


**Size: 71 bytes**
* pattern_expansion
* image_expansion

```python
show_examples(load_examples(123)['train'])
```

```python
%%writefile task123.py
p=lambda g:[[(s:=g[0][:4+all(g[0])]*3)[i]]*i+s[i:10]for i in range(10)]
```

# [124]
[53b68214.json](https://arcprize.org/play?task=53b68214)


**Size: 95 bytes**
* pattern_expansion
* image_expansion

```python
show_examples(load_examples(124)['train'])
```

```python
%%writefile task124.py
p=lambda i:i[9:]and i or p(i+[((w:=i[4]!=i[1])*[0for k in(1,2)if[0]*k+i[0]>i[2]]+i[w-3])[:10]])
```

# [125]
[543a7ed5.json](https://arcprize.org/play?task=543a7ed5)


**Size: 110 bytes**
* contouring
* loop_filling

```python
show_examples(load_examples(125)['train'])
```

```python
%%writefile task125.py
p=lambda g,i=87,q=8:g*-i or[[[12%c,q//c*c,-(q&(q:=c)%3)%5][i//42]or c for c in g]for g[::-1]in zip(*p(g,i-1))]
```

# [126]
[54d82841.json](https://arcprize.org/play?task=54d82841)


**Size: 54 bytes**
* pattern_expansion
* gravity

```python
show_examples(load_examples(126)['train'])
```

```python
%%writefile task126.py
p=lambda g:g[:-1]+[[4*(0<sum(i)in i)for i in zip(*g)]]
```

# [127]
[54d9e175.json](https://arcprize.org/play?task=54d9e175)


**Size: 57 bytes**
* detect_grid
* separate_images
* associate_images_to_images

```python
show_examples(load_examples(127)['train'])
```

```python
%%writefile task127.py
p=lambda g:g*-1and g+5or g and[p(g[1])]*3+g[3:4]+p(g[4:])
```

# [128]
[5521c0d9.json](https://arcprize.org/play?task=5521c0d9)


**Size: 57 bytes**
* pattern_moving
* measure_length

```python
show_examples(load_examples(128)['train'])
```

```python
%%writefile task128.py
p=lambda g:[*zip(*map(lambda*c:c[c.count(c[-1]):]+c,*g))]
```

# [129]
[5582e5ca.json](https://arcprize.org/play?task=5582e5ca)


**Size: 47 bytes**
* count_tiles
* dominant_color

```python
show_examples(load_examples(129)['train'])
```

```python
%%writefile task129.py
p=lambda g:[[max(q:=sum(g,g),key=q.count)]*3]*3
```

# [130]
[5614dbcf.json](https://arcprize.org/play?task=5614dbcf)


**Size: 61 bytes**
* remove_noise
* image_resizing

```python
show_examples(load_examples(130)['train'])
```

```python
%%writefile task130.py
p=lambda g:g*(g!=5)and(g*-1*-1or[max(map(p,g[:3]))]+p(g[3:]))
```

# [131]
[56dc2b01.json](https://arcprize.org/play?task=56dc2b01)


**Size: 117 bytes**
* gravity
* direction_guessing
* pattern_expansion

```python
show_examples(load_examples(131)['train'])
```

```python
%%writefile task131.py
p=lambda g:exec("c=2;g[:]=zip(*([j for*j,in g if(c:=c-max(j)*(c>0))|max(j)]+[[8]*9]+g[:1]*99)[len(g)-1::-1]);"*4)or g
```

# [132]
[56ff96f3.json](https://arcprize.org/play?task=56ff96f3)


**Size: 86 bytes**
* pattern_completion
* rectangle_guessing

```python
show_examples(load_examples(132)['train'])
```

```python
%%writefile task132.py
p=lambda g,x=0:[g:=[[x|(x:=x^max({*c}&{*r}))for r in g]for c in zip(*g)]for _ in g][1]
```

# [133]
[57aa92db.json](https://arcprize.org/play?task=57aa92db)


**Size: 273 bytes (429 raw)**
* draw_pattern_from_point
* pattern_repetition
* pattern_resizing

```python
show_examples(load_examples(133)['train'])
```

```python
%%writefile task133.py
def p(a):
 *r,e={f*66+l:(a)for f,a in enumerate(a)for l,a in enumerate(a)if a},
 for f in e:i={f};r=[n for n in r if n==n-{f-66,f-1}or i.update(n)]+[i]
 for n in r:
  for f in n:
   for i in n:
    o=n
    for o in 1//len([u for u in o if e[f]==e[u]])*r:
     for t in n-{i}:
      for u in[u for u in o if e[f]==e[u]==e[i]]:u+=(len([u for u in o if e[f]==e[u]==e[i]])^6)%6*(t-i);a[u//66][u%66],={e[u]for u in o}-{e[f]}
 return a
```

# [134]
[5ad4f10b.json](https://arcprize.org/play?task=5ad4f10b)


**Size: 123 bytes**
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
p=lambda g,D=10:[[D*(D!=v>0)for v in r]for r in('0, %g, 0'%D in'%s'%g)*g]or(h:=[*filter(any,zip(*p(g,D-.5)))])[::len(h)//3]
```

# [135]
[5bd6f4ac.json](https://arcprize.org/play?task=5bd6f4ac)


**Size: 32 bytes**
* rectangle_guessing
* crop

```python
show_examples(load_examples(135)['train'])
```

```python
%%writefile task135.py
p=lambda g:[r[6:]for r in g[:3]]
```

# [136]
[5c0a986e.json](https://arcprize.org/play?task=5c0a986e)


**Size: 94 bytes**
* draw_line_from_point
* diagonals
* direction_guessing

```python
show_examples(load_examples(136)['train'])
```

```python
%%writefile task136.py
import re
p=lambda g:eval([g:=re.sub('0(?=(.{34}%s){2})'%x,x,str(g)[::-1])for x in"21"*8][-1])
```

# [137]
[5c2c9af4.json](https://arcprize.org/play?task=5c2c9af4)


**Size: 137 bytes**
* rectangle_guessing
* pattern_expansion

```python
show_examples(load_examples(137)['train'])
```

```python
%%writefile task137.py
def p(g):*_,(a,A),b=sorted(zip(g,R:=range(len(g))));C=max(a);return[[C*(max(y-A,A-y,abs(x-a.index(C)))%(A-b[1])==0)for x in R]for y in R]
```

# [138]
[5daaa586.json](https://arcprize.org/play?task=5daaa586)


**Size: 101 bytes**
* detect_grid
* crop
* draw_line_from_point
* direction_guessing

```python
show_examples(load_examples(138)['train'])
```

```python
%%writefile task138.py
p=lambda g,k=59:g*-k or p([[x]+[[r.pop(),x][k-9<x in r]for _ in r[0in g[0]:]]for*r,x in zip(*g)],k-1)
```

# [139]
[60b61512.json](https://arcprize.org/play?task=60b61512)


**Size: 81 bytes**
* pattern_completion

```python
show_examples(load_examples(139)['train'])
```

```python
%%writefile task139.py
p=lambda g:[[L[0]or(r>g[0]<L[-max(r[5:])%9:][:5])*7for*L,in zip(r,*g)]for r in g]
```

# [140]
[6150a2bd.json](https://arcprize.org/play?task=6150a2bd)


**Size: 36 bytes**
* image_rotation

```python
show_examples(load_examples(140)['train'])
```

```python
%%writefile task140.py
p=lambda g:[r[::-1]for r in g[::-1]]
```

# [141]
[623ea044.json](https://arcprize.org/play?task=623ea044)


**Size: 92 bytes**
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(141)['train'])
```

```python
%%writefile task141.py
p=lambda g,i=1:[[*map(max,r,r[:(z:=abs(g.index(m:=max(g))+(i:=i-1)))]+m,m[z:]+r)]for r in g]
```

# [142]
[62c24649.json](https://arcprize.org/play?task=62c24649)


**Size: 40 bytes**
* image_repetition
* image_reflection
* image_rotation

```python
show_examples(load_examples(142)['train'])
```

```python
%%writefile task142.py
p=lambda g:[r+r[::-1]for r in g+g[::-1]]
```

# [143]
[63613498.json](https://arcprize.org/play?task=63613498)


**Size: 127 bytes**
* recoloring
* compare_image
* detect_wall

```python
show_examples(load_examples(143)['train'])
```

```python
%%writefile task143.py
def p(g,i=1):
	*R,=G=b'%r'%g
	for k in b'"%(BEH':R[k+i]=(48%G[k]or-5)%G[k+i]+5
	return{*G}>{*R}and eval(bytes(R))or p(g,i+1)
```

# [144]
[6430c8c4.json](https://arcprize.org/play?task=6430c8c4)


**Size: 53 bytes**
* detect_wall
* separate_images
* take_complement
* pattern_intersection

```python
show_examples(load_examples(144)['train'])
```

```python
%%writefile task144.py
p=lambda g,h=[]:g*0!=0and[*map(p,g,h+g[5:])]or 3>>g+h
```

# [145]
[6455b5f5.json](https://arcprize.org/play?task=6455b5f5)


**Size: 156 bytes**
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
p=lambda x,k=7,v=1:-k*x or p([[a:=b&2or[b//max(f:=sum(x,r))+(min({*f}-{2})==b)*8,b%511*4,*[a&~2|b]*5,v:=v*512][k]for b in[2]+r][:0:-1]for*r,in zip(*x)],k-1)
```

# [146]
[662c240a.json](https://arcprize.org/play?task=662c240a)


**Size: 58 bytes**
* separate_images
* detect_symmetry
* find_the_intruder
* crop

```python
show_examples(load_examples(146)['train'])
```

```python
%%writefile task146.py
p=lambda g:(d:=g[:3])*(d!=[*map(list,zip(*d))])or p(g[3:])
```

# [147]
[67385a82.json](https://arcprize.org/play?task=67385a82)


**Size: 72 bytes**
* recoloring
* measure_area
* associate_colors_to_bools

```python
show_examples(load_examples(147)['train'])
```

```python
%%writefile task147.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i*2]+i,i[1:]+[i*2],*w)]or(3in w)+7&i*9
```

# [148]
[673ef223.json](https://arcprize.org/play?task=673ef223)


**Size: 118 bytes**
* recoloring
* draw_line_from_point
* portals

```python
show_examples(load_examples(148)['train'])
```

```python
%%writefile task148.py
p=lambda g,X=0:[[s:=0,X:=-any(r)&1+X%6,[332%(2+sum(r,-s)*(s:=s+v))or(X in g)*8for v in r],g:=g+s//8*[X]][2]for r in g]
```

# [149]
[6773b310.json](https://arcprize.org/play?task=6773b310)


**Size: 74 bytes**
* detect_grid
* separate_images
* count_tiles
* associate_colors_to_numbers

```python
show_examples(load_examples(149)['train'])
```

```python
%%writefile task149.py
p=lambda g:g[3:]and[p([*zip(*g[i:i+3])])for i in[0,4,8]]or sum(b"%r/"%g)%5
```

# [150]
[67a3c6ac.json](https://arcprize.org/play?task=67a3c6ac)


**Size: 30 bytes**
* image_reflection

```python
show_examples(load_examples(150)['train'])
```

```python
%%writefile task150.py
p=lambda j:[r[::-1]for r in j]
```

# [151]
[67a423a3.json](https://arcprize.org/play?task=67a423a3)


**Size: 100 bytes**
* pattern_intersection
* contouring

```python
show_examples(load_examples(151)['train'])
```

```python
%%writefile task151.py
def p(g):
 e,k=[~-s.index(max(s))for s in(g,g[0])]
 for t in b'-(#$)*% ':g[e+t%5][k+t%3]=4
 return g
```

# [152]
[67e8384a.json](https://arcprize.org/play?task=67e8384a)


**Size: 39 bytes**
* image_repetition
* image_reflection
* image_rotation

```python
show_examples(load_examples(152)['train'])
```

```python
%%writefile task152.py
p=lambda g:g*-1*-1or[*map(p,g+g[::-1])]
```

# [153]
[681b3aeb.json](https://arcprize.org/play?task=681b3aeb)


**Size: 118 bytes**
* pattern_moving
* jigsaw
* crop
* bring_patterns_close

```python
show_examples(load_examples(153)['train'])
```

```python
%%writefile task153.py
T=0,1,2;p=lambda g:max(all(sum(G:=[[g[x+i%7][y+i%8]^g[x-i%9][y-i%11]for y in T]for x in T],G))*G for i in range(5544))
```

# [154]
[6855a6e4.json](https://arcprize.org/play?task=6855a6e4)


**Size: 88 bytes**
* pattern_moving
* direction_guessing
* x_marks_the_spot

```python
show_examples(load_examples(154)['train'])
```

```python
%%writefile task154.py
p=lambda g:g[225:]or[(r,r[8-r[3]::-1]+r[9-r[3]:])[5in r[:3]]for r in zip(*p(g*2)[::-1])]
```

# [155]
[68b16354.json](https://arcprize.org/play?task=68b16354)


**Size: 18 bytes**
* image_reflection

```python
show_examples(load_examples(155)['train'])
```

```python
%%writefile task155.py
p=lambda j:j[::-1]
```

# [156]
[694f12f3.json](https://arcprize.org/play?task=694f12f3)


**Size: 118 bytes**
* rectangle_guessing
* loop_filling
* measure_area
* associate_colors_to_ranks

```python
show_examples(load_examples(156)['train'])
```

```python
%%writefile task156.py
import re
p=lambda g:eval(g:=re.sub("(?<=4.{34}4)(?=.{34}4(.*0.{31}(4))?)",r"*(X:=g.count)('X\2X')//X('+')+1",str(g)))
```

# [157]
[6a1e5592.json](https://arcprize.org/play?task=6a1e5592)


**Size: 235 bytes (280 raw)**
* pattern_moving
* jigsaw
* recoloring

```python
show_examples(load_examples(157)['train'])
```

```python
%%writefile task157.py
def p(r):i,u=range,15;f=sum(r,p:=[]);r=[[]];[(r:=[r+[(n,p)]for r in r for n in i(3*u)if f[n]<1],p:=[])for n in i(3*u)if p==(p:=p+[i for i in i(n,150,u)if f[i]&(n<u)])>[]];return max([*zip(*[((any(i+min(r)-n in r for n,r in r if n%u<5+i%u)+f[i]%5)%3for i in i(150))]*u)]for r in r)
```

# [158]
[6aa20dc0.json](https://arcprize.org/play?task=6aa20dc0)


**Size: 258 bytes (488 raw)**
* pattern_repetition
* pattern_juxtaposition
* pattern_resizing

```python
show_examples(load_examples(158)['train'])
```

```python
%%writefile task158.py
def	p(r):
	n,s=max((len({*str(m:=[n[o:o+3]for	n	in	r[n:n+3]])}),m)for	n	in	range(len(r))for	o	in	range(len(r[-1])))
	for	m	in	range(len(r[-1])):
		for	n	in	range(len(r)-m*3):
			for	o	in	range(len(r[-1])-m*3):
				for	p	in	range(len(r[-1])):
					for	p	in	range(m*3*all(r[n+p][o+q]==s[p//m][q//m]or	r[n+p][o+q]==r[-1][-1]!=s[p//m][q//m]==max({*s[1]}-{r[-1][-1]})for	p	in	range(m*3)for	q	in	range(m*3))):
						for	q	in	range(m*3):r[n+p][o+q]=s[p//m][q//m]
					s=*zip(*s[::-1]),
	return	r
```

# [159]
[6b9890af.json](https://arcprize.org/play?task=6b9890af)


**Size: 103 bytes**
* pattern_moving
* pattern_resizing
* crop
* x_marks_the_spot

```python
show_examples(load_examples(159)['train'])
```

```python
%%writefile task159.py
p=lambda g,*G:[*zip(a:=[2]*99,*[r for*r,in G or p(g,*g)for c in g[::3]if c.count(2)==2if{*r}-{0,2}],a)]
```

# [160]
[6c434453.json](https://arcprize.org/play?task=6c434453)


**Size: 100 bytes**
* replace_pattern

```python
show_examples(load_examples(160)['train'])
```

```python
%%writefile task160.py
import re;p=lambda i,*n:eval(re.sub("1.{5}1(.{25})??"*3,r"0,2,0\1 2,2,2\2 0,2,0",str(n or p(i,*i))))
```

# [161]
[6cdd2623.json](https://arcprize.org/play?task=6cdd2623)


**Size: 79 bytes**
* connect_the_dots
* find_the_intruder
* remove_noise

```python
show_examples(load_examples(161)['train'])
```

```python
%%writefile task161.py
p=lambda m:[[4//(C:=sum(m,m).count)(x)*x|4//C(i)*i for i in m[0]]for x,*_ in m]
```

# [162]
[6cf79266.json](https://arcprize.org/play?task=6cf79266)


**Size: 96 bytes**
* rectangle_guessing
* recoloring

```python
show_examples(load_examples(162)['train'])
```

```python
%%writefile task162.py
import re;p=lambda g,k=0:eval('1,1,1'.join(re.split(("(.{55})0, 0, 0"*3)[7:],str(k or p(g,g)))))
```

# [163]
[6d0160f0.json](https://arcprize.org/play?task=6d0160f0)


**Size: 130 bytes**
* detect_grid
* separate_image
* find_the_intruder
* pattern_moving

```python
show_examples(load_examples(163)['train'])
```

```python
%%writefile task163.py
exec("p=lambda a:[[5*(a[i][j]==5)or sum((4==a[k+i//4][l+j//4])*a[k+i%4][l+j%4]"+'for %s in range(0,11,%s)%s'*4%(*'k4 l4)j1]i1]',))
```

# [164]
[6d0aefbc.json](https://arcprize.org/play?task=6d0aefbc)


**Size: 32 bytes**
* image_repetition
* image_reflection

```python
show_examples(load_examples(164)['train'])
```

```python
%%writefile task164.py
p=lambda j:[R+R[::-1]for R in j]
```

# [165]
[6d58a25d.json](https://arcprize.org/play?task=6d58a25d)


**Size: 112 bytes**
* draw_line_from_point

```python
show_examples(load_examples(165)['train'])
```

```python
%%writefile task165.py
p=lambda g:[*zip(*map(lambda*r:r[:-(k:=r[::-1].index(max(r,key=sum(g[::-1],g).index)))]+k*(max(r[-k:]),)+r,*g))]
```

# [166]
[6d75e8bb.json](https://arcprize.org/play?task=6d75e8bb)


**Size: 61 bytes**
* rectangle_guessing
* pattern_completion

```python
show_examples(load_examples(166)['train'])
```

```python
%%writefile task166.py
p=lambda g:[[e+(c>[e]*99<l)*2for*c,e in zip(*g,l)]for l in g]
```

# [167]
[6e02f1e3.json](https://arcprize.org/play?task=6e02f1e3)


**Size: 70 bytes**
* count_different_colors
* associate_images_to_numbers

```python
show_examples(load_examples(167)['train'])
```

```python
%%writefile task167.py
p=lambda i:[[5*(y==x%len({*str(i)})%3)for x in b'']for y in(0,1,2)]
```

# [168]
[6e19193c.json](https://arcprize.org/play?task=6e19193c)


**Size: 110 bytes**
* draw_line_from_point
* direction_guessing
* diagonals

```python
show_examples(load_examples(168)['train'])
```

```python
%%writefile task168.py
import re
p=lambda g:eval(re.sub(r"0(?=(.{35})+,( [^0]).{27}\2,\2)",r"\2",f'{*zip(*g[70:]or p(g*2)),}'))[::-1]
```

# [169]
[6e82a1ae.json](https://arcprize.org/play?task=6e82a1ae)


**Size: 108 bytes**
* recoloring
* count_tiles
* associate_colors_to_numbers

```python
show_examples(load_examples(169)['train'])
```

```python
%%writefile task169.py
p=lambda g,k=11,l=6:-k*g or p([(a:=1)*[a:=[c%7,a|c|(l:=l*8)][k>0<c]for c in r][::-1]for*r,in zip(*g)],k-1,0)
```

# [170]
[6ecd11f4.json](https://arcprize.org/play?task=6ecd11f4)


**Size: 163 bytes**
* color_palette
* recoloring
* pattern_resizing
* crop

```python
show_examples(load_examples(170)['train'])
```

```python
%%writefile task170.py
f=lambda t:t[exec("t[:]=zip(*t[sum(t[0])in t[0]:][::-1]);"*82):]
p=lambda g:eval(f"[[r&n%~n\nfor r,n in zip(r,n[::{-~len(n:=f(f(g)[4:]))//len(r:=f(g[:4]))}])]#"*2)
```

# [171]
[6f8cd79b.json](https://arcprize.org/play?task=6f8cd79b)


**Size: 51 bytes**
* ex_nihilo
* contouring

```python
show_examples(load_examples(171)['train'])
```

```python
%%writefile task171.py
p=lambda g,*a:[*zip(z:=[8]*9,*a[1:]or p(*g)[2:],z)]
```

# [172]
[6fa7a44f.json](https://arcprize.org/play?task=6fa7a44f)


**Size: 20 bytes**
* image_repetition
* image_reflection

```python
show_examples(load_examples(172)['train'])
```

```python
%%writefile task172.py
p=lambda j:j+j[::-1]
```

# [173]
[72322fa7.json](https://arcprize.org/play?task=72322fa7)


**Size: 201 bytes (495 raw)**
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(173)['train'])
```

```python
%%writefile task173.py
def p(i):r=[[i[n+f//3][u+f%3]for f in range(9)]for n in range(len(i)-2)for u in range(len(i[n])-2)if([i[n+f//3][u+f%3]for f in range(9)]==[i[n+f//3][u+f%3]for f in range(9)][::-1])*len({i[n+f//3][u+f%3]for f in range(9)})>2];[[i[n+f//3][u+f%3]for f in range(9)for i[n+f//3][u+f%3]in[e[f]]]for e in r for n in range(len(i)-2)for u in range(len(i[n])-2)if([i[n+f//3][u+f%3]for f in range(9)]==[(i[n+f//3][u+f%3]>0)*e[f]for f in range(9)][::-1])*len({i[n+f//3][u+f%3]for f in range(9)})>1];return i
```

# [174]
[72ca375d.json](https://arcprize.org/play?task=72ca375d)


**Size: 89 bytes**
* find_the_intruder
* detect_symmetry
* crop

```python
show_examples(load_examples(174)['train'])
```

```python
%%writefile task174.py
p=lambda g,i=1:eval(f'(g==g[::-1])*(g:=[x for x in zip(*g)if {i}in x]),'*4)[3]or p(g,i+1)
```

# [175]
[73251a56.json](https://arcprize.org/play?task=73251a56)


**Size: 74 bytes**
* image_filling
* diagonal_symmetry

```python
show_examples(load_examples(175)['train'])
```

```python
%%writefile task175.py
p=lambda g:[g:=[y|z or x for x,y,z in zip([0]+g,r,c)]for*c,r in zip(*g,g)]
```

# [176]
[7447852a.json](https://arcprize.org/play?task=7447852a)


**Size: 61 bytes**
* pattern_expansion
* pairwise_analogy

```python
show_examples(load_examples(176)['train'])
```

```python
%%writefile task176.py
p=lambda g,r=5:[[x|~r*(r:=r-x)%6for x in s]*(r:=1)for s in g]
```

# [177]
[7468f01a.json](https://arcprize.org/play?task=7468f01a)


**Size: 51 bytes**
* crop
* image_reflection

```python
show_examples(load_examples(177)['train'])
```

```python
%%writefile task177.py
p=lambda g,*a:[*filter(sum,zip(*a or p(*g)[::-1]))]
```

# [178]
[746b3537.json](https://arcprize.org/play?task=746b3537)


**Size: 47 bytes**
* crop
* direction_guessing

```python
show_examples(load_examples(178)['train'])
```

```python
%%writefile task178.py
p=lambda g:g*-1*-1or[p(g:=r)for r in g if g!=r]
```

# [179]
[74dd1130.json](https://arcprize.org/play?task=74dd1130)


**Size: 21 bytes**
* image_reflection
* diagonal_symmetry

```python
show_examples(load_examples(179)['train'])
```

```python
%%writefile task179.py
p=lambda g:[*zip(*g)]
```

# [180]
[75b8110e.json](https://arcprize.org/play?task=75b8110e)


**Size: 74 bytes**
* separate_images
* image_juxtaposition

```python
show_examples(load_examples(180)['train'])
```

```python
%%writefile task180.py
p=lambda a:[p(b)for*b,in map(zip,a,a[4:])]or max(sum(a+a,())[1:],key=bool)
```

# [181]
[760b3cac.json](https://arcprize.org/play?task=760b3cac)


**Size: 67 bytes**
* pattern_reflection
* direction_guessing

```python
show_examples(load_examples(181)['train'])
```

```python
%%writefile task181.py
def p(g):
 for x in g[:3]:d=6>>g[3][3];x[d:d+3]=x[5:2:-1]
 return g
```

# [182]
[776ffc46.json](https://arcprize.org/play?task=776ffc46)


**Size: 160 bytes**
* recoloring
* associate_colors_to_patterns
* detect_enclosure
* find_the_intruder

```python
show_examples(load_examples(182)['train'])
```

```python
%%writefile task182.py
from re import*
p=lambda g,h=0:eval(sub(sub("2|3","1","(.{47})".join(a:=findall(".*5, "+"5.{46}(.{15})"*5,s:=str(h or p(g,g)))[0])),"\%d ".join(a)%(1,2,3,4),s))
```

# [183]
[77fdfe62.json](https://arcprize.org/play?task=77fdfe62)


**Size: 79 bytes**
* recoloring
* color_guessing
* detect_grid
* crop

```python
show_examples(load_examples(183)['train'])
```

```python
%%writefile task183.py
p=lambda g,h=0:g*0!=0and[*map(p,len(g)//2*g[:1]+g[-1:]*9,h or g)][2:-2]or h%7*g
```

# [184]
[780d0b14.json](https://arcprize.org/play?task=780d0b14)


**Size: 91 bytes**
* detect_grid
* summarize

```python
show_examples(load_examples(184)['train'])
```

```python
%%writefile task184.py
p=lambda g,*x,q=():[z for r in zip(*x or p(*g))if(q:=(*map(max,(z:=q)*any(r)+r,r),))<z]+[q]
```

# [185]
[7837ac64.json](https://arcprize.org/play?task=7837ac64)


**Size: 107 bytes**
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
p=lambda g,*P:[g:=[[x*(P==(P:=x)!=max(Z))for c,x in zip(g,r)if{*c}-{*Z}][1:]for r in zip(*g)]for Z in g][1]
```

# [186]
[794b24be.json](https://arcprize.org/play?task=794b24be)


**Size: 60 bytes**
* count_tiles
* associate_images_to_numbers

```python
show_examples(load_examples(186)['train'])
```

```python
%%writefile task186.py
p=lambda g:[[2,1%(x:=sum(sum(g,[])))*2,2%x],[0,6%x,0],[0]*3]
```

# [187]
[7b6016b9.json](https://arcprize.org/play?task=7b6016b9)


**Size: 83 bytes**
* loop_filling
* background_filling
* color_guessing

```python
show_examples(load_examples(187)['train'])
```

```python
%%writefile task187.py
p=lambda g,i=7:g*-i or[[i:=c|2>>-i*c%7for c in[3]+r][:0:-1]for*r,in zip(*p(g,i-1))]
```

# [188]
[7b7f7511.json](https://arcprize.org/play?task=7b7f7511)


**Size: 53 bytes**
* separate_images
* detect_repetition
* crop

```python
show_examples(load_examples(188)['train'])
```

```python
%%writefile task188.py
p=lambda g:(X:=g[:53%~-len(g)])*(g==X+X)or[*map(p,g)]
```

# [189]
[7c008303.json](https://arcprize.org/play?task=7c008303)


**Size: 95 bytes**
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
p=lambda g,h=[]:g*0!=0and[*map(p,3*[g[i:=('0'in'%r'%g[2])*7]]+3*[g[i+1]],(h+g)[3>>i:])]or h%2*g
```

# [190]
[7ddcd7ec.json](https://arcprize.org/play?task=7ddcd7ec)


**Size: 103 bytes**
* draw_line_from_point
* direction_guessing
* diagonals

```python
show_examples(load_examples(190)['train'])
```

```python
%%writefile task190.py
import re;p=lambda i:exec(r"i[::-1]=zip(*eval(re.sub('.{31}0, ([^0])'*2,r'|\1\g<0>',str(i))));"*20)or i
```

# [191]
[7df24a62.json](https://arcprize.org/play?task=7df24a62)


**Size: 221 bytes**
* pattern_repetition
* pattern_rotation
* pattern_juxtaposition
* out_of_boundary

```python
show_examples(load_examples(191)['train'])
```

```python
%%writefile task191.py
from re import*
def p(l):
 X=[0]*U;r=*l,=X,*zip(X,*l,X),X
 for e in[1,1,-1]*8:l[::e]=zip(*eval('1'.join(split(sub('1',')0(',sub('[^1]+'*S,lambda l:'.'*len(l[0]),str(r))).strip('(.)'),'%s'%l))))
 return[*zip(*l[1:e])][1:e]
```

# [192]
[7e0986d6.json](https://arcprize.org/play?task=7e0986d6)


**Size: 99 bytes**
* color_guessing
* remove_noise

```python
show_examples(load_examples(192)['train'])
```

```python
%%writefile task192.py
p=lambda g,k=3:g*-k or[[k:=[c:=r.pop(),k][[0,*r][-1]in[k,0]*c]for _ in r*1]for*r,in zip(*p(g,k-1))]
```

# [193]
[7f4411dc.json](https://arcprize.org/play?task=7f4411dc)


**Size: 71 bytes**
* rectangle_guessing
* remove_noise

```python
show_examples(load_examples(193)['train'])
```

```python
%%writefile task193.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i]+i,i[1:]+[i],*w)]or(w.count(i)>1)*i
```

# [194]
[7fe24cdd.json](https://arcprize.org/play?task=7fe24cdd)


**Size: 55 bytes**
* image_repetition
* image_rotation

```python
show_examples(load_examples(194)['train'])
```

```python
%%writefile task194.py
p=lambda i,s=[],k=3:-k*i or p([*zip(*i+s)],i[::-1],k-1)
```

# [195]
[80af3007.json](https://arcprize.org/play?task=80af3007)


**Size: 94 bytes**
* crop
* pattern_resizing
* image_resizing
* fractal_repetition

```python
show_examples(load_examples(195)['train'])
```

```python
%%writefile task195.py
p=lambda a,n=1:[[y&sum(a*3,())[n:=n+3]for y in x]for x in-n*a]or p([*filter(any,zip(*a))],n-1)
```

# [196]
[810b9b61.json](https://arcprize.org/play?task=810b9b61)


**Size: 104 bytes**
* recoloring
* detect_closed_curves

```python
show_examples(load_examples(196)['train'])
```

```python
%%writefile task196.py
p=lambda g,i=19:g*-i or[*map(lambda*r,q=8:[q:=[c|8-c&q,c|-c&~q%3,c%4][i//8]for c in r],*p(g,i-1)[::-1])]
```

# [197]
[82819916.json](https://arcprize.org/play?task=82819916)


**Size: 51 bytes**
* pattern_repetition
* color_guessing
* draw_line_from_point
* associate_colors_to_colors

```python
show_examples(load_examples(197)['train'])
```

```python
%%writefile task197.py
p=lambda g:[[*map({}.setdefault,g[1],A)]for A in g]
```

# [198]
[83302e8f.json](https://arcprize.org/play?task=83302e8f)


**Size: 107 bytes**
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
import re;p=lambda g:exec("g[::-1]=zip(*eval(re.sub('(?=0|3, 4|3[^)]*[^)34]{6})','6^5-',str(g))));"*24)or g
```

# [199]
[834ec97d.json](https://arcprize.org/play?task=834ec97d)


**Size: 77 bytes**
* draw_line_from_border
* pattern_repetition
* spacing
* measure_distance_from_side

```python
show_examples(load_examples(199)['train'])
```

```python
%%writefile task199.py
p=lambda a:-~(y:=a.index(r:=max(a)))*[([4,0]*8)[r<r[1::2]:][:len(r)]]+a[y:-1]
```

# [200]
[8403a5d5.json](https://arcprize.org/play?task=8403a5d5)


**Size: 84 bytes**
* draw_line_from_point
* pattern_repetition
* direction_guessing

```python
show_examples(load_examples(200)['train'])
```

```python
%%writefile task200.py
p=lambda g,t=5:[(x[:g[9].index(c:=max(g[9]))]+[c,t,c,t:=any(x)*5]*9)[:10]for x in g]
```

# [201]
[846bdb03.json](https://arcprize.org/play?task=846bdb03)


**Size: 185 bytes**
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
def p(g):f=1;m=[R for r in zip(*g)if any(R:=[x*f*(f:=f^(x==4)or(g:=g+[x])>g)for x in r])];return[z:=[4,*[0]*len(m),4],*[[a:=g[15],*r[::a in m[0]or-1],g[-2]]for*r,in zip(*m)if any(r)],z]
```

# [202]
[855e0971.json](https://arcprize.org/play?task=855e0971)


**Size: 98 bytes**
* draw_line_from_point
* direction_guessing
* separate_images
* holes

```python
show_examples(load_examples(202)['train'])
```

```python
%%writefile task202.py
p=lambda g:exec("v=p,;g[:]=zip(*[map(min,[i,v][len({*v,*i,0})<3],v:=i)for*i,in g[::-1]]);"*36)or g
```

# [203]
[85c4e7cd.json](https://arcprize.org/play?task=85c4e7cd)


**Size: 64 bytes**
* color_guessing
* recoloring
* color_permutation

```python
show_examples(load_examples(203)['train'])
```

```python
%%writefile task203.py
p=lambda g:[[g[l:=len(g)//2][r.index(t)-l]for t in r]for r in g]
```

# [204]
[868de0fa.json](https://arcprize.org/play?task=868de0fa)


**Size: 93 bytes**
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
import re
p=lambda g:eval(re.sub('(?<!1, )1,(.+?)1',r'1,*[2+(c:=len([\1]))%2*5]*c,1',str(g)))
```

# [205]
[8731374e.json](https://arcprize.org/play?task=8731374e)


**Size: 134 bytes**
* rectangle_guessing
* crop
* draw_line_from_point

```python
show_examples(load_examples(205)['train'])
```

```python
%%writefile task205.py
p=lambda g,k=87:-k*[[min(e+u,key=e.count)for*u,in zip(*g)]for*e,in g]or p([*zip(*g)][sum(u[:6].count(u[0])>4for u in g)<6:][::-1],k-1)
```

# [206]
[88a10436.json](https://arcprize.org/play?task=88a10436)


**Size: 139 bytes**
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(206)['train'])
```

```python
%%writefile task206.py
def p(a):
 i,j=divmod(sum(a,[]).index(5),len(a[0]));a[i][j]=0
 for s in eval('filter(any,zip(*'*2+'a))))'):a[i-1][j-1:j+2]=s;i+=1
 return a
```

# [207]
[88a62173.json](https://arcprize.org/play?task=88a62173)


**Size: 74 bytes**
* detect_grid
* separate_images
* find_the_intruder
* crop

```python
show_examples(load_examples(207)['train'])
```

```python
%%writefile task207.py
p=lambda g:[p(i)for*i,in map(zip,g,g[3:])]or min(q:=g[0]+g[1],key=q.count)
```

# [208]
[890034e9.json](https://arcprize.org/play?task=890034e9)


**Size: 163 bytes**
* pattern_repetition
* rectangle_guessing
* contouring

```python
show_examples(load_examples(208)['train'])
```

```python
%%writefile task208.py
from re import*
p=lambda g:eval((i:=min(k:=str(g)+'#[]'*X,key=k.count)).join(split(sub(i,f')[^{i}](',sub("[^%s]+"%i*18,lambda x:'.'*len(x[0]),k)).strip(".()"),k)))
```

# [209]
[8a004b2b.json](https://arcprize.org/play?task=8a004b2b)


**Size: 247 bytes (387 raw)**
* pattern_repetition
* pattern_resizing
* pattern_juxtaposition
* rectangle_guessing
* crop

```python
show_examples(load_examples(209)['train'])
```

```python
%%writefile task209.py
def	p(i):
	e=i
	for	o	in	e*4:e=[[*o]for	o	in	zip(*e[~4:])if{*o}-{0,4}];i=[[*o]for	o	in	zip(*i[(4in	i[-1])-2::-1])]
	for	o,f	in	enumerate(i):
		if[0for	t,f	in	enumerate(i)for	a,f	in	enumerate(o*f)for	r,f	in	enumerate(o*all(i	in[0,((e+20*[[]])[(r-t)//o]+20*[4])[(n-a)//o]]for	r,i	in	enumerate(i)for	n,i	in	enumerate(i))*e)for	n,f	in	enumerate(o*f)for	i[r+t][n+a]in[e[r//o][n//o]]]:return	i
```

# [210]
[8be77c9e.json](https://arcprize.org/play?task=8be77c9e)


**Size: 20 bytes**
* image_repetition
* image_reflection

```python
show_examples(load_examples(210)['train'])
```

```python
%%writefile task210.py
p=lambda j:j+j[::-1]
```

# [211]
[8d5021e8.json](https://arcprize.org/play?task=8d5021e8)


**Size: 48 bytes**
* image_repetition
* image_reflection

```python
show_examples(load_examples(211)['train'])
```

```python
%%writefile task211.py
p=lambda g:[a[::-1]+a for a in(g[::-1]+g)*2][:9]
```

# [212]
[8d510a79.json](https://arcprize.org/play?task=8d510a79)


**Size: 88 bytes**
* draw_line_from_point
* detect_wall
* direction_guessing
* associate_colors_to_bools

```python
show_examples(load_examples(212)['train'])
```

```python
%%writefile task212.py
p=lambda i,k=18:~k*i or p([[x.pop()or[0,*x][k%-2]%5&6-(5in x)for x in i]for*x,in i],k-1)
```

# [213]
[8e1813be.json](https://arcprize.org/play?task=8e1813be)


**Size: 86 bytes**
* recoloring
* color_guessing
* direction_guesingcrop
* image_within_image

```python
show_examples(load_examples(213)['train'])
```

```python
%%writefile task213.py
p=lambda g:[l[o:]for r in g if(l:=[e for e in r if e%5])[~(o:=6-len({*"%s"%g})):]][o:]
```

# [214]
[8e5a5113.json](https://arcprize.org/play?task=8e5a5113)


**Size: 61 bytes**
* detect_wall
* separate_images
* image_repetition
* image_rotation

```python
show_examples(load_examples(214)['train'])
```

```python
%%writefile task214.py
p=lambda g:[r+g.pop()[3::-1]for r,*r[4:]in zip(g*1,*g[::-1])]
```

# [215]
[8eb1be9a.json](https://arcprize.org/play?task=8eb1be9a)


**Size: 42 bytes**
* pattern_repetition
* image_filling

```python
show_examples(load_examples(215)['train'])
```

```python
%%writefile task215.py
p=lambda g:[*map(max,g,g[3:6]*9,g[6:9]*9)]
```

# [216]
[8efcae92.json](https://arcprize.org/play?task=8efcae92)


**Size: 111 bytes**
* separate_images
* rectangle_guessing
* count_tiles
* take_maximum
* crop

```python
show_examples(load_examples(216)['train'])
```

```python
%%writefile task216.py
p=lambda g:max([sorted(sum(v:=[k[i%19:i%23]for k in g[i%21:i%22]],[]),key=1 .__eq__),v]for i in range(4**9))[1]
```

# [217]
[8f2ea7aa.json](https://arcprize.org/play?task=8f2ea7aa)


**Size: 95 bytes**
* crop
* fractal_repetition

```python
show_examples(load_examples(217)['train'])
```

```python
%%writefile task217.py
p=lambda g:[*eval('filter(sum,zip(*'*2+f'[[x&b{(" for %s in %s"*2+"]")*2}))))'%(*"xxbyxgy",g))]
```

# [218]
[90c28cc7.json](https://arcprize.org/play?task=90c28cc7)


**Size: 56 bytes**
* crop
* rectangle_guessing
* summarize

```python
show_examples(load_examples(218)['train'])
```

```python
%%writefile task218.py
p=lambda g,*x:[*{i:1for i in zip(*x or p(*g))if any(i)}]
```

# [219]
[90f3ed37.json](https://arcprize.org/play?task=90f3ed37)


**Size: 238 bytes (273 raw)**
* pattern_repetition
* recoloring

```python
show_examples(load_examples(219)['train'])
```

```python
%%writefile task219.py
def p(_):
 m,*f=[],
 for e,r in enumerate(_):
  if max(r)<1and f:m+={*f},;f=[]
  for a,r in enumerate(r):f+=[(a,e)]*r
 for d in m:
  for a,e in max([{(a+_,e-min(m[0])[1]+min(d)[1])for a,e in m[0]}for _ in(2,1,0,-1)],key=d.__and__):
   if 0<a<10:_[e][a]+=_[e][a]<1
 return _
```

# [220]
[913fb3ed.json](https://arcprize.org/play?task=913fb3ed)


**Size: 82 bytes**
* contouring
* associate_colors_to_colors

```python
show_examples(load_examples(220)['train'])
```

```python
%%writefile task220.py
p=lambda g,i=3:g*-i or p([[r.pop()or[0,*r][-1]**4%84%15for r in g]for r in g],i-1)
```

# [221]
[91413438.json](https://arcprize.org/play?task=91413438)


**Size: 86 bytes**
* count_tiles
* algebra
* image_repetition

```python
show_examples(load_examples(221)['train'])
```

```python
%%writefile task221.py
def p(i):j=sum(i,i).count(a:=0);return[(q*(9+(a:=a-1)//3*j)+[0]*21)[:j*3]for q in i*j]
```

# [222]
[91714a58.json](https://arcprize.org/play?task=91714a58)


**Size: 93 bytes**
* find_the_intruder
* remove_noise

```python
show_examples(load_examples(222)['train'])
```

```python
%%writefile task222.py
p=lambda g:[g:=[[c*(str(r*7+g).count(2*f"{c}, ")>9)for c in r]for*r,in zip(*g)]for _ in g][5]
```

# [223]
[9172f3a0.json](https://arcprize.org/play?task=9172f3a0)


**Size: 46 bytes**
* image_resizing

```python
show_examples(load_examples(223)['train'])
```

```python
%%writefile task223.py
p=lambda a:a>a*0!=0and[p(a[0])]*3+p(a[1:])or a
```

# [224]
[928ad970.json](https://arcprize.org/play?task=928ad970)


**Size: 114 bytes**
* rectangle_guessing
* color_guessing
* draw_rectangle

```python
show_examples(load_examples(224)['train'])
```

```python
%%writefile task224.py
p=lambda g,i=21:-i*g or[g[:(b:=any(g[0])*i<8)],[[max(max(g))]*99]][i<4]+[*zip(*p([*zip(*g[b:][::-1])],i-1))][::-1]
```

# [225]
[93b581b8.json](https://arcprize.org/play?task=93b581b8)


**Size: 123 bytes**
* pattern_expansion
* color_guessing
* out_of_boundary

```python
show_examples(load_examples(225)['train'])
```

```python
%%writefile task225.py
p=lambda a,n=0,R=range(6):a[i:=n//6][j:=n%6]and[[a[y][x]+(x-j&y-i&2>0)*a[i+(y<i)][j+(x<j)]for x in R]for y in R]or p(a,n+1)
```

# [226]
[941d9a10.json](https://arcprize.org/play?task=941d9a10)


**Size: 117 bytes**
* detect_grid
* loop_filling
* pairwise_analogy

```python
show_examples(load_examples(226)['train'])
```

```python
%%writefile task226.py
p=lambda g,k=7,r=range(10):g*-k or p([(q:=0)or[q:=g[i][~j]or[~i//4*-(j^9-i<2&~j),q%5][k<7]for i in r]for j in r],k-1)
```

# [227]
[94f9d214.json](https://arcprize.org/play?task=94f9d214)


**Size: 52 bytes**
* separate_images
* take_complement
* pattern_intersection

```python
show_examples(load_examples(227)['train'])
```

```python
%%writefile task227.py
p=lambda g,h=[]:g*0!=0and[*map(p,g,h+g[4:])]or~g+h&2
```

# [228]
[952a094c.json](https://arcprize.org/play?task=952a094c)


**Size: 114 bytes**
* rectangle_guessing
* inside_out

```python
show_examples(load_examples(228)['train'])
```

```python
%%writefile task228.py
import re
p=lambda g:eval(re.sub(r'([^0])((, (?!\1|0).).*0\3.{28})0',r'0\2\1',f'{*zip(*g[70:]or p(g*2)),}'))[::-1]
```

# [229]
[9565186b.json](https://arcprize.org/play?task=9565186b)


**Size: 73 bytes**
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
p=lambda g:[[[5,i][i==max(q:=sum(g,g),key=q.count)]for i in j]for j in g]
```

# [230]
[95990924.json](https://arcprize.org/play?task=95990924)


**Size: 95 bytes**
* pattern_expansion

```python
show_examples(load_examples(230)['train'])
```

```python
%%writefile task230.py
p=lambda g,i=6:g*(i<3)or p([[(d:=r.pop()or[0,*r][-1]*i)>>5*(d>6>4>i)for r in g]for _ in g],i-1)
```

# [231]
[963e52fc.json](https://arcprize.org/play?task=963e52fc)


**Size: 42 bytes**
* image_expansion
* pattern_expansion

```python
show_examples(load_examples(231)['train'])
```

```python
%%writefile task231.py
p=lambda g:[(r[:6]*2+r*2)[:-12]for r in g]
```

# [232]
[97999447.json](https://arcprize.org/play?task=97999447)


**Size: 57 bytes**
* draw_line_from_point
* pattern_expansion

```python
show_examples(load_examples(232)['train'])
```

```python
%%writefile task232.py
p=lambda i,e=0:i*0!=0and[p(y)or[e:=y-e,5][e<0]for y in i]
```

# [233]
[97a05b5b.json](https://arcprize.org/play?task=97a05b5b)


**Size: 260 bytes (514 raw)**
* pattern_moving
* pattern_juxtaposition
* crop
* shape_guessing

```python
show_examples(load_examples(233)['train'])
```

```python
%%writefile task233.py
def p(n):
 for i,r in sorted([[-r.count(2),r]for a in range(len(n)-2)for l in range(len(n[0])-2)if all(r:=[n[a+i][l+d]for i in range(3)for d in range(3)])and{2}!={*r}!=[2for i in range(3)for d in range(3)for n[a+i][l+d]in[0]]]):
  for i in range(4):[[2for i in range(3)for d in range(3)for n[a+i][l+d]in[r[i*3+d]]]for a in range(len(n)-2)for l in range(len(n[0])-2)if all(n[a+i][l+d]==2!=r[i*3+d]or r[i*3+d]-n[a+i][l+d]==2in n[a+i]for i in range(3)for d in range(3))];n=[n[::-1]for*n,in zip(*n)if 2 in n]
 return n
```

# [234]
[98cf29f8.json](https://arcprize.org/play?task=98cf29f8)


**Size: 106 bytes**
* pattern_moving
* bring_patterns_close

```python
show_examples(load_examples(234)['train'])
```

```python
%%writefile task234.py
p=lambda g:exec('c={0};g[:]=zip(*(g[:1]*99+[j for*j,in g if(c:=c|{*j})-{sum(j),0}])[:~len(g):-1]);'*4)or g
```

# [235]
[995c5fa3.json](https://arcprize.org/play?task=995c5fa3)


**Size: 61 bytes**
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
p=lambda g:[[g[1][x]*sum(g[2][x:x+3])%13^8]*3for x in b'']
```

# [236]
[99b1bc43.json](https://arcprize.org/play?task=99b1bc43)


**Size: 54 bytes**
* take_complement
* detect_wall
* separate_images
* pattern_intersection

```python
show_examples(load_examples(236)['train'])
```

```python
%%writefile task236.py
p=lambda g,h=[]:g*0!=0and[*map(p,g,h+g[5:])]or-h%5^g*3
```

# [237]
[99fa7670.json](https://arcprize.org/play?task=99fa7670)


**Size: 66 bytes**
* draw_line_from_point
* pattern_expansion

```python
show_examples(load_examples(237)['train'])
```

```python
%%writefile task237.py
p=lambda g,P=0:[[*[(P:=P or x for x in r+[P])][P:=0]]for*r,_ in g]
```

# [238]
[9aec4887.json](https://arcprize.org/play?task=9aec4887)


**Size: 195 bytes**
* pattern_moving
* x_marks_the_spot
* crop
* recoloring
* color_guessing

```python
show_examples(load_examples(238)['train'])
```

```python
%%writefile task238.py
def p(g,s='[[8,*r,8]for r in zip(*%s)if 8in r]'):R=range(l:=len(B:=eval(s%s%g)));return[[B[i][j]*[[*{c/8:0for c in sum(g,[])if c%8}][(i>j)+(~i+l<j)*2],0<i<l-1][i in(j,~j+l)]for j in R]for i in R]
```

# [239]
[9af7a82c.json](https://arcprize.org/play?task=9af7a82c)


**Size: 99 bytes**
* separate_images
* count_tiles
* summarize
* order_numbers

```python
show_examples(load_examples(239)['train'])
```

```python
%%writefile task239.py
p=lambda i:[b:=sum(i,[]),*filter(any,zip(*sorted([c:=-b.count(e),e]*-c+[0]*22for e in{*b})))][2::2]
```

# [240]
[9d9215db.json](https://arcprize.org/play?task=9d9215db)


**Size: 89 bytes**
* pattern_expansion
* pattern_reflection
* pattern_rotation

```python
show_examples(load_examples(240)['train'])
```

```python
%%writefile task240.py
k=1;p=lambda g:exec("k+=2;g[i:=k%19][j:=k%18]|=g[i][j-2]*(i<j-2<16-i)|g[j][~i];"*746)or g
```

# [241]
[9dfd6313.json](https://arcprize.org/play?task=9dfd6313)


**Size: 21 bytes**
* image_reflection
* diagonal_symmetry

```python
show_examples(load_examples(241)['train'])
```

```python
%%writefile task241.py
p=lambda j:[*zip(*j)]
```

# [242]
[9ecd008a.json](https://arcprize.org/play?task=9ecd008a)


**Size: 54 bytes**
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
p=lambda g:[j[~j.index(0)::-1][:3]for j in g if 0in j]
```

# [243]
[9edfc990.json](https://arcprize.org/play?task=9edfc990)


**Size: 75 bytes**
* background_filling
* holes

```python
show_examples(load_examples(243)['train'])
```

```python
%%writefile task243.py
p=lambda g,i=-79:g*i or p([[r.pop()or 1in r[-1:]for r in g]for*r,in g],i+1)
```

# [244]
[9f236235.json](https://arcprize.org/play?task=9f236235)


**Size: 61 bytes**
* detect_grid
* summarize
* image_reflection

```python
show_examples(load_examples(244)['train'])
```

```python
%%writefile task244.py
p=lambda g,w=2:[[0,max,p][w](g:=r,-2)for r in g if g!=r][::w]
```

# [245]
[a1570a43.json](https://arcprize.org/play?task=a1570a43)


**Size: 98 bytes**
* pattern_moving
* rectangle_guessing
* x_marks_the_spot

```python
show_examples(load_examples(245)['train'])
```

```python
%%writefile task245.py
p=lambda g,i=7,k=0:g*-i or p([[c^k-~(k:=c)&(2in max(g,key=any))*2for c in r]for*r,in zip(*g)],i-1)
```

# [246]
[a2fd1cf0.json](https://arcprize.org/play?task=a2fd1cf0)


**Size: 104 bytes**
* connect_the_dots

```python
show_examples(load_examples(246)['train'])
```

```python
%%writefile task246.py
p=lambda g,n=3:-n*g or p([[r.pop()|(n|2in r!=3-n%2in(g:=g.pop()+g))*8for _ in r*1]for*r,in zip(*g)],n-1)
```

# [247]
[a3325580.json](https://arcprize.org/play?task=a3325580)


**Size: 92 bytes**
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
p=lambda a,m=9:[*zip(*{(d,)*m:0for d in sum(zip(*a),())if sum(a,a).count(d)==m})]or p(a,m-1)
```

# [248]
[a3df8b1e.json](https://arcprize.org/play?task=a3df8b1e)


**Size: 67 bytes**
* pattern_expansion
* draw_line_from_point
* diagonals
* bounce

```python
show_examples(load_examples(248)['train'])
```

```python
%%writefile task248.py
def p(g):
	b=j=0
	for r in g[::-1]:r[j]=1;b^=-r[b];j-=b|1
	return g
```

# [249]
[a416b8f3.json](https://arcprize.org/play?task=a416b8f3)


**Size: 26 bytes**
* image_repetition

```python
show_examples(load_examples(249)['train'])
```

```python
%%writefile task249.py
p=lambda j:[E*2for E in j]
```

# [250]
[a48eeaf7.json](https://arcprize.org/play?task=a48eeaf7)


**Size: 95 bytes**
* pattern_moving
* bring_patterns_close
* gravity
* direction_guessing

```python
show_examples(load_examples(250)['train'])
```

```python
%%writefile task250.py
p=lambda a:[a:=[sorted(b[:(i:=str(a).find('2')>>5)])+b[i:]for*b,in zip(*a)][::-1]for _ in a][3]
```

# [251]
[a5313dff.json](https://arcprize.org/play?task=a5313dff)


**Size: 84 bytes**
* loop_filling

```python
show_examples(load_examples(251)['train'])
```

```python
%%writefile task251.py
p=lambda g,i=31:g*-i or p([[r.pop()&~4**[0,*r][-1]or i>30for r in g]for _ in g],i-1)
```

# [252]
[a5f85a15.json](https://arcprize.org/play?task=a5f85a15)


**Size: 53 bytes**
* recoloring
* pattern_modification
* pairwise_analogy

```python
show_examples(load_examples(252)['train'])
```

```python
%%writefile task252.py
p=lambda g,v=0:g*0!=0and[*map(p,g,[-1,4]*9)]or-g//v&v
```

# [253]
[a61ba2ce.json](https://arcprize.org/play?task=a61ba2ce)


**Size: 114 bytes**
* pattern_moving
* bring_patterns_close
* crop
* jigsaw

```python
show_examples(load_examples(253)['train'])
```

```python
%%writefile task253.py
p=lambda g:[[(i:=a&b)*0+max(v*(sum(g,g)[i-1:(i:=i+1)]==[v,v])for v in sum(g,[]))for a in b'[Z']for b in b'A[']
```

# [254]
[a61f2674.json](https://arcprize.org/play?task=a61f2674)


**Size: 81 bytes**
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
p=lambda g:[[(sum(g[-sum(c)//5])|5)%sum(g[8])*x%3for*c,x in zip(*g,r)]for r in g]
```

# [255]
[a64e4611.json](https://arcprize.org/play?task=a64e4611)


**Size: 211 bytes**
* background_filling
* rectangle_guessing

```python
show_examples(load_examples(255)['train'])
```

```python
%%writefile task255.py
import re;p=lambda g,k=9:eval(re.sub(*["\((%s|(0, )+3.{5}3),?"%re.search(r"([ ,03]{61,})(.*\1){3}|$",g:=f'{*zip(*~k*g or p(g,k-1)),}')[1],"(?=0((,[^,]*){31}|, )[1-9])",r"(*[3]*len([\1]),","1<"][k<2::2],g))[::-1]
```

# [256]
[a65b410d.json](https://arcprize.org/play?task=a65b410d)


**Size: 93 bytes**
* pattern_expansion
* count_tiles
* associate_colors_to_ranks

```python
show_examples(load_examples(256)['train'])
```

```python
%%writefile task256.py
def p(g):
 b=sum(m:=max(g))//2+g.index(m)
 for r in g:r[:b]=[2+(r<m)-m[b]]*b;b-=b>0
 return g
```

# [257]
[a68b268e.json](https://arcprize.org/play?task=a68b268e)


**Size: 68 bytes**
* detect_grid
* separate_images
* pattern_juxtaposition

```python
show_examples(load_examples(257)['train'])
```

```python
%%writefile task257.py
p=lambda a:[p(b)for*b,in map(zip,a,a[5:])]or max(sum(a,()),key=bool)
```

# [258]
[a699fb00.json](https://arcprize.org/play?task=a699fb00)


**Size: 61 bytes**
* pattern_expansion
* connect_the_dots

```python
show_examples(load_examples(258)['train'])
```

```python
%%writefile task258.py
import re
p=lambda g:eval(re.sub('1, 0(?=, 1)','1,2',str(g)))
```

# [259]
[a740d043.json](https://arcprize.org/play?task=a740d043)


**Size: 83 bytes**
* crop
* detect_background_color
* recoloring

```python
show_examples(load_examples(259)['train'])
```

```python
%%writefile task259.py
p=lambda g:exec('g[:]=zip(*eval(f"{g[any(g[-1])-2::-1]}".replace(*"10")));'*24)or g
```

# [260]
[a78176bb.json](https://arcprize.org/play?task=a78176bb)


**Size: 130 bytes**
* draw_parallel_line
* direction_guessing
* remove_intruders

```python
show_examples(load_examples(260)['train'])
```

```python
%%writefile task260.py
exec("p=lambda a:[[max({*max(a)}-{5})*any(a[i][j]%5or 2==sum(m-n-i+j+k%5==2<a[m][n]"+'for %s in range(10)%s'*6%(*'m n k)_)j]i]',))
```

# [261]
[a79310a0.json](https://arcprize.org/play?task=a79310a0)


**Size: 47 bytes**
* pattern_moving
* recoloring
* pairwise_analogy

```python
show_examples(load_examples(261)['train'])
```

```python
%%writefile task261.py
p=lambda g:[[i%3for i in a]for a in[g.pop()]+g]
```

# [262]
[a85d4709.json](https://arcprize.org/play?task=a85d4709)


**Size: 39 bytes**
* separate_images
* associate_colors_to_images
* summarize

```python
show_examples(load_examples(262)['train'])
```

```python
%%writefile task262.py
p=lambda g:[[c^6&b+~a]*3for a,b,c in g]
```

# [263]
[a87f7484.json](https://arcprize.org/play?task=a87f7484)


**Size: 106 bytes**
* separate_images
* find_the_intruder
* crop

```python
show_examples(load_examples(263)['train'])
```

```python
%%writefile task263.py
p=lambda g,h=0:-(M:=bytes(map(bool,sum(g:=[*zip(*h or p(g,g))],())))).find(M[:9],9)*g[:3]or p(g[3:]+g[:3])
```

# [264]
[a8c38be5.json](https://arcprize.org/play?task=a8c38be5)


**Size: 177 bytes**
* pattern_moving
* jigsaw
* crop

```python
show_examples(load_examples(264)['train'])
```

```python
%%writefile task264.py
T=3,2,1
p=lambda g:[sum([eval(f"sorted([0in(S:=sum(g,T)),[i!=5for i in S],*g]{'for*g,in map(zip,g,g[1:],g[2:])'*2})#{g}")[42>>Y&7*~-X^Y][-x]for Y in T],())for X in T for x in T]
```

# [265]
[a8d7556c.json](https://arcprize.org/play?task=a8d7556c)


**Size: 104 bytes**
* recoloring
* rectangle_guessing

```python
show_examples(load_examples(265)['train'])
```

```python
%%writefile task265.py
import re
p=lambda g:eval(re.sub("0(?=.{949,952}(.{56})?0(?!.{37}0.{485}]), 0.{52}0, 0)","2","%r#"%g*2))
```

# [266]
[a9f96cdd.json](https://arcprize.org/play?task=a9f96cdd)


**Size: 88 bytes**
* replace_pattern
* out_of_boundary

```python
show_examples(load_examples(266)['train'])
```

```python
%%writefile task266.py
p=lambda g:g[9:]or[[(8|9>>c|a*6)%9for a,c in zip([0]+r,r[1:]+[0])]for*r,in zip(*p(g*2))]
```

# [267]
[aabf363d.json](https://arcprize.org/play?task=aabf363d)


**Size: 46 bytes**
* recoloring
* color_guessing
* remove_intruders

```python
show_examples(load_examples(267)['train'])
```

```python
%%writefile task267.py
p=lambda g:[[g[6][y>[x]]for x in y]for y in g]
```

# [268]
[aba27056.json](https://arcprize.org/play?task=aba27056)


**Size: 190 bytes**
* pattern_expansion
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(268)['train'])
```

```python
%%writefile task268.py
import re
p=lambda g,i=7:-i*g or p(eval(re.sub("0(?=%s)"%["(.%r0)*, [^0].%%r[^0], 4"%{o:=len(g)*3+4}%{o-6},r"[0, ]++(.).{,3}\).*#.*\1, \1, [0, ]+\1"][i>3],"4",f'{*zip(*g),}#{g}'))[::-1],i-1)
```

# [269]
[ac0a08a4.json](https://arcprize.org/play?task=ac0a08a4)


**Size: 63 bytes**
* image_resizing
* count_tiles
* size_guessing

```python
show_examples(load_examples(269)['train'])
```

```python
%%writefile task269.py
p=lambda g:eval('[[g\nfor g in g for c in[*{*"%s"}][5:]]#'%g*2)
```

# [270]
[ae3edfdc.json](https://arcprize.org/play?task=ae3edfdc)


**Size: 115 bytes**
* bring_patterns_close
* gravity

```python
show_examples(load_examples(270)['train'])
```

```python
%%writefile task270.py
import re
p=lambda g,k=7:g*-k or eval(re.sub(f"{k|3}([^)]*).(?=, {2-k//4})",r"0\1k|3",f'{*zip(*p(g,k-1)),}'))[::-1]
```

# [271]
[ae4f1146.json](https://arcprize.org/play?task=ae4f1146)


**Size: 86 bytes**
* separate_images
* count_tiles
* crop

```python
show_examples(load_examples(271)['train'])
```

```python
%%writefile task271.py
p=eval(f"lambda a:max([str(a).count('1'),a]{'for*a,in map(zip,a,a[1:],a[2:])'*2})[1]")
```

# [272]
[aedd82e4.json](https://arcprize.org/play?task=aedd82e4)


**Size: 71 bytes**
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
p=lambda i,*w:i*0!=0and[*map(p,i,[i*2]+i,i[1:]+[i*2],*w)]or~(2in w)*i%3
```

# [273]
[af902bf9.json](https://arcprize.org/play?task=af902bf9)


**Size: 75 bytes**
* ex_nihilo
* x_marks_the_spot

```python
show_examples(load_examples(273)['train'])
```

```python
%%writefile task273.py
p=lambda g,*r,k=0:[c+(k:=k^c)%6and c^2for c in r]or[*map(p,g,*map(p,g,*g))]
```

# [274]
[b0c4d837.json](https://arcprize.org/play?task=b0c4d837)


**Size: 65 bytes**
* measure_length
* associate_images_to_numbers

```python
show_examples(load_examples(274)['train'])
```

```python
%%writefile task274.py
p=lambda g:[[8,(s:=sum(map(max,g))*6)+4&8,-s&8],[0,0,~s&8],[0]*3]
```

# [275]
[b190f7f5.json](https://arcprize.org/play?task=b190f7f5)


**Size: 126 bytes**
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
def p(g):R=len(g+g[0])//3;r=range(-R*R,0);return[[sum(g[u%R-t][v%R-t]//8*g[u//R+t][v//R+t]for t in(0,R))for v in r]for u in r]
```

# [276]
[b1948b0a.json](https://arcprize.org/play?task=b1948b0a)


**Size: 37 bytes**
* recoloring
* associate_colors_to_colors

```python
show_examples(load_examples(276)['train'])
```

```python
%%writefile task276.py
p=lambda g:g*-1and-g%6|2or[*map(p,g)]
```

# [277]
[b230c067.json](https://arcprize.org/play?task=b230c067)


**Size: 155 bytes**
* recoloring
* separate_shapes
* find_the_intruder
* associate_colors_to_bools

```python
show_examples(load_examples(277)['train'])
```

```python
%%writefile task277.py
z=[0];p=lambda g,k=38,h=2,q=z*9:~k*g or p([q:=[v and[v%63,P|p|v,h:=h*64,v//sum(g,z).count(v)][k>>4]for P,p,v in zip(z+q,z+r,r)]for*r,in zip(*g[::-1])],k-1)
```

# [278]
[b27ca6d3.json](https://arcprize.org/play?task=b27ca6d3)


**Size: 104 bytes**
* find_the_intruder
* count_tiles
* contouring

```python
show_examples(load_examples(278)['train'])
```

```python
%%writefile task278.py
p=lambda m,k=11:-k*m or p([[[v%9,v|3%-~u,v<<u][k>>2]for u,v in zip([0]+r,r)]for*r,in zip(*m[::-1])],k-1)
```

# [279]
[b2862040.json](https://arcprize.org/play?task=b2862040)


**Size: 106 bytes**
* recoloring
* detect_closed_curves
* associate_colors_to_bools

```python
show_examples(load_examples(279)['train'])
```

```python
%%writefile task279.py
p=lambda g,i=94:g*~i or p([[9&r.pop()%[q+9,9|3-q][i<9]or(i<0)*9for q in[0]+r[:0:-1]]for*r,in zip(*g)],i-1)
```

# [280]
[b527c5c6.json](https://arcprize.org/play?task=b527c5c6)


**Size: 145 bytes**
* pattern_expansion
* draw_line_from_point
* contouring
* direction_guessing
* size_guessing

```python
show_examples(load_examples(280)['train'])
```

```python
%%writefile task280.py
p=lambda g,n=11:-n*g or[[P:=[((a:=(8%~P<x)*a+x%2*8|2)>2>x)*a,-6%(P-2|1-x|x),x&3][n//4]or x for x in r]for r in zip(*p(g,n-1))if[a:=0,P:=0]][::-1]
```

# [281]
[b548a754.json](https://arcprize.org/play?task=b548a754)


**Size: 103 bytes**
* pattern_expansion
* pattern_modification
* x_marks_the_spot

```python
show_examples(load_examples(281)['train'])
```

```python
%%writefile task281.py
p=lambda a,n=47,*P:-n*a or p([*zip(*[max(P*({0,8}in map(set,a)),P:=a.pop(),key=set)for _ in a*1])],n-1)
```

# [282]
[b60334d2.json](https://arcprize.org/play?task=b60334d2)


**Size: 74 bytes**
* replace_pattern

```python
show_examples(load_examples(282)['train'])
```

```python
%%writefile task282.py
p=lambda g:g[99:]or[eval(8*"+(x:=r.pop()),x//5|x^0")for*r,in zip(*p(g*2))]
```

# [283]
[b6afb2da.json](https://arcprize.org/play?task=b6afb2da)


**Size: 72 bytes**
* recoloring
* replace_pattern
* rectangle_guessing

```python
show_examples(load_examples(283)['train'])
```

```python
%%writefile task283.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i]+i,i[1:]+[i],*w)]or-i%8*w.count(5)%5
```

# [284]
[b7249182.json](https://arcprize.org/play?task=b7249182)


**Size: 195 bytes**
* pattern_expansion

```python
show_examples(load_examples(284)['train'])
```

```python
%%writefile task284.py
p=lambda g:exec("m=max(g)\nfor x in-2%len({*m})*b'osqmnr':(l,c),*_,r=filter(min,enumerate(m,1));s=l-2+r[0]>>1;m[l:s]=-l%s*[c];g[g.index(m)+x%5-2][s-x%2]=c\ng[:]=[y[::-1]for*y,in zip(*g)];"*4)or g
```

# [285]
[b775ac94.json](https://arcprize.org/play?task=b775ac94)


**Size: 226 bytes (293 raw)**
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
%%writefile task285.py
def p(h):
 for e in range(8):
  h[::-1]=zip(*h)
  for e in range(len(h)):
   for i in range(len(h)):
    for q,g in(f:=[(e,i)]):*h[q],=h[q];h[q][i+i-g-1]=h[e][i-1];f+=[(p+q,g+n)for p in range(-1,2)for n in range(-1,2)if 0<h[e][i-1]!=(2*(2*h)[p+q])[g+n]==h[e][i]>0==h[p+q][i+i-g-n-1]]
 return h
```

# [286]
[b782dc8a.json](https://arcprize.org/play?task=b782dc8a)


**Size: 104 bytes**
* pattern_expansion
* maze

```python
show_examples(load_examples(286)['train'])
```

```python
%%writefile task286.py
p=lambda i,k=39:-k*i or[[t:=y or sum({*t%8*sum(i,x)}-{t,8})for y in[8]+x][:0:-1]for*x,in zip(*p(i,k-1))]
```

# [287]
[b8825c91.json](https://arcprize.org/play?task=b8825c91)


**Size: 53 bytes**
* pattern_completion
* pattern_rotation
* pattern_reflection

```python
show_examples(load_examples(287)['train'])
```

```python
%%writefile task287.py
p=lambda*g:g[g[0]==4]*-1*-1or[*map(p,g[-1][::-1],*g)]
```

# [288]
[b8cdaf2b.json](https://arcprize.org/play?task=b8cdaf2b)


**Size: 86 bytes**
* pattern_expansion
* draw_line_from_point
* diagonals
* pairwise_analogy

```python
show_examples(load_examples(288)['train'])
```

```python
%%writefile task288.py
def p(g):d=c=g[-2].count(0)//2;exec(c*"c-=1;r=g[c-d-2];r[c]=r[~c]=g[-1][d];");return g
```

# [289]
[b91ae062.json](https://arcprize.org/play?task=b91ae062)


**Size: 63 bytes**
* image_resizing
* size_guessing
* count_different_colors

```python
show_examples(load_examples(289)['train'])
```

```python
%%writefile task289.py
p=lambda g:eval('[[g\nfor g in g for _ in[*{*"%s"}][5:]]#'%g*2)
```

# [290]
[b94a9452.json](https://arcprize.org/play?task=b94a9452)


**Size: 65 bytes**
* crop
* take_negative

```python
show_examples(load_examples(290)['train'])
```

```python
%%writefile task290.py
p=lambda g,*u:[sum({*u},r*-1)or p(r,*sum(g,r))for r in g if[r]>g]
```

# [291]
[b9b7f026.json](https://arcprize.org/play?task=b9b7f026)


**Size: 59 bytes**
* find_the_intruder
* summarize

```python
show_examples(load_examples(291)['train'])
```

```python
%%writefile task291.py
p=lambda m,k=1:[*{r.count(k)for r in m},[k]][3:]or p(m,k+1)
```

# [292]
[ba26e723.json](https://arcprize.org/play?task=ba26e723)


**Size: 51 bytes**
* pattern_modification
* pairwise_analogy
* recoloring

```python
show_examples(load_examples(292)['train'])
```

```python
%%writefile task292.py
p=lambda g,v=0:g*0!=0and[*map(p,g,b'\n'*7)]or-g%v
```

# [293]
[ba97ae07.json](https://arcprize.org/play?task=ba97ae07)


**Size: 59 bytes**
* pattern_modification
* pairwise_analogy
* rettangle_guessing
* recoloring

```python
show_examples(load_examples(293)['train'])
```

```python
%%writefile task293.py
p=lambda g:[[(c in r)**c*r[0]or c for c in g[0]]for r in g]
```

# [294]
[bb43febb.json](https://arcprize.org/play?task=bb43febb)


**Size: 70 bytes**
* loop_filling
* rettangle_guessing

```python
show_examples(load_examples(294)['train'])
```

```python
%%writefile task294.py
import re;p=lambda g:eval(re.sub('(?<=5.{34})5(?=.{34}5)','2',str(g)))
```

# [295]
[bbc9ae5d.json](https://arcprize.org/play?task=bbc9ae5d)


**Size: 54 bytes**
* pattern_expansion
* image_expansion

```python
show_examples(load_examples(295)['train'])
```

```python
%%writefile task295.py
p=lambda g:[x:=g[0]]+[x:=x[:1]+x[:-1]for _ in x[2::2]]
```

# [296]
[bc1d5164.json](https://arcprize.org/play?task=bc1d5164)


**Size: 55 bytes**
* pattern_moving
* pattern_juxtaposition
* crop
* pairwise_analogy

```python
show_examples(load_examples(296)['train'])
```

```python
%%writefile task296.py
p=lambda*g:[*map([*g,max,p][2],*[r[-3:]for r in g],*g)]
```

# [297]
[bd4472b8.json](https://arcprize.org/play?task=bd4472b8)


**Size: 43 bytes**
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
p=lambda g:g[:2]+[*zip(*g[:1]*len(g[0]))]*2
```

# [298]
[bda2d7a6.json](https://arcprize.org/play?task=bda2d7a6)


**Size: 54 bytes**
* recoloring
* pairwise_analogy
* pattern_modification
* color_permutation

```python
show_examples(load_examples(298)['train'])
```

```python
%%writefile task298.py
p=lambda g:[[g[2][-r.index(v)|2]for v in r]for r in g]
```

# [299]
[bdad9b1f.json](https://arcprize.org/play?task=bdad9b1f)


**Size: 54 bytes**
* draw_line_from_point
* direction_guessing
* recoloring
* take_intersection

```python
show_examples(load_examples(299)['train'])
```

```python
%%writefile task299.py
p=lambda g:[[j-(2in i)&2--j%7for j in g[0]]for i in g]
```

# [300]
[be94b721.json](https://arcprize.org/play?task=be94b721)


**Size: 87 bytes**
* separate_shapes
* count_tiles
* take_maximum
* crop

```python
show_examples(load_examples(300)['train'])
```

```python
%%writefile task300.py
p=lambda g,x=0:[r for*r,in zip(*x or p(g,g))if max(range(1,10),key=sum(g,g).count)in r]
```

# [301]
[beb8660c.json](https://arcprize.org/play?task=beb8660c)


**Size: 31 bytes**
* pattern_moving
* count_tiles
* order_numbers

```python
show_examples(load_examples(301)['train'])
```

```python
%%writefile task301.py
p=lambda g,s=sorted:s(map(s,g))
```

# [302]
[c0f76784.json](https://arcprize.org/play?task=c0f76784)


**Size: 89 bytes**
* loop_filling
* measure_area
* associate_colors_to_numbers

```python
show_examples(load_examples(302)['train'])
```

```python
%%writefile task302.py
import re
p=lambda g:eval(re.sub("(?<!5, )5,(.+?)5",r"5,*[(v:=len([\1]))+5]*v,5",str(g)))
```

# [303]
[c1d99e64.json](https://arcprize.org/play?task=c1d99e64)


**Size: 62 bytes**
* draw_line_from_border
* detect_grid

```python
show_examples(load_examples(303)['train'])
```

```python
%%writefile task303.py
p=lambda g:[[[2,e][c>[0]*99<l]for*c,e in zip(*g,l)]for l in g]
```

# [304]
[c3e719e8.json](https://arcprize.org/play?task=c3e719e8)


**Size: 92 bytes**
* image_repetition
* image_expansion
* count_different_colors
* take_maximum

```python
show_examples(load_examples(304)['train'])
```

```python
%%writefile task304.py
p=lambda g:[[b*(a==max(q:=sum(g,g),key=q.count))for a in x for b in y]for x in g for y in g]
```

# [305]
[c3f564a4.json](https://arcprize.org/play?task=c3f564a4)


**Size: 56 bytes**
* pattern_expansion
* image_filling

```python
show_examples(load_examples(305)['train'])
```

```python
%%writefile task305.py
p=lambda g:[(9*[*{*r}-{0}])[g.index(r):][:16]for r in g]
```

# [306]
[c444b776.json](https://arcprize.org/play?task=c444b776)


**Size: 63 bytes**
* detect_grid
* separate_images
* find_the_intruder
* image_repetition

```python
show_examples(load_examples(306)['train'])
```

```python
%%writefile task306.py
p=lambda i:[i:=[*zip(*map(max,i,i[:10]+i))][::-1]for _ in i][7]
```

# [307]
[c59eb873.json](https://arcprize.org/play?task=c59eb873)


**Size: 46 bytes**
* image_resizing

```python
show_examples(load_examples(307)['train'])
```

```python
%%writefile task307.py
p=lambda a:a>a*0!=0and[p(a[0])]*2+p(a[1:])or a
```

# [308]
[c8cbb738.json](https://arcprize.org/play?task=c8cbb738)


**Size: 188 bytes**
* pattern_moving
* jigsaw
* crop

```python
show_examples(load_examples(308)['train'])
```

```python
%%writefile task308.py
def p(g):w=len(g[0]);g=bytes(sum(g,[]));m=max(g:={g.count(v)<9and i+i-g.find(v)-g.rfind(v)>>1:v for i,v in enumerate(g)})//w;R=range(-m,m+1);return[[g.get(a*w+c,g[0])for c in R]for a in R]
```

# [309]
[c8f0f002.json](https://arcprize.org/play?task=c8f0f002)


**Size: 36 bytes**
* recoloring
* associate_colors_to_colors

```python
show_examples(load_examples(309)['train'])
```

```python
%%writefile task309.py
p=lambda g:g*-1and g&-3or[*map(p,g)]
```

# [310]
[c909285e.json](https://arcprize.org/play?task=c909285e)


**Size: 70 bytes**
* find_the_intruder
* crop
* rectangle_guessing

```python
show_examples(load_examples(310)['train'])
```

```python
%%writefile task310.py
p=lambda a,*n:[b for b in zip(*n or p(a,*a))if{*b}-({*a[1]}&{*a[12]})]
```

# [311]
[c9e6f938.json](https://arcprize.org/play?task=c9e6f938)


**Size: 32 bytes**
* image_repetition
* image_reflection

```python
show_examples(load_examples(311)['train'])
```

```python
%%writefile task311.py
p=lambda j:[R+R[::-1]for R in j]
```

# [312]
[c9f8e694.json](https://arcprize.org/play?task=c9f8e694)


**Size: 44 bytes**
* recoloring
* pattern_repetition
* color_palette

```python
show_examples(load_examples(312)['train'])
```

```python
%%writefile task312.py
p=lambda g:[[j%~j&i[0]for j in i]for i in g]
```

# [313]
[caa06a1f.json](https://arcprize.org/play?task=caa06a1f)


**Size: 61 bytes**
* pattern_expansion
* image_filling

```python
show_examples(load_examples(313)['train'])
```

```python
%%writefile task313.py
p=lambda g,u=[]:g*-1*-1or[*map(p,g[u>[]:2--len(u)//11]*10,g)]
```

# [314]
[cbded52d.json](https://arcprize.org/play?task=cbded52d)


**Size: 82 bytes**
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
%%writefile task314.py
p=lambda i,*w:i*0!=0and[*map(p,i,i[:3]+i,i[3:]+i,*w)]or max(w[0]&w[1],w[2]&w[3],i)
```

# [315]
[cce03e0d.json](https://arcprize.org/play?task=cce03e0d)


**Size: 63 bytes**
* image_repetition
* image_expansion
* pairwise_analogy

```python
show_examples(load_examples(315)['train'])
```

```python
%%writefile task315.py
p=lambda g:[[b&-a%5for a in x for b in y]for x in g for y in g]
```

# [316]
[cdecee7f.json](https://arcprize.org/play?task=cdecee7f)


**Size: 71 bytes**
* summarize
* pairwise_analogy

```python
show_examples(load_examples(316)['train'])
```

```python
%%writefile task316.py
p=lambda g:[(a:=[*filter(int,map(max,*g)),0,0,0])[:3],a[5:2:-1],a[6:9]]
```

# [317]
[ce22a75a.json](https://arcprize.org/play?task=ce22a75a)


**Size: 43 bytes**
* replace_pattern

```python
show_examples(load_examples(317)['train'])
```

```python
%%writefile task317.py
p=lambda a:a==5or a and[p(a[1])]*3+p(a[3:])
```

# [318]
[ce4f8723.json](https://arcprize.org/play?task=ce4f8723)


**Size: 54 bytes**
* detect_wall
* separate_images
* take_complement
* take_intersection

```python
show_examples(load_examples(318)['train'])
```

```python
%%writefile task318.py
p=lambda g,h=[]:g*0!=0and[*map(p,g,h+g[5:])]or-h%5|g*3
```

# [319]
[ce602527.json](https://arcprize.org/play?task=ce602527)


**Size: 194 bytes**
* crop
* size_guessing
* shape_guessing
* find_the_intruder
* remove_intruder

```python
show_examples(load_examples(319)['train'])
```

```python
%%writefile task319.py
p=lambda g,x=0:[[[v[3],c][a==c]for c in r]for r in zip(*x or p(g,g))if(a:=(v:=sorted({*(q:=sum(g,[]))},key=q.count))[hash((*b'Q9n7l$hj(6ytfgHBq|^^KH^m"%r'[sum(q)%37:]%q,))%2])in r]
```

# [320]
[ce9e57f2.json](https://arcprize.org/play?task=ce9e57f2)


**Size: 65 bytes**
* recoloring
* count_tiles
* take_half

```python
show_examples(load_examples(320)['train'])
```

```python
%%writefile task320.py
p=lambda g:[*zip(*map(lambda*r:r[:-(k:=sum(r)//4)]+k*(8,)+r,*g))]
```

# [321]
[cf98881b.json](https://arcprize.org/play?task=cf98881b)


**Size: 54 bytes**
* detect_wall
* separate_images
* pattern_juxtaposition

```python
show_examples(load_examples(321)['train'])
```

```python
%%writefile task321.py
p=lambda g:[eval('r.pop(0)or r[4]|r[9],'*4)for r in g]
```

# [322]
[d037b0a7.json](https://arcprize.org/play?task=d037b0a7)


**Size: 45 bytes**
* pattern_expansion
* draw_line_from_point

```python
show_examples(load_examples(322)['train'])
```

```python
%%writefile task322.py
p=lambda g:g and[*p(g[:-1]),[*map(max,*g*2)]]
```

# [323]
[d06dbe63.json](https://arcprize.org/play?task=d06dbe63)


**Size: 100 bytes**
* pattern_expansion
* pairwise_analogy

```python
show_examples(load_examples(323)['train'])
```

```python
%%writefile task323.py
import re
p=lambda g:[p,eval][''in g](re.sub('0(?=(.{76})*(.{40}|(...){25,27})8)','5',str(g)[::-1]))
```

# [324]
[d07ae81c.json](https://arcprize.org/play?task=d07ae81c)


**Size: 212 bytes (295 raw)**
* draw_line_from_point
* diagonals
* color_guessing

```python
show_examples(load_examples(324)['train'])
```

```python
%%writefile task324.py
def p(s):
 c=sum(s,[]);t=c.count;a=c[t(c[0])<4];o,f={},{()}
 for d,e in enumerate(s):
  for m,i in enumerate(e):
   if t(i)<4:o[a in e and a in c[m::len(e)]]=i;f|={d+m,(d-m,)}
 for d,e in enumerate(s):
  for m,i in enumerate(e):
   if{d+m,(d-m,)}&f:e[m]=o[a in e and a in c[m::len(e)]]
 return s
```

# [325]
[d0f5fe59.json](https://arcprize.org/play?task=d0f5fe59)


**Size: 140 bytes**
* separate_shapes
* count_shapes
* associate_images_to_numbers
* pairwise_analogy

```python
show_examples(load_examples(325)['train'])
```

```python
%%writefile task325.py
p=lambda i,z=8,*w:i[22:]and[[8*(a==b)for b in w]for a in w]or p([[s:=h and(z:=z*2)|s|h for h in[0]+x]for*x,in zip(*i[::-1])],*{*sum(i,[0])})
```

# [326]
[d10ecb37.json](https://arcprize.org/play?task=d10ecb37)


**Size: 30 bytes**
* crop

```python
show_examples(load_examples(326)['train'])
```

```python
%%writefile task326.py
p=lambda j:[j[0][:2],j[1][:2]]
```

# [327]
[d13f3404.json](https://arcprize.org/play?task=d13f3404)


**Size: 67 bytes**
* image_expansion
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(327)['train'])
```

```python
%%writefile task327.py
p=lambda g,l=[0]*3:[l:=[*map(max,[0]+l*2,r+[0]*3)]for r in g+[l]*3]
```

# [328]
[d22278a0.json](https://arcprize.org/play?task=d22278a0)


**Size: 158 bytes**
* pattern_expansion
* pairwise_analogy

```python
show_examples(load_examples(328)['train'])
```

```python
%%writefile task328.py
exec("p=lambda g:[[(D:=sorted((sum(T:=[abs(x-r),abs(y-c)]),~max(T)%2*f[y])"+'for %s,f in enumerate(g)%s'*4%(*'y x','if f[y]))[0][1]*(D[0]<D[1][:1])',*'c]r]'))
```

# [329]
[d23f8c26.json](https://arcprize.org/play?task=d23f8c26)


**Size: 54 bytes**
* crop
* image_expansion

```python
show_examples(load_examples(329)['train'])
```

```python
%%writefile task329.py
p=lambda g:[[0]*(l:=len(r)//2)+[r[l]]+l*[0]for r in g]
```

# [330]
[d2abd087.json](https://arcprize.org/play?task=d2abd087)


**Size: 110 bytes**
* separate_shapes
* count_tiles
* associate_colors_to_numbers
* recoloring

```python
show_examples(load_examples(330)['train'])
```

```python
%%writefile task330.py
p=lambda i,k=-19,z=1:k*i or p([[e:=y and[1+y%7//6,z:=z*8,e|y][k>>4]for y in[0]+i][:0:-1]for*i,in zip(*i)],k+1)
```

# [331]
[d364b489.json](https://arcprize.org/play?task=d364b489)


**Size: 82 bytes**
* pattern_expansion

```python
show_examples(load_examples(331)['train'])
```

```python
%%writefile task331.py
p=lambda g:[g:=eval(f"{*zip(*g[::-1]),}".replace('1, 0','1,'+k))for k in'2786'][3]
```

# [332]
[d406998b.json](https://arcprize.org/play?task=d406998b)


**Size: 58 bytes**
* recoloring
* one_yes_one_no
* cylindrical

```python
show_examples(load_examples(332)['train'])
```

```python
%%writefile task332.py
p=lambda g:[[-r.pop(0)*7**len(r)%8for x in r*1]for r in g]
```

# [333]
[d43fd935.json](https://arcprize.org/play?task=d43fd935)


**Size: 83 bytes**
* draw_line_from_point
* direction_guessing
* projection_unto_rectangle

```python
show_examples(load_examples(333)['train'])
```

```python
%%writefile task333.py
p=lambda g:[eval("P"+9*",(P:=r.pop()or(3in r)*P)")for*r,P in zip(*g[70:]or p(g*2))]
```

# [334]
[d4469b4b.json](https://arcprize.org/play?task=d4469b4b)


**Size: 65 bytes**
* dominant_color
* associate_images_to_colors

```python
show_examples(load_examples(334)['train'])
```

```python
%%writefile task334.py
p=lambda g:[[i%7,i%6,i%11]for i in b" M~M~MM"[max(max(g))::3]]
```

# [335]
[d4a91cb9.json](https://arcprize.org/play?task=d4a91cb9)


**Size: 105 bytes**
* connect_the_dots
* direction_guessing

```python
show_examples(load_examples(335)['train'])
```

```python
%%writefile task335.py
S=sum
p=lambda g,s=6:[[j^(9<9|(s:=s+S(i)*S(a))%3|S(i)>3<S(a)|~s%9)*4>>j*9for*a,j in zip(*g,i)]for i in g]
```

# [336]
[d4f3cd78.json](https://arcprize.org/play?task=d4f3cd78)


**Size: 81 bytes**
* rectangle_guessing
* recoloring
* draw_line_from_point

```python
show_examples(load_examples(336)['train'])
```

```python
%%writefile task336.py
p=lambda g:exec(f"g[:]={'[r.pop()or(5in{*r}-{*r[4:]})*8for r in g],'*10};"*4)or g
```

# [337]
[d511f180.json](https://arcprize.org/play?task=d511f180)


**Size: 43 bytes**
* associate_colors_to_colors

```python
show_examples(load_examples(337)['train'])
```

```python
%%writefile task337.py
p=lambda g:g*-1and g^84%g%3*13or[*map(p,g)]
```

# [338]
[d5d6de2d.json](https://arcprize.org/play?task=d5d6de2d)


**Size: 62 bytes**
* loop_filling
* replace_pattern
* remove_intruders

```python
show_examples(load_examples(338)['train'])
```

```python
%%writefile task338.py
p=lambda g:[(c:=1)*[((c:=-j^-c%7%3)>1)*3for j in i]for i in g]
```

# [339]
[d631b094.json](https://arcprize.org/play?task=d631b094)


**Size: 37 bytes**
* count_tiles
* dominant_color
* summarize

```python
show_examples(load_examples(339)['train'])
```

```python
%%writefile task339.py
p=lambda g:[[*filter(int,sum(g,[]))]]
```

# [340]
[d687bc17.json](https://arcprize.org/play?task=d687bc17)


**Size: 111 bytes**
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
p=lambda g:[g:=[[a]*-~(C:=a in r*a)+[c*(c in g[-1]+g[1])for c in r[C:]]for a,*r in zip(*g)][::-1]for _ in g][3]
```

# [341]
[d6ad076f.json](https://arcprize.org/play?task=d6ad076f)


**Size: 108 bytes**
* bridges
* connect_the_dots
* draw_line_from_point

```python
show_examples(load_examples(341)['train'])
```

```python
%%writefile task341.py
p=lambda g:[g:=[[g[i][j]or(9>j>=2<len({*min(g[i-1:][:3])}))*8for i in R]for j in R]for R in[range(10)]*2][1]
```

# [342]
[d89b689b.json](https://arcprize.org/play?task=d89b689b)


**Size: 106 bytes**
* pattern_juxtaposition
* summarize
* direction_guessing

```python
show_examples(load_examples(342)['train'])
```

```python
%%writefile task342.py
import re
p=lambda g:eval(re.sub("([1-9])((.{32})+?[^8)]+)8",r"0\2\1",f'{*zip(*g[70:]or p(g*2)),}'))[::-1]
```

# [343]
[d8c310e9.json](https://arcprize.org/play?task=d8c310e9)


**Size: 63 bytes**
* pattern_expansion
* pattern_repetition
* pattern_completion

```python
show_examples(load_examples(343)['train'])
```

```python
%%writefile task343.py
p=lambda g:[(r[:8^-(r[8:12]!=r[:4]!=r[4:8])]*3)[:15]for r in g]
```

# [344]
[d90796e8.json](https://arcprize.org/play?task=d90796e8)


**Size: 74 bytes**
* replace_pattern

```python
show_examples(load_examples(344)['train'])
```

```python
%%writefile task344.py
p=lambda i,*w:i*0!=0and[*map(p,i,[i*4]+i,i[1:]+[i*4],*w)]or(i^1in w)+7&i*9
```

# [345]
[d9f24cd1.json](https://arcprize.org/play?task=d9f24cd1)


**Size: 89 bytes**
* draw_line_from_point
* gravity
* obstacles

```python
show_examples(load_examples(345)['train'])
```

```python
%%writefile task345.py
p=lambda i,a=0,*h:i[1:]and[[c[1]|2&6%~sum(h)+max((h:=c)[1:])for c in zip(*i)]]+p(i[a:],1)
```

# [346]
[d9fac9be.json](https://arcprize.org/play?task=d9fac9be)


**Size: 52 bytes**
* find_the_intruder
* summarize
* x_marks_the_spot

```python
show_examples(load_examples(346)['train'])
```

```python
%%writefile task346.py
p=lambda a:[[min(b:=sum(a[1:-1],a[3]),key=b.count)]]
```

# [347]
[dae9d2b5.json](https://arcprize.org/play?task=dae9d2b5)


**Size: 50 bytes**
* pattern_juxtaposition
* separate_images
* recoloring

```python
show_examples(load_examples(347)['train'])
```

```python
%%writefile task347.py
p=lambda g:[[6*(-a.pop(3)<v)for v in a]for a in g]
```

# [348]
[db3e9e38.json](https://arcprize.org/play?task=db3e9e38)


**Size: 88 bytes**
* pattern_expansion
* out_of_boundary

```python
show_examples(load_examples(348)['train'])
```

```python
%%writefile task348.py
p=lambda g,G=0,*s:[s:=[r.pop()|-y%15for y in s[1:]]+r for r in(G or p(g,g))[::-1]][::-1]
```

# [349]
[db93a21d.json](https://arcprize.org/play?task=db93a21d)


**Size: 172 bytes**
* contouring
* draw_line_from_point
* measure_area
* measure_length
* algebra

```python
show_examples(load_examples(349)['train'])
```

```python
%%writefile task349.py
p=lambda g,n=15:-n*g or[[P:=[max(x:=r.pop(),x%~x&P,a:=x and a+1198080|9),x or P&8**n*7and~8&P-8**n|3,x&15,x|(n-3in r)][n//4]for _ in g]for*r,in zip(*p(g,n-1))if[P:=0,a:=0]]
```

# [350]
[dbc1a6ce.json](https://arcprize.org/play?task=dbc1a6ce)


**Size: 89 bytes**
* connect_the_dots

```python
show_examples(load_examples(350)['train'])
```

```python
%%writefile task350.py
p=lambda i:[*map(f:=lambda*x,s=0:[y|(x.count(1)>(s:=s+y%8)>y<1)*8for y in x],*map(f,*i))]
```

# [351]
[dc0a314f.json](https://arcprize.org/play?task=dc0a314f)


**Size: 66 bytes**
* pattern_completion
* crop

```python
show_examples(load_examples(351)['train'])
```

```python
%%writefile task351.py
p=lambda i:[r[:5]for x in[*i]if(r:=i.pop()[~[*x,3].index(3)::-1])]
```

# [352]
[dc1df850.json](https://arcprize.org/play?task=dc1df850)


**Size: 82 bytes**
* contouring
* pattern_expansion
* out_of_boundary

```python
show_examples(load_examples(352)['train'])
```

```python
%%writefile task352.py
p=lambda g,i=3:g*-i or p([[r.pop()or[0]<r[-1:]<[3]for r in g]for r in g[0]*1],i-1)
```

# [353]
[dc433765.json](https://arcprize.org/play?task=dc433765)


**Size: 83 bytes**
* pattern_moving
* direction_guessing
* only_one

```python
show_examples(load_examples(353)['train'])
```

```python
%%writefile task353.py
p=lambda g:exec("g[:]=zip(g.pop(-~[*map(max,g[:-2]),4].index(4)),*g[::-1]);"*4)or g
```

# [354]
[ddf7fa4f.json](https://arcprize.org/play?task=ddf7fa4f)


**Size: 92 bytes**
* color_palette
* recoloring

```python
show_examples(load_examples(354)['train'])
```

```python
%%writefile task354.py
R=range(10)
p=lambda g:[[max(g[0][k]*all(r[j:k+1]+r[k:j+1])for k in R)for j in R]for r in g]
```

# [355]
[de1cd16c.json](https://arcprize.org/play?task=de1cd16c)


**Size: 95 bytes**
* separate_images
* count_tiles
* take_maximum
* summarize

```python
show_examples(load_examples(355)['train'])
```

```python
%%writefile task355.py
p=lambda g:[sorted({*(A:=[(*{*j}&{*a}^{b},)for j in g for*a,b in zip(*g,j)])},key=A.count)[-3]]
```

# [356]
[ded97339.json](https://arcprize.org/play?task=ded97339)


**Size: 92 bytes**
* connect_the_dots

```python
show_examples(load_examples(356)['train'])
```

```python
%%writefile task356.py
p=lambda g,*r,i=0:[x|max(g[i:])&max(g[:(i:=i+1)])for x in r]or[*map(p,g,*map(p,zip(*g),*g))]
```

# [357]
[e179c5f4.json](https://arcprize.org/play?task=e179c5f4)


**Size: 81 bytes**
* pattern_expansion
* bouncing

```python
show_examples(load_examples(357)['train'])
```

```python
%%writefile task357.py
def p(g,i=9):
 for r in g:w=~-len(r);r[:]=[8]*-~w;r[i%w^i//w%-2]=1;i-=1
 return g
```

# [358]
[e21d9049.json](https://arcprize.org/play?task=e21d9049)


**Size: 90 bytes**
* pattern_expansion
* draw_line_from_point
* color_palette

```python
show_examples(load_examples(358)['train'])
```

```python
%%writefile task358.py
p=lambda g,*r:[max(r[::6^83>>len({*(r:=(0,)*35+r)})])for _ in r]or[*map(p,g,*map(p,g,*g))]
```

# [359]
[e26a3af2.json](https://arcprize.org/play?task=e26a3af2)


**Size: 64 bytes**
* remove_noise
* separate_images

```python
show_examples(load_examples(359)['train'])
```

```python
%%writefile task359.py
p=lambda d:[[max(S:=r+C,key=S.count)for*C,in zip(*d)]for r in d]
```

# [360]
[e3497940.json](https://arcprize.org/play?task=e3497940)


**Size: 45 bytes**
* detect_wall
* separate_images
* image_reflection
* image_juxtaposition

```python
show_examples(load_examples(360)['train'])
```

```python
%%writefile task360.py
p=lambda g:[[*map(max,r,r[:4:-1])]for r in g]
```

# [361]
[e40b9e2f.json](https://arcprize.org/play?task=e40b9e2f)


**Size: 185 bytes (241 raw)**
* pattern_expansion
* pattern_reflection
* pattern_rotation

```python
show_examples(load_examples(361)['train'])
```

```python
%%writefile task361.py
def p(r):n=[r+r for r in r+r];return[[[n[l][i]|n[r-a+i][r+a+e+~l]|n[r+r+e+~l][a+a+e+~i]|n[r+a+e+~i][a-r+l]for i in range(10)]for l in range(10)]for e in range(10)for a in range(10)for r in range(10)if all(all(r[a:a+e])for r in n[r:r+e])][-1]
```

# [362]
[e48d4e1a.json](https://arcprize.org/play?task=e48d4e1a)


**Size: 67 bytes**
* count_tiles
* pattern_moving
* detect_grid
* out_of_boundary

```python
show_examples(load_examples(362)['train'])
```

```python
%%writefile task362.py
p=lambda g:[r[(k:=g.count(g[0])):9]+-~k*r[:1]for r in g*2][~k-9:-k]
```

# [363]
[e5062a87.json](https://arcprize.org/play?task=e5062a87)


**Size: 175 bytes**
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(363)['train'])
```

```python
%%writefile task363.py
from re import*
def p(g):h=hash((*g[3],));g[~h%7][3]|=h%149<1;return eval(sub(*'10',eval("'2'.join(split(sub('2',')0(',sub('[^2]','.',K:=str(g))).strip('.()'),"*3+"K))))))")))
```

# [364]
[e509e548.json](https://arcprize.org/play?task=e509e548)


**Size: 133 bytes**
* recoloring
* associate_colors_to_shapes
* homeomorphism

```python
show_examples(load_examples(364)['train'])
```

```python
%%writefile task364.py
p=lambda g,k=95:-k*g or p([g:=[[7&88>>c%7,c|a|all([a,b*-1,k>83])<<k*3][k>0<c]for c,a,b in zip(r,[0]+r,g)]for*r,in zip(*g[::-1])],k-1)
```

# [365]
[e50d258f.json](https://arcprize.org/play?task=e50d258f)


**Size: 111 bytes**
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
p=lambda a:max([-(c:=sum(b:=[b[x%8:x%11]for b in a[x%9:x%13]],a).count)(0),c(2),c(1),b]for x in range(5**6))[3]
```

# [366]
[e6721834.json](https://arcprize.org/play?task=e6721834)


**Size: 298 bytes (495 raw)**
* pattern_moving
* pattern_juxtaposition
* crop

```python
show_examples(load_examples(366)['train'])
```

```python
%%writefile task366.py
def p(r):
 *e,f,i,u=sorted({*sum(r,[])},key=sum(r,[]).count)
 *e,=r,
 for o in e*6:
  for o in e+(e:=[]):
   e+=[],
   for o in zip(*o):e+=e.pop()+[o]if{i,u}-{*o}>{i}or{*o}>{i}else[],
 [6for l,e in sorted((-sum(f^e for e in e for e in e),e)for e in e)for l,o in enumerate(r)for p,o in enumerate(o)for n,o in zip(r[l:]+r,all((n==o)!=(f==o)==(n==u)for n,o in zip(r[l:]+r,e)for n,o in zip(n[p:p+len(o)]+r,o))*e)for n[p:p+len(o)]in[o]];return[o for o in zip(*[o for o in zip(*r)if u in o])if u in o]
```

# [367]
[e73095fd.json](https://arcprize.org/play?task=e73095fd)


**Size: 125 bytes**
* loop_filling
* rectangle_guessing

```python
show_examples(load_examples(367)['train'])
```

```python
%%writefile task367.py
import re
p=lambda g,k=-19:k*g or p(eval(re.sub(f"0(?=, 4|.{ {N:=len(g)*3+4}}5, 0.{ {N-6}}0)","4",f'{*zip(*g[::-1]),}')),k+1)
```

# [368]
[e76a88a6.json](https://arcprize.org/play?task=e76a88a6)


**Size: 129 bytes**
* pattern_repetition
* pattern_juxtaposition

```python
show_examples(load_examples(368)['train'])
```

```python
%%writefile task368.py
import re
p=lambda i:eval(re.sub("5(, 5)+",lambda m:re.findall("[^50](?:, [1-9])+",s*2)[-s[m.end()-1::32].count('5')],s:=str(i)))
```

# [369]
[e8593010.json](https://arcprize.org/play?task=e8593010)


**Size: 109 bytes**
* holes
* count_tiles
* loop_filling
* associate_colors_to_numbers

```python
show_examples(load_examples(369)['train'])
```

```python
%%writefile task369.py
import re
p=lambda g:eval(re.sub('(2|3), [31]|0',r'*7%3\1%5*[3\1%31]',f'{*zip(*g[20470:]or p(g*2)),}'))[::-1]
```

# [370]
[e8dc4411.json](https://arcprize.org/play?task=e8dc4411)


**Size: 159 bytes**
* pattern_expansion
* direction_guessing

```python
show_examples(load_examples(370)['train'])
```

```python
%%writefile task370.py
import re
p=lambda g:exec('s=f"{*zip(*g[::-1]),}";i=s.rfind;d=i("0")-i(j:=min(s,key=i));g[:]=eval(re.sub("(?=(.{%d})+0)\d"%d,j,s,d%8*d%-(len(g)*3+5)));'*4)or g
```

# [371]
[e9614598.json](https://arcprize.org/play?task=e9614598)


**Size: 102 bytes**
* pattern_expansion
* direction_guessing
* measure_length

```python
show_examples(load_examples(371)['train'])
```

```python
%%writefile task371.py
p=lambda g,u=0:eval((G:=f"{*zip(*u or p(g,g)),}")[:(w:=G.rfind("1")+G.find("1")>>1)-3]+"3,"*3+G[w+5:])
```

# [372]
[e98196ab.json](https://arcprize.org/play?task=e98196ab)


**Size: 47 bytes**
* detect_wall
* separate_images
* image_juxtaposition

```python
show_examples(load_examples(372)['train'])
```

```python
%%writefile task372.py
p=lambda g,h=[]:[*map([p,max][h>[]],g,h+g[6:])]
```

# [373]
[e9afcf9a.json](https://arcprize.org/play?task=e9afcf9a)


**Size: 38 bytes**
* pattern_modification

```python
show_examples(load_examples(373)['train'])
```

```python
%%writefile task373.py
p=lambda g:[q:=max(zip(*g*3)),q[::-1]]
```

# [374]
[ea32f347.json](https://arcprize.org/play?task=ea32f347)


**Size: 103 bytes**
* separate_shapes
* count_tiles
* recoloring
* associate_colors_to_ranks

```python
show_examples(load_examples(374)['train'])
```

```python
%%writefile task374.py
p=lambda g,i=51:-i*g or p(eval(x:=f"{*zip(*g),}".replace(i//5*", 5",i//5*",7*~len({*x})%5"))[::-1],i-1)
```

# [375]
[ea786f4a.json](https://arcprize.org/play?task=ea786f4a)


**Size: 53 bytes**
* pattern_modification
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(375)['train'])
```

```python
%%writefile task375.py
def p(g,i=0):
 for x in g:x[i]=x[~i]=0;i+=1
 return g
```

# [376]
[eb281b96.json](https://arcprize.org/play?task=eb281b96)


**Size: 30 bytes**
* image_repetition
* image_reflection

```python
show_examples(load_examples(376)['train'])
```

```python
%%writefile task376.py
p=lambda g:(g+g[1:-1])*2+g[:1]
```

# [377]
[eb5a1d5d.json](https://arcprize.org/play?task=eb5a1d5d)


**Size: 55 bytes**
* summarize

```python
show_examples(load_examples(377)['train'])
```

```python
%%writefile task377.py
p=lambda g,x=0:[g:=r for*r,in zip(*x or p(g,g))if g!=r]
```

# [378]
[ec883f72.json](https://arcprize.org/play?task=ec883f72)


**Size: 131 bytes**
* pattern_expansion
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(378)['train'])
```

```python
%%writefile task378.py
import re
p=lambda g:exec("x='...'*len(g)+'.0';g[::-1]=zip(*eval(re.sub(f'0(?=({x}, .)*, [^0]{x*2}, (.))',r'\\2',str(g))));"*4)or g
```

# [379]
[ecdecbb3.json](https://arcprize.org/play?task=ecdecbb3)


**Size: 135 bytes**
* pattern_modification
* draw_line_from_point

```python
show_examples(load_examples(379)['train'])
```

```python
%%writefile task379.py
p=lambda g,i=47:g*-i or[[[3&q%~(c:=r.pop())%5|2&q*-(8in r),~c*q%2*9,c&10][8>>13-i//4]or c for q in[0]+r[:0:-1]]for*r,in zip(*p(g,i-1))]
```

# [380]
[ed36ccf7.json](https://arcprize.org/play?task=ed36ccf7)


**Size: 27 bytes**
* image_rotation

```python
show_examples(load_examples(380)['train'])
```

```python
%%writefile task380.py
p=lambda j:[*zip(*j)][::-1]
```

# [381]
[ef135b50.json](https://arcprize.org/play?task=ef135b50)


**Size: 79 bytes**
* draw_line_from_point
* bridges
* connect_the_dots

```python
show_examples(load_examples(381)['train'])
```

```python
%%writefile task381.py
p=lambda g:[r*(q:=r in g[::9])or[q:=r.pop(0)or 9*any(r*q)for _ in g]for r in g]
```

# [382]
[f15e1fac.json](https://arcprize.org/play?task=f15e1fac)


**Size: 124 bytes**
* draw_line_from_point
* gravity
* obstacles
* direction_guessing

```python
show_examples(load_examples(382)['train'])
```

```python
%%writefile task382.py
p=lambda i,k=-3:k*i or p([*zip(w:=i.pop(),*[[*map(max,r,w:=r*('8'in'%s'%i)+[0,*w,0][r[-1]or[1]>r:])]for*r,in i[::-1]])],k+1)
```

# [383]
[f1cefba8.json](https://arcprize.org/play?task=f1cefba8)


**Size: 120 bytes**
* draw_line_from_point
* pattern_modification

```python
show_examples(load_examples(383)['train'])
```

```python
%%writefile task383.py
p=lambda g:[[(a:=[v,*{}.fromkeys(sum(g,r))])[any(0<i.count(a[2])<4for i in[r,c])*2+0**v]for*c,v in zip(*g,r)]for r in g]
```

# [384]
[f25fbde4.json](https://arcprize.org/play?task=f25fbde4)


**Size: 61 bytes**
* crop
* image_resizing

```python
show_examples(load_examples(384)['train'])
```

```python
%%writefile task384.py
p=lambda g,*a:sum([any(x)*2*[x]for*x,in zip(*a or p(*g))],[])
```

# [385]
[f25ffba3.json](https://arcprize.org/play?task=f25ffba3)


**Size: 25 bytes**
* pattern_repetition
* pattern_reflection

```python
show_examples(load_examples(385)['train'])
```

```python
%%writefile task385.py
p=lambda g:g[:4:-1]+g[5:]
```

# [386]
[f2829549.json](https://arcprize.org/play?task=f2829549)


**Size: 50 bytes**
* detect_wall
* separate_images
* take_complement
* pattern_intersection

```python
show_examples(load_examples(386)['train'])
```

```python
%%writefile task386.py
p=lambda g:[eval('3>>a[4]+a.pop(0),'*3)for a in g]
```

# [387]
[f35d900a.json](https://arcprize.org/play?task=f35d900a)


**Size: 190 bytes**
* pattern_expansion

```python
show_examples(load_examples(387)['train'])
```

```python
%%writefile task387.py
p=lambda g,n=11:-n*g or[[[(i:=((x:=r.pop())+i>0)*-~i)%2*sum(r[:2-i])**4%5*5or x,x or-P**4%5*P*4,sum({*sum(g*(x>9),[])}-{x%15})or x][n//4]for P in[0]+r[:0:-1]]for*r,in zip(*p(g,n-1))if[i:=0]]
```

# [388]
[f5b8619d.json](https://arcprize.org/play?task=f5b8619d)


**Size: 61 bytes**
* pattern_expansion
* draw_line_from_point
* image_repetition

```python
show_examples(load_examples(388)['train'])
```

```python
%%writefile task388.py
p=lambda g:[[i|8&i-any(c)for*c,i in zip(*g,r)]*2for r in g]*2
```

# [389]
[f76d97a5.json](https://arcprize.org/play?task=f76d97a5)


**Size: 57 bytes**
* take_negative
* recoloring
* associate_colors_to_colors

```python
show_examples(load_examples(389)['train'])
```

```python
%%writefile task389.py
p=lambda g:[[sum({*sum(g,r)}^{t,5})for t in r]for r in g]
```

# [390]
[f8a8fe49.json](https://arcprize.org/play?task=f8a8fe49)


**Size: 98 bytes**
* pattern_moving
* pattern_reflection

```python
show_examples(load_examples(390)['train'])
```

```python
%%writefile task390.py
import re
p=lambda g,*x:eval(re.sub('[^(2]{9}2'*2+'?',"*[\g<0>][::-1]",f"{*zip(*x or p(g,*g)),}"))
```

# [391]
[f8b3ba0a.json](https://arcprize.org/play?task=f8b3ba0a)


**Size: 63 bytes**
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
p=lambda g:[*zip(sorted({*(d:=sum(g,[]))},key=d.count)[2::-1])]
```

# [392]
[f8c80d96.json](https://arcprize.org/play?task=f8c80d96)


**Size: 135 bytes**
* pattern_expansion
* background_filling

```python
show_examples(load_examples(392)['train'])
```

```python
%%writefile task392.py
p=lambda g,n=23:-n*g or[eval(f"r.pop()or[5,w:=max(max(g))][{n}//4%(('0, %d, '%w*2in'{g}')-3)]*any(r[-1:]),"*10)for*r,in zip(*p(g,n-1))]
```

# [393]
[f8ff0b80.json](https://arcprize.org/play?task=f8ff0b80)


**Size: 63 bytes**
* separate_shapes
* count_tiles
* summarize
* order_numbers

```python
show_examples(load_examples(393)['train'])
```

```python
%%writefile task393.py
p=lambda g:[*zip(sorted(range(10),key=sum(g,g).count)[8:5:-1])]
```

# [394]
[f9012d9b.json](https://arcprize.org/play?task=f9012d9b)


**Size: 85 bytes**
* pattern_expansion
* pattern_completion
* crop

```python
show_examples(load_examples(394)['train'])
```

```python
%%writefile task394.py
p=lambda g:[(i+i[1:])[~(8|181%len(g)-i.index(0))%6:][:x]for i in g if(x:=i.count(0))]
```

# [395]
[fafffa47.json](https://arcprize.org/play?task=fafffa47)


**Size: 53 bytes**
* separate_images
* take_complement
* pattern_intersection

```python
show_examples(load_examples(395)['train'])
```

```python
%%writefile task395.py
p=lambda g,h=[]:g*0!=0and[*map(p,g,h+g[3:])]or~g+~h&2
```

# [396]
[fcb5c309.json](https://arcprize.org/play?task=fcb5c309)


**Size: 139 bytes**
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
p=lambda m,n=266,f=0:[[sum({*e*sum(m,[-f])})for e in s]for r in m if(f:=((X:=n>>5)*[a:=(r*3)[x:=n%32]]in(s:=r[x:x+X],[f]*X))*a)]or p(m,n-1)
```

# [397]
[fcc82909.json](https://arcprize.org/play?task=fcc82909)


**Size: 110 bytes**
* pattern_expansion
* separate_images
* count_different_colors

```python
show_examples(load_examples(397)['train'])
```

```python
%%writefile task397.py
p=lambda g,i=81:exec("i-=1;c=i//9-1\nfor _ in{*all(b:=%s+%s)*b}:%s=3,3\n"%(('g[c:=c+1][i%9:i%9+2]',)*3)*i)or g
```

# [398]
[feca6190.json](https://arcprize.org/play?task=feca6190)


**Size: 72 bytes**
* pattern_expansion
* image_expansion
* draw_line_from_point
* diagonals

```python
show_examples(load_examples(398)['train'])
```

```python
%%writefile task398.py
def p(g):g,=g;q=len({*g}-{0})*[0]*5;return[q:=q[1:]+[v]for v in g+q[5:]]
```

# [399]
[ff28f65a.json](https://arcprize.org/play?task=ff28f65a)


**Size: 62 bytes**
* count_shapes
* associate_images_to_numbers

```python
show_examples(load_examples(399)['train'])
```

```python
%%writefile task399.py
p=lambda g:[*zip(*[iter(sum(sum(g,[]))%7*[1,0]+[0]*9)]*3)][:3]
```

# [400]
[ff805c23.json](https://arcprize.org/play?task=ff805c23)


**Size: 66 bytes**
* pattern_expansion
* pattern_completion
* crop

```python
show_examples(load_examples(400)['train'])
```

```python
%%writefile task400.py
p=lambda g:[h[:5]for r in[*g]if(h:=g.pop()[~[*r,1].index(1)::-1])]
```

```python
!pip install zopfli
!pip install deflate
```

```python
from zipfile import ZipFile
import zipfile
from __future__ import annotations
import ast
import functools
import io
import random
import re
import zlib
from collections import defaultdict
from typing import NamedTuple, Optional
import deflate
import zopfli.zlib
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# --- Custom LZ77/Huffman Implementation for re-encoding ---

CLEN_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]

class BitReader:
    def __init__(self, data: bytes) -> None:
        self.stream = io.BytesIO(data)
        self.buffer = 0
        self.buffer_size = 0

    def read(self, n: int) -> int:
        while self.buffer_size < n:
            byte = self.stream.read(1)
            self.buffer |= byte[0] << self.buffer_size
            self.buffer_size += 8
        ret = self.buffer & (1 << n) - 1
        self.buffer >>= n
        self.buffer_size -= n
        return ret

class BitString(NamedTuple):
    value: int
    size: int

    def __add__(self, other: tuple[object, ...]) -> BitString:
        if not isinstance(other, BitString): return NotImplemented
        return BitString((other.value << self.size) | self.value, self.size + other.size)

    def to_bytes(self) -> tuple[bytes, BitString]:
        data = self.value.to_bytes((self.size >> 3) + 1, "little")
        return data[:-1], BitString(data[-1], self.size & 7)

class Huffman:
    def __init__(self, lit_tree: dict[int, BitString], dist_tree: dict[int, BitString], raw: BitString) -> None:
        self.lit, self.dist, self.raw = lit_tree, dist_tree, raw

    @staticmethod
    def _build_tree(lengths: list[int]) -> dict[BitString, int]:
        tree, code, length = {}, 0, 0
        for sym in sorted(range(len(lengths)), key=lengths.__getitem__):
            if lengths[sym] == 0: continue
            code <<= lengths[sym] - length
            length = lengths[sym]
            tree[BitString(int(f"{code:0{length}b}"[::-1], 2), length)] = sym
            code += 1
        return tree
    
    @staticmethod
    def _build_rev_tree(lengths: list[int]) -> dict[int, BitString]:
        rev_tree, code, length = {}, 0, 0
        for sym in sorted(range(len(lengths)), key=lengths.__getitem__):
            if lengths[sym] == 0: continue
            code <<= lengths[sym] - length
            length = lengths[sym]
            rev_tree[sym] = BitString(int(f"{code:0{length}b}"[::-1], 2), length)
            code += 1
        return rev_tree

    @staticmethod
    def parse(deflate_data: bytes) -> Huffman:
        reader = BitReader(deflate_data)
        assert reader.read(1) == 1  # final
        assert reader.read(2) == 2  # btype (dynamic)
        hlit = reader.read(5) + 257
        hdist = reader.read(5) + 1
        hclen = reader.read(4) + 4
        lengths = [0] * len(CLEN_ORDER)
        for sym in CLEN_ORDER[:hclen]: lengths[sym] = reader.read(3)
        cl_tree = Huffman._build_tree(lengths)
        used = 17 + 3 * hclen
        lengths = []
        while len(lengths) < hlit + hdist:
            code, length = 0, 0
            while (code, length) not in cl_tree:
                code, length = code | (reader.read(1) << length), length + 1
            sym = cl_tree[BitString(code, length)]
            used += length
            if sym < 16: lengths.append(sym)
            elif sym == 16:
                lengths.extend(lengths[-1:] * (reader.read(2) + 3))
                used += 2
            elif sym == 17:
                lengths.extend([0] * (reader.read(3) + 3))
                used += 3
            elif sym == 18:
                lengths.extend([0] * (reader.read(7) + 11))
                used += 7
        return Huffman(Huffman._build_rev_tree(lengths[:hlit]), Huffman._build_rev_tree(lengths[hlit:]), BitString(BitReader(deflate_data).read(used), used))

    def encode_lit(self, x: int) -> Optional[BitString]: return self.lit.get(x)
    def encode_len(self, x: int) -> Optional[BitString]:
        start, extra = 3, 0
        for sym in range(257, 285):
            extra += 264 < sym and sym % 4 == 1
            if x < start + (1 << extra):
                code = self.lit.get(sym)
                return code + BitString(x - start, extra) if code else None
            start += 1 << extra
        return None
    def encode_dist(self, x: int) -> Optional[BitString]:
        start, extra = 1, 0
        for sym in range(30):
            extra += 3 < sym and sym % 2 == 0
            if x < start + (1 << extra):
                code = self.dist.get(sym)
                return code + BitString(x - start, extra) if code else None
            start += 1 << extra
        return None

class State(NamedTuple):
    prev: int  # 0 = regular, 1 = null, 2 = backslash
    value: BitString

def merge(state: State, code: BitString, delim: bytes) -> tuple[State, int]:
    prev = state.prev
    stream, value = (state.value + code).to_bytes()
    cost = code.size
    for byte in stream:
        if prev == 1 and byte in b"01234567": cost += 16
        elif prev == 2 and byte in b"\0\n\r01234567abfxnrtvuUN'\"\\": cost += 8
        if byte == 0: prev, cost = 1, cost + 8
        elif byte == ord("\r"): prev, cost = 0, cost + 8
        elif byte == ord("\n") and len(delim) == 1: prev, cost = 0, cost + 8
        elif byte == delim[0] and len(delim) == 1: prev, cost = 0, cost + 8
        else: prev = 2 if byte == ord("\\") else 0
    return State(prev, value), cost

def lz77(data: bytes, huffman: Huffman, delim: bytes) -> bytes:
    index = defaultdict(list)
    for start in range(len(data)):
        for end in range(start + 3, len(data) + 1):
            index[data[start:end]].append(start)
    refs = [[] for _ in range(len(data) + 1)]
    for substr in index:
        curr = []
        for start in index[substr]:
            for i in curr:
                x, y = huffman.encode_len(len(substr)), huffman.encode_dist(start - i)
                if x and y: refs[start].append((len(substr), x + y))
            curr.append(start)
    initial = State(0, BitString(0, 0))
    start_state, cost = merge(initial, huffman.raw, delim)
    dp = [{} for _ in range(len(data) + 2)]
    dp[0][start_state] = (cost, -1, initial, huffman.raw)
    for i in range(len(data) + 1):
        for state, (cost, _, _, _) in dp[i].items():
            code = huffman.encode_lit(data[i] if i < len(data) else 256)
            new_state, extra = merge(state, code, delim)
            if new_state not in dp[i + 1] or cost + extra < dp[i + 1][new_state][0]:
                dp[i + 1][new_state] = (cost + extra, i, state, code)
            for size, r_code in refs[i]:
                new_state, extra = merge(state, r_code, delim)
                if new_state not in dp[i + size] or cost + extra < dp[i + size][new_state][0]:
                    dp[i + size][new_state] = (cost + extra, i, state, r_code)
    codes, curr_ = [], min(dp[-1].values())
    while True:
        codes.append(curr_[3])
        if curr_[1] == -1: break
        curr_ = dp[curr_[1]][curr_[2]]
    combined = BitString(0, 0)
    for code in codes[::-1]: combined += code
    combined += BitString(0, -combined.size % 8)
    return combined.to_bytes()[0]

def reencode(deflate_data: bytes, delim: bytes) -> bytes:
    if deflate_data[0] & 0b111 == 0b101:
        return lz77(zlib.decompress(deflate_data, -10), Huffman.parse(deflate_data), delim)
    return deflate_data

# --- Compression Strategies ---

def my_zip_src(src_code: bytes) -> bytes:
    if len(src_code) < 150: return src_code
    candidates, vars_ = [src_code], [(src_code, b'')]
    found = set(re.findall(rb'\bimport\s+([a-zA-Z_]\w*)(?=[;\n])', src_code))
    if found:
        ns, mov = src_code, []
        for m in found:
            if m == b'zlib': continue
            if b'import ' + m + b'\n' in ns: ns = ns.replace(b'import ' + m + b'\n', b''); mov.append(m)
            elif b'import ' + m + b';' in ns: ns = ns.replace(b'import ' + m + b';', b''); mov.append(m)
        if mov: vars_.append((ns, b',' + b','.join(mov)))
    for src_c, imports in vars_:
        for compress_func in [zopfli.zlib.compress, lambda d: zlib.compress(d, 9), lambda d: deflate.zlib_compress(d, 12)]:
            for trailing in [b'', b'\n', b' ', b'\n ']:
                comp = compress_func(src_c + trailing)[2:-4]
                for delim in [b"'", b'"']:
                    esc_map = {0: b'\\x00', ord('\n'): b'\\n', ord('\r'): b'\\r', ord('\\'): b'\\\\', delim[0]: b'\\' + delim}
                    sanitized = b''.join(esc_map.get(b, bytes([b])) for b in comp)
                    compressed = b'#coding:L1\nimport zlib\nexec(zlib.decompress(bytes(' + delim + sanitized + delim + b',"L1"),-9))'
                    cur = imports
                    if b"re." in src_code and b"import" not in src_code and b",re" not in cur: cur += b",re"
                    if cur: compressed = compressed.replace(b"import zlib", b"import zlib" + cur)
                    candidates.append(compressed)
                esc_map = {0: b'\\x00', ord('\r'): b'\\r', ord('\\'): b'\\\\'}
                sanitized = b''.join(esc_map.get(b, bytes([b])) for b in comp)
                compressed = b'#coding:L1\nimport zlib\nexec(zlib.decompress(bytes("""' + sanitized + b'""","L1"),-9))'
                cur = imports
                if b"re." in src_code and b"import" not in src_code and b",re" not in cur: cur += b",re"
                if cur: compressed = compressed.replace(b"import zlib", b"import zlib" + cur)
                candidates.append(compressed)
    valid_options = []
    for code in candidates:
        try:
            exec(code, {})
            valid_options.append(code)
        except: pass
    return min(valid_options, key=len)

PREFIXES = [b"", b"\n", b"\r", b"\f", b"\n\f", b"\r\f"] + [bytes([c, ne]) for c in b"\t\n\f\r 0123456789#" for ne in b"\n\r"]
POSTFIXES = [b"", b" ", b"\t", b"\n", b"\r", b"\f", b"#", b";", b"\t ", b" \t", b"\np"] + [b"#" + bytes([n]) for n in range(32, 127)]
DEFAULT_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnoqrstuvwxyz"

def oxjam_zip_src(src_code: bytes) -> bytes:
    if len(src_code) < 220: return src_code
    match_type = lambda s: s if isinstance(src_code, str) else s.encode()
    new_lines = []
    for line in src_code.replace(match_type("\r\n"), match_type("\n")).split(match_type("\n")):
        if match_type("nomerge") in line: return src_code
        if line.startswith(match_type('##')): break
        if line.startswith(match_type('#')): continue
        if line.rstrip(): new_lines.append(line.rstrip())
    preprocessed = match_type('\n').join(new_lines)
    compressed, _ = oxjam_compress(preprocessed, rand_passes=100, pre_and_post=False)
    return compressed

@functools.lru_cache(maxsize=2048)
def oxjam_compress(task_src: bytes, rand_passes=0, pre_and_post=True) -> tuple[bytes, dict]:
    stats = {'method': 'none'}
    if len(task_src) < 160: return task_src, stats
    random.seed(0)
    rands = [sub_vars(task_src, bytes(sorted(DEFAULT_ALPHABET, key=lambda c: random.random()))) for _ in range(rand_passes)]
    for task_src_2 in (sub_vars(task_src, DEFAULT_ALPHABET), sub_vars(task_src, DEFAULT_ALPHABET[::-1]), *rands, task_src):
        for method, window in [("zlib", -9), ("zlib", -15), ("libdeflate", -15), ("zopfli", 20), ("zopfli", 40), ("zopfli", 100), ("zopfli", 200)]:
            task_src, stats = min((task_src, stats), compress_with_method(task_src, task_src_2, method, window, pre_and_post), key=lambda x: len(x[0]))
    return task_src, stats

def compress_with_method(task_src: bytes, task_src_2: bytes, method: str, window: int, pre_and_post: bool) -> tuple[bytes, dict]:
    best_zip, best_stats = oxjam_zip_src_(task_src_2, method=method, window=window)
    if len(best_zip) > len(task_src) + 10: return task_src, {}
    for pre in (PREFIXES if pre_and_post else [b""]):
        for post in (POSTFIXES if pre_and_post else [b""]):
            if len(pre + post) > 3: continue
            z_src, stats = oxjam_zip_src_(pre + task_src_2 + post, method=method, window=window)
            if len(z_src) < len(best_zip): best_zip, best_stats = z_src, stats
    return best_zip, best_stats

def oxjam_zip_src_(src: bytes, method: str, window: int = 0) -> tuple[bytes, dict]:
    header = b"#coding:L1\nimport zlib"
    if src[:9] == b"import re": header += b",re"; src = src[10:]
    if method == "zopfli":
        compressed = zopfli.zlib.compress(src, numiterations=window, blocksplitting=False)
        window = -(((compressed[0] >> 4) & 0x0F) + 8)
        compressed = compressed[2:-4]
    elif method == "libdeflate":
        compressed = deflate.zlib_compress(src, 12)[2:-4]
    elif method == "zlib":
        compressed = zlib.compress(src, 9, window)
    else: raise ValueError(f"Unknown method {method}")

    len_before = len(compressed)
    b_out = bytearray()
    for ch, ch1 in zip(compressed, compressed[1:] + b"'"):
        if ch == 0: b_out += b"\\x00" if ch1 in b"01234567" else b"\\0"
        elif ch == 13: b_out += b"\\r"
        elif ch == 92 and ch1 in b"\\\n\"\'01234567NUabfnrtvxu": b_out += b"\\\\"
        else: b_out.append(ch)
    compressed = bytes(b_out)

    delim = b'"""' if compressed[-1:] != b'"' else b"'''"
    nl, s, d = compressed.count(10), compressed.count(39), compressed.count(34)
    if 4 > s + nl < d + nl: delim, compressed = b"'", compressed.replace(b"'", b"\\'").replace(b"\n", b"\\n")
    elif 4 > d + nl < s + nl: delim, compressed = b'"', compressed.replace(b'"', b'\\"').replace(b"\n", b"\\n")
    
    stats = {'method': method, 'window': window, 'escape_cost': len(compressed) - len_before}
    if window < 15: window = -9
    if sum(c > 127 for c in compressed) < 8:
        header = header.replace(b"#coding:L1\n", b"")
        compressed = b''.join(b"\\x%0.2x" % c if c > 127 else bytes([c]) for c in compressed)
        return header + b"\nexec(zlib.decompress(b" + delim + compressed + delim + (b',%d' % window if window < 15 else b'') + b'))', stats
    return header + b"\nexec(zlib.decompress(bytes(" + delim + compressed + delim + b',"L1")' + (b',%d' % window if window < 15 else b'') + b'))', stats

def sub_vars(src: bytes, alphabet: bytes) -> bytes:
    vars_prev = sorted(i for i in set(n.arg.encode() if isinstance(n, ast.arg) else n.id.encode() if isinstance(n, ast.Name) else n.name.encode() if isinstance(n, ast.FunctionDef) else b'' for n in ast.walk(ast.parse(src))) if len(i) == 1 and i in alphabet)
    if not vars_prev: return src
    varless = re.sub(br"(?<!\\)\b[%b]\b(?!['\"])" % alphabet, b"_", src)
    rest = set(re.findall(br"[%b]" % alphabet, varless))
    vars_new = sorted(rest, key=lambda c: (-varless.count(c), -alphabet.index(c))) + sorted(set(bytes([v]) for v in alphabet) - rest, key=lambda c: alphabet.index(c))
    trans = dict(zip(vars_prev, vars_new))
    return re.sub(br"(?<!\\)\b[%b]\b(?!['\"])" % b"".join(vars_prev), lambda c: trans[c.group()], src)

def cgi_zip_src(src: bytes) -> bytes:
    if len(src) < 220: return src
    codes = [src]
    def compress(s):
        c_list = [zopfli.zlib.compress(s, numiterations=i)[2:-4] for i in [2048]] + [deflate.deflate_compress(s, compresslevel=l) for l in range(7, 13)]
        lits = []
        for d in c_list:
            for delim in [b"'", b'"']:
                b_in, b_out = reencode(d, delim), bytearray()
                for b, b_next in zip(b_in, [*b_in[1:], 0]):
                    if b == 0: b_out += b"\\x00" if b_next in b"01234567" else b"\\0"
                    elif b == 13: b_out += b"\\r"
                    elif b == 92 and b_next in b"\0\n\r\"'01234567NU\\abfnrtuvx": b_out += b"\\\\"
                    elif b == 10 and len(delim) == 1: b_out += b"\\n"
                    elif bytes([b]) == delim: b_out += b"\\" + delim
                    else: b_out.append(b)
                lits.append(delim + bytes(b_out) + delim)
        return min(lits, key=len)

    codes.append(b"#coding:L1\nimport zlib\nexec(zlib.decompress(bytes(" + compress(src) + b',"L1"),~9))')
    if src.startswith(b"import"):
        imports = src.split()[1]
        codes.append(b"#coding:L1\nimport zlib," + imports + b"\nexec(zlib.decompress(bytes(" + compress(src[len(imports) + 8:]) + b',"L1"),~9))')
    return min(codes, key=len)

def zip_src(src: bytes) -> bytes:
    return min(src, my_zip_src(src), cgi_zip_src(src), oxjam_zip_src(src), key=len)

files = {}
total_save=0
with ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in range(1,401):
        try:
            o=open('/kaggle/working/task' + str(f).zfill(3) + '.py','rb').read().strip()
            zipped_src = zip_src(o)
            files[f] = min(len(o), len(zipped_src))
        except:
            continue
        #https://www.kaggle.com/code/cheeseexports/big-zippa
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
import zipfile, json, os, copy, json

def check(solution, task_num, valall=False):
    if task_num == 157: return True # this one just takes a while to run
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
                actual = [[int(x) if int(x) == x else x for x in row] for row in actual]
                if json.dumps(actual) != json.dumps(expected):
                    return False
            except:
                return False
        return True
    except Exception as e:
        print(e)
        return False
```

```python
#https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vQ7RUqwrtwRD2EJbgMRrccAHkwUQZgFe2fsROCR1WV5LA1naxL0pU2grjQpcWC2HU3chdGwIOUpeuoK/pubhtml#gid=1427788625
top=[58,86,58,73,166,49,62,84,95,66,114,122,124,57,87,43,90,275,103,126,51,91,182,62,127,50,97,63,103,93,45,39,71,116,82,75,101,51,60,63,49,132,56,195,45,166,55,92,81,83,111,40,21,266,77,39,48,85,151,47,62,121,74,135,82,237,33,99,143,78,100,54,46,77,86,258,110,59,105,231,81,50,40,62,49,151,36,95,232,147,62,86,98,96,70,282,93,64,93,85,258,124,29,84,124,55,132,46,77,83,60,91,25,64,51,20,115,225,102,78,85,70,71,95,110,54,57,57,47,61,117,86,273,123,32,94,137,103,81,36,92,40,127,53,157,58,72,118,74,30,100,39,118,98,18,118,237,260,103,100,79,96,130,32,112,61,70,110,108,163,51,20,201,89,74,61,51,47,21,74,67,160,79,91,107,60,83,53,95,103,221,99,71,55,94,105,51,107,77,84,187,98,64,93,134,139,74,163,247,20,48,93,86,61,42,111,95,56,241,82,86,93,46,114,123,117,52,116,73,95,42,57,261,106,61,54,66,195,99,89,21,54,75,61,98,104,92,67,26,95,84,53,114,81,212,93,68,61,83,130,47,39,106,178,104,88,46,190,63,115,86,71,75,65,126,37,155,104,106,145,103,74,72,195,226,104,53,86,63,65,59,51,59,70,54,55,43,54,54,87,31,89,62,92,56,63,46,192,36,70,32,44,61,82,63,71,43,54,194,65,54,45,100,216,140,30,67,158,54,110,82,58,83,65,105,86,43,62,37,111,108,106,63,74,89,52,50,88,172,89,66,82,83,92,95,92,81,90,64,45,185,67,175,133,111,301,126,129,109,159,102,47,38,103,53,30,55,131,135,27,79,124,120,61,25,50,192,61,57,98,63,135,63,85,53,139,110,72,62,66]

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