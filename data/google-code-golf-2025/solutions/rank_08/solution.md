# 8th place write-up: import itertools

- **Author:** jacekwl
- **Date:** 2025-11-08T12:30:07.147Z
- **Topic ID:** 615039
- **URL:** https://www.kaggle.com/competitions/google-code-golf-2025/discussion/615039

**GitHub links found:**
- https://github.com/Natanaelel/GoogleCodeGolf2025
- https://github.com/lynn/pysearch

---

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2F49c8620b75967d6937284391f06d652f%2Fintro.webp?generation=1762596152173797&alt=media)

# Introduction

We’d like to begin by thanking the organizers @mmoffitt and all our fellow competitors we’ve shared the last three months with. It’s been a lot of work—especially for a mostly manual, mostly non-expert benchmark team like us—but we’re very happy with where we ended up on the leaderboard. We genuinely worked our behinds off to get there.

You can find our solutions, compression scripts and other interesting things at this link: https://github.com/Natanaelel/GoogleCodeGolf2025

# Approach

Like many other teams, we manually worked on each task, so we didn’t have a secret trick that suddenly cut dozens of bytes. Our final result was the sum of many small insights we picked up along the way. Since most of us came into this with little to no code-golf experience, our first instinct was to lean heavily on LLMs. It didn’t take long to realize it was not going to get us anywhere near the top in what was already a very saturated field just a few weeks in.

# Solutions / techniques overwiew

### Recursion, cell automata, pysearch - task002

In this task we have to fill all regions closed by green borders with yellow color. There are many ways to solve this task. The simplest way is straightforward DFS/Flood-fill algorithm. Another approach is to look only at 2 cells at a time and set value of current cell based on neighbor's value, almost like [Cellular automata](https://en.wikipedia.org/wiki/Cellular_automaton). You just need to do enough iterations (using recursion to save bytes) while rotating array after each step to take care of neighbor pairs in each orientation.

```python
p=lambda g,i=67:-i*g or p([[4-4%(0**i|b or~a%9)for a,b in zip([4]+r,r)]for*r,in zip(*g[::-1])],i-1)
```

There are 3 important parts here:
- recursion template `p=lambda g,i=67:-i*g or p(...,i-1)`
- how do you access the neighbor pair, we used pretty simple `for a,b in zip([4]+r,r)`
- how do you calculate new value based on condition, we flood from the side for all iterations except last one and then swap colors in the last step using `4-4%(0**i|b or~a%9` that was figured out with help of tool called [pysearch](https://github.com/lynn/pysearch) by lynn.

Those parts can be done in many different ways. And there are many more tricks to be used here that you can find by looking at top teams solutions.

Task002 animation, keep in mind that rotations are not shown so it's easier to follow what is happening (steps go from 67 to -1).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2F3f69ef7107ea346cc045219307d0a345%2Ftask002.gif?generation=1762594915845044&alt=media)

### Loop trick instead of recursion template - task095

We used recursion template shown above in many tasks, but turns out that if number of iterations is smaller than grid size and we don't need to make calculations based on the counter we can use shorter template used below.
What is also interesting here is that parameter `b` could be just lambda parameter even though technically it should be set to zero before processing each row.

```python
p=lambda g,b=0:[g:=[[b%4|(b:=a)for a in r]for r in zip(*g[::-1])]for _ in g][3]
```

### Iterating while mutating using pop() - task214

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2Fb22cf963e20068477aab8c59eab4fe78%2Fupload_a256e11de938e616f2c337573f5512da.png?generation=1762595057379835&alt=media)

In this task we have to finish pattern located in the left square.

```python
p=lambda g:[r+g.pop()[3::-1]for r,*r[4:]in zip(g*1,*g[::-1])]
```

There are a couple of interesting tricks used here:
- `g.pop()` that returns last row from `g`
- `g*1` to make a shallow copy so we can still iterate over full `g` even though `g.pop()` is mutating `g` (`zip(g*1,*g[::-1])` if resolved before any `pop()`)

### Recursion with a*0==0 trick - task021

There were tasks that could be solved in a way that we perform the same operation both for whole row and for single number in a row. So we needed a way to test if current value is a row or a number.
We can take advantage of the fact that:

`list * 0 = []`

`number * 0 = 0`

```python
p=lambda g:len(g[g.count(a:=g[0])-1:])*[a*(a*0==0)or p(a)]
```

If `a*0==0` is true then we return `a` (number)
Otherwise we recurse with `or p(a)` branch.

### Loop flattening and bitwise operations - task365.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2F44cd40e20c943a053e05718145505088%2FScreenshot%20from%202025-11-08%2012-36-32.png?generation=1762601919168586&alt=media)


The goal of this task is to find the rectangle with the most red pixels. We can achieve it by simply brute forcing all possible candidates using 4 loops over coordinates `i,j` and sizes `w,h`. For this task it's enough if each loop is from 0 to 7, that means we can alias `R=range(8)` and use 4 loops.

```python
R=range(8)
p=lambda g:max(((s:=sum(m:=[r[j:][:w]for r in g[i:][:h]],[])).count(2)*all(s),len(s),m)for i in R for j in R for w in R for h in R)[2]
```

But we can also flatten all 4 loops into single one and take advantage of the fact that 8 is power of 2 so we can calculate `i,j,w,h` using bitwise operations (instead of `//` and `%` arithmetic).

```python
p=lambda g:max((all(s:=sum(m:=[x[t&7:][:t>>6&7]for x in g[t>>3&7:][:t>>9]],g))*s.count(2),len(s),m)for t in range(4096))[2]
```

### Using hash to deal with edge cases - task070

There were tasks that could be solved simple way but failed 1 or 2 edge cases. So there were 2 options:
- use longer more reliable solution
- hash / hardcode that 1-2 edge cases and use the simpler solution

Here we use `hash((*c,))%709==8` that calculates hash of column (as tuple) to detect that edge case so we can apply additional logic so solution pass all test cases.

```python
p=lambda g:[[x+2*(x<8in r)*(hash((*c,))%709==8or 8in c)for*c,x in zip(*g,r)]for r in g]
```

### Flattening array - task286

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2Fcf0a1631f57ad8da512e479d693c4ef3%2Ftask286.gif?generation=1762595439433161&alt=media)

This is another task that uses recursion+rotation template.
If the current cell is empty and neighbor has one of 2 maze colors, then current cell will have the other color.
To find the other color we can flatten the 2D array into 1D array using `sum(g,[])`, then we can convert to set to figure out the color that we need (the only colors possible are black, cyan (wall color), neighbor color (that we know) and other color (that we are looking for)). Also interesting trick to save 1b is to set initial value of sum to e.g. `r` instead of `[]` to save a byte.

```python=
p=lambda g,k=271:-k*g or p([[b or a%8and sum({*sum(g,r)})-a-8for a,b in zip([0]+r,r)]for*r,in zip(*g[::-1])],k-1)
```

### Iterating over raw bytes - task317

The most obvious solution for task317 is quite simple, we just use list comprehension and for each position we check the closest center of 3x3 square. 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2F8d0bd0e9639dc400689d87dec370bf80%2FScreenshot%20from%202025-11-08%2012-39-28.png?generation=1762602000742974&alt=media)


```python
R=[1,1,1,4,4,4,7,7,7]
p=lambda g:[[0<g[y][x]for x in R]for y in R]
```

That R array is quite long but we can use raw bytes intead.

```python
R=b""
p=lambda g:[[0<g[y][x]for x in R]for y in R]
```

### Counting bits - task369

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2F7f0ab9372bbccde4f0d93a5e2591a216%2FScreenshot%20from%202025-11-08%2012-40-59.png?generation=1762602112385832&alt=media)

In this task we have to set color of black shapes based on the size of the shape.
Initially we solved this task using DFS/Flood fill algorithm, but it was definitely too long.
Next we tried regex but we still weren't happy with the result.

The last solution used the following idea:
- initialize each black pixel with bits set on unique position
- then do enough iterations/rotations where we make bitwise OR operation for each neighbor pair
- in the last recursion step set the value based on number of bits set

What is also interesting here:
- instead of using extra counter parameter, we reused recursion parameter `k` to save bytes, it made the program much slower but it worked
- instead of initializing with `1<<` we initialized with `3<<` (which set 2 unique bits instead of 1) because this way pysearch expression was 1 byte shorter

```python
p=lambda g,k=3:-k*g or p([[0**k*~14//~a.bit_count()or a|b*(a>5)or 3<<(k:=k+4)for a,b in zip(r,[0]+r)]for*r,in zip(*g[::-1])],k-1)
```

### Bruteforcing solution to avoid bad test cases - task346

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2Ff8deefd24f0ca2fdbf2814c5264c8a2b%2FScreenshot%20from%202025-11-08%2012-42-44.png?generation=1762602195690805&alt=media)

I
In this task we have to return 1x1 array with color that is inside a different color square in input array.
The simplest solution would be to try to return the least common color using this solution:

```python
p=lambda g:[[min(S:=sum(g,[]),key=S.count)]]
```

But there is good news and bad news.
The bad news is that it fails 3 test cases.
The good news is that it fails 3 test cases, so maybe we can make a slight code modifications to make all tests pass.
The general idea is to look at failing cases and try to figure out how we can modify parameters of sum function so it will produce expected result. We did it by trial and error with a help a tiny bruteforce script.

```python
p=lambda g:[[min(S:=sum(g[1:7]+g[1:],[]),key=S.count)]]
```

The simple script we used, we tried different templates.
```python
for i in range(10):
    for j in range(i+1,10):
        print(i,j)
        for k in range(10):
            solution=f"p=lambda g:[[min(S:=sum(g[{i}:{j}]+g[{k}:],[]),key=S.count)]]"
            if verify_program(task_num, solution, task_data):
                print(solution)
```

Unfortunately we still missed the shortest possible solution that was really close.

### Regex is quite powerful - task017

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2Fc3b9fa1f80cdd67e48ad92ca525e2f6a%2Ftask017.gif?generation=1762595807517392&alt=media)

The idea was to rotate the grid, then use a regex to look for a sequence of five non-zero values followed by a zero. If that exact 5-value pattern appears again later in the grid, we replace the zero with whatever value follows the second occurrence. We repeat this until all the gaps are filled.

This solution was a kind of catalyst for the team to realize regex makes reasoning about some types of tasks a lot easier. It gave some of us the motivation to read the regex manual more carefully.

```python=
import re
p=lambda g,k=31:-k*g or p(eval(re.sub(r"(([^0], ){5})0(?=.*?\1(.))",r"\1\3",str([*zip(*g[::-1])]))),k-1)
```

### Compression optimizations - task238

Many of our solutions ended up being long enough that compressing them was a viable way to shorten them. For this, we based our initial compression script on the ones that were shared publicly and improved them by adding more variants of `zlib` and `zopfli` compression levels, with or without blocksplitting, and setting the number of iterations to a specific value in the range 1-100. We also made heavy use of the variable optimization notebook https://www.kaggle.com/code/garrymoss/compressed-variable-name-optimization by @garrymoss, thanks a lot for sharing it.

While far from optimal (-28b) from diamond in the end. The solution for task238 is a very silly example of why compression golfing has a completely different approach than normal golfing.

We had this code that compressed from the original 281b to 251b

```python
def p(t):
 f=lambda f:[[*e]for e in zip(*filter(f,zip(*t)))if f(e)];r=f(lambda e:any({*e}-{8}))
 for f,e in enumerate(f(lambda e:8in e)):
  for p,n in enumerate(e):d=len(e);r[f+1][p+1]=n and(f<p<d+~f)*r[0][1]+(d+~f<p<f)*r[-1][1]+(p<f<d+~p)*r[1][0]+(d+~p<f<p)*r[1][-1]or n
 return r
```

But if you unpacked all the clever lambdas you would favor in the traditional golfing into this:

```python
def p(n):
 e=n
 e=[e for*e,in zip(*e)if any(n%8for n in e)]
 e=[e for*e,in zip(*e)if any(n%8for n in e)]
 n=[e for*e,in zip(*n)if 8in e]
 n=[e for*e,in zip(*n)if 8in e]
 for f,n in enumerate(n):
  for a,t in enumerate(n):r=len(n);e[f+1][a+1]=t and(f<a<r+~f)*e[0][1]+(r+~f<a<f)*e[-1][1]+(a<f<r+~a)*e[1][0]+(r+~a<f<a)*e[1][-1]or t
 return e
```

We have made the original far bigger 338b, but it now compressed to 223b. The same algorithm, just written in a DRY (Do repeat yourself) fashion yielded 28b.

# Conclusion

We dedicated a lot of time and effort to manually improve our solutions for each task. We tried using LLMs but in most cases they were unhelpful, it was much easier for us to tackle the problems manually instead. We are very happy with our result, especially considering how little code-golfing experience some of us had prior to this competition.
We would also like to thank all the people that shared helpful insights in kaggle discussions, especially @kayjoking, @hellopir2, @lukeg10081, @kenkrige and @garrymoss.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F977970%2Fc97916c7bc78f2e87d3e3441ff00ab84%2Foutro.jpg?generation=1762597022697898&alt=media)



