import sys
input=sys.stdin.readline
for i in range(int(input())):
    b,c=map(str,input().split())
    b=b.replace(c,' ')
    print(len(b))
