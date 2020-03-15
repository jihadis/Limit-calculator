from math import *
from functools import reduce
thresholds={
    "a":1e-2,
    "nums":50,
    "mode":2,
    "point":50,
}
consts={"inf":-thresholds["a"],"-inf":thresholds["a"]}
def Lagrange(p, x=None):
    lp = list(range(len(p)))
    if not x: x = lp.copy()
    return lambda v: sum(
        [p[j] * reduce(lambda x, y: x * y, [(v - x[i]) / (x[j] - x[i]) for i in lp if i != j]) for j in lp])
conditions=[
    lambda x:round(x,5) ,
    lambda x:x if abs(float(x))>2e-5 else 0,
    lambda x:x if x<1e+10 else "+∞",
    lambda x:x if type(x)==str or x>-1e+10  else "-∞",
]
p_conditions=[
    lambda v,l:v if l!="inf" else -0.5/v,
    lambda v,l:v if l!="-inf" else 0.5/v,
]
func=lambda c,f,a=():(lambda l:([l.append(i(l[-1],*a)) for i in c],l[-1])[1])([f])
def require_Limit(algorithm,limits,mode=1,params=None):
    values=[[float(consts[i] if i in consts else i)+params["a"],params["a"]] for i in limits.values()]
    grads = []
    def append(j):
        values[j][1] /= 2
        values[j][0] -= func(p_conditions, values[j][1], [list(limits.values())[j]])
        grads.append(eval(algorithm.format(*[v[0] for v in values])))
    if mode==1:#同时逼近
        try:
            [[append(j) for j in range(len(values))]for i in range(params["nums"])]
        except:pass
        res=func(conditions,[i for i in grads if not isinstance(i,complex)][-1])
    elif mode==2:#依次逼近
        res =[]
        for j in range(len(values)):
            for i in range(int(params["nums"]/len(values))):
                try:append(j)
                except:break
            res.append(func(conditions, [i for i in grads if not isinstance(i, complex)][-1]))
            #print(grads)
            grads=[]
    return res
    # if i<placeHolder:
    #     values[i]+=thresholds["d"]
    #     grads.append(eval(algorithm.format(*values)))
    #     require_Limit(algorithm, limits, placeHolder,values)
    # else:
    #     pass


def input_Format(algorithm,limits,params):
    placeHolder=list(limits.keys())
    algorithm="".join(["{"+str(placeHolder.index(i))+"}"if i in limits else i for i in algorithm])
    return require_Limit(algorithm,limits,params["mode"],params)
def Limit(limits,algorithm,params={}):
    params=dict(thresholds.copy(),**params)
    algorithm=algorithm.replace("^","**")
    print("Limit ("+limits+")",algorithm.replace("**","^").replace("*","")+":")
    print(input_Format(algorithm,{i.split("->")[0]:i.split("->")[1] for i in limits.split(";")},params))
#input_Format("sin(x-1)/(y**2-1)",{"x":1})
Limit("x->0","sin(x)/x")
Limit("x->inf;y->0","(1+1/(x*y))^(x^2/(x+y))",{"a":1})
Limit("x->0;y->0","x^2*abs(y)**0.5/(x^4+y^2)")


