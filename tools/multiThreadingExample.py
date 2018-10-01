from multiprocess import Pool


def f(x): return x*x

if __name__ == '__main__':
    
    p = Pool(4)
    result = p.map_async(f, range(10))
    print (result.get(timeout=1))
    