def minEatingSpeed(piles, h):
    
    if len(piles) ==1:
        res = 0
        res +=  piles[0]//h
        if piles[0]%h != 0:
            res+=1
        return res
    biggest_pile = max(piles)
    k_index = [i for i in range(1,biggest_pile + 1)]
    
    p1 = 0
    p2 = len(k_index) - 1
    mid_good = False
    while p2 >= p1:
        print(p1,p2)
        mid_idx = int((p1+p2)/2)
        
        mid = k_index[mid_idx]
        mid_h = self.get_h(piles,mid)
        
        if mid_h <= h:
            if mid_idx == 0:
                return mid
            else:
                next = k_index[mid_idx - 1]
                next_h = self.get_h(piles,next)
                if next_h <= h:
                    p2 = mid_idx - 1
                else:
                    return mid
        else:
            p1 = mid_idx + 1
    return 10
        
def get_h(lst,k):
    res = 0
    for i in lst:
        how_many_times = i//k
        if i%k != 0:
            how_many_times+=1
        res += how_many_times
    return res
minEatingSpeed([332484035,524908576,855865114,632922376,222257295,690155293,112677673,679580077,337406589,290818316,877337160,901728858,679284947,688210097,692137887,718203285,629455728,941802184], 823855818)