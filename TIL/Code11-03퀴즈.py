## 함수
def selectionSort(ary) :
    n = len(ary)
    for i in range(0, n-1):
        maxIndex = i
        for k in range(i+1, n) :
            if ary[maxIndex] < ary[k] :
                maxIndex = k
        ary[i], ary[maxIndex] = ary[maxIndex], ary[i]
    return ary

## 전역
import random
dataAry = [random.randint(-100, 100) for _ in range(30)]

## 메인
print('정렬 전 -->', dataAry)
dataAry = selectionSort(dataAry)
print('정렬 후 -->', dataAry)



## -100부터 100까지 숫자를 랜덤하게 30개 만들기
## 내림 차순으로 정렬하기.
