# GuidedFilter

****
	
|Author|luochuwei
|---|---
|E-mail|luochuwei@gmail.com

****
## Catalog
* [How_to_use](#How_to_use)
* [Referrence](#Referrence)
****
### How_to_use
Put the GuidedFilter.py in your source file foler or in XXX:/python/Lib/site-packages
```
import GuidedFilter as GF
GuidenceMap = cv2.imread("Lenna.png")
img = cv2.imread("Lenna.png")
radius = 3
epsilon = 0.005
myGF = GF.GuidedFilter(GuidenceMap, img, radius, epsilon)
result_img = myGF.filter()
```
****
### Referrence
1. [Guided Image Filtering](http://kaiminghe.com/publications/pami12guidedfilter.pdf), Kaiming He, Jian Sun, and Xiaoou Tang, IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2013
2. [Guided Image Filtering in CSDN](http://blog.csdn.net/wushanyun1989/article/details/18225259)
