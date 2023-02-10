# t-Distributed Stochastic Neighbor Embedding (t-SNE algorithm)

## Table of contents
1. [t-SNE algorithm](#p1)
2. [Barnes-Hut-SNE algorithm](#p2)
3. [References](#p5)

## t-SNE algorithm <a name="p1" /></a>

<p align="justify">
t-SNE algorithm represents non linear dimensionality reduction method which maps high-dimensional data X = {x1, x2, ..., xn} in low dimensional data Y = {y1, y2, ..., yn} such that we perserve as much information as possible. This is done by making pariwise simmilarities <i>pij</i> between high dimensional points <i>X</i> as simmilar as possible to pairwise similarities <i>qij</i> between low dimensional points <i>Y</i>. <br/>
Pariwise simmilarities <i>pij</i> are defined as:<br/>
</p>

![image](https://user-images.githubusercontent.com/24530942/218084487-93d401dc-6198-4944-93a4-4d0a1794100a.png) <br/>

<p align="justify">
  Where each of Gaussian variances centered in data point <i>xi</i> we can obtain by binary search of <i>sigma_i</i> for predefined <i>perplexity</i> (neighbourhood points): <br/>
 </p>
 
  ![image](https://user-images.githubusercontent.com/24530942/218085105-00690dd0-953e-4aa8-8e47-2e78f06b843b.png)

 
## Barnes-Hut-SNE algorithm <a name="p2" /></a>

<p>
Documentation in progress
</p>

## References <a name="p5" /></a>
Implementation of t-SNE algorithm<br/>
t-SNE docs:<br/>
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/www2016_BigDataVisualization-1.pdf<br/>
https://distill.pub/2016/misread-tsne/<br/>
https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a<br/>
https://jeremy9959.net/Blog/tsne_annotated-fixed/<br/>
http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf<br/><br/>

adaptive learning_rate update:<br/>
https://web.cs.umass.edu/publication/docs/1987/UM-CS-1987-117.pdf<br/>

KD tree:<br/>
https://upcommons.upc.edu/bitstream/handle/2117/76382/hpgm_15_1.pdf<br/>
https://arxiv.org/pdf/cs/9901013.pdf (sliding split point)<br>
https://en.wikipedia.org/wiki/K-d_tree (wiki)<br><br/>
Neighbors Heap:<br>
https://www.hackerearth.com/practice/notes/heaps-and-priority-queues/<br>

Barnes-Hut aproximation in t-SNE:<br/>
https://arxiv.org/pdf/1301.3342.pdf
