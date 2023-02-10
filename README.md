# t-Distributed Stochastic Neighbor Embedding (t-SNE algorithm)

## Table of contents
1. [t-SNE algorithm](#p1)
2. [Barnes-Hut-SNE algorithm](#p2)
3. [t-SNE and Barnes-Hut-SNE project structure](#p3)
4. [Databases and results](#p4)
5. [References](#p5)

## t-SNE algorithm <a name="p1" /></a>

<p align="justify">
t-SNE algorithm represents non linear dimensionality reduction method which maps high-dimensional data X = {x1, x2, ..., xn} in low dimensional data Y = {y1, y2, ..., yn} such that we preserve as much information as possible. This is done by making pairwise similarities <b><i>pij</i></b> between high dimensional points <i>X</i> as similar as possible to pairwise similarities <b><i>qij</i></b> between low dimensional points <i>Y</i>. <br/>
Pairwise similarities <b><i>pij</i></b> are defined with Gaussian distributions:<br/>
</p>

<img src="https://user-images.githubusercontent.com/24530942/218084487-93d401dc-6198-4944-93a4-4d0a1794100a.png" height="200" width="400"><br/>

<p align="justify">
  Where each of Gaussian variances centered in data point <i>xi</i> we can obtain by binary search of <i>sigma_i</i> for predefined <i>perplexity</i> (neighborhood points): <br/>
 </p>
 
<img src="https://user-images.githubusercontent.com/24530942/218085105-00690dd0-953e-4aa8-8e47-2e78f06b843b.png" height="50" width="400"><br/>

<p align="justify">
  Pairwise similarities <b><i>qij</i></b> are defined with Student t-distribution:<br/>
 </p> 

<img src="https://user-images.githubusercontent.com/24530942/218094389-545672ba-a1ed-4c34-a10a-ce49dcf6d0f0.png" height="70" width="400"><br/>

<p align="justify">
Student t-distribution is used to overcome <b>"Crowding Problem"</b>. Similarities between same points are set to zero so <i>pij</i> and <i>qij</i> are set to 0.<br/>
For mapping to be successful we want that these high dimensional distributions <i>pij</i> are as same as possible to low dimensional distributions <i>qij</i>. Hance, <b>Kullback-Leibler divergence</b> is used as criterium function which is minimized:<br/>
</p> 
<img src="https://user-images.githubusercontent.com/24530942/218095840-53886d17-30ee-427e-8d08-b86dcca5be7d.png" height="60" width="250">

<p align="justify">
KL divergence is minimized using <b><i>gradient decent algorithm, with adaptive learning rate and momentum.</i></b> <br/><br/>
  <ins>Pseudo code od the algorithm can be found below:</ins><br/>
</p> 

<img src="https://user-images.githubusercontent.com/24530942/218096573-1811ded6-f999-4833-8a8f-7fee2afa24e6.png" height="300" width="550">
  
## Barnes-Hut-SNE algorithm <a name="p2" /></a>

<p align="justify"> 
Time and memory complexity of t-SNE algorithm is <i>O(N^2)</i>, where N is number of data points, which is not appropriate for datasets with more then few thousend points. Barnes-Hut-SNE is approximation of t-SNE algorithms which reuires  <i>O(NlogN)</i> time and memory complexity and can be used for large datasets. Barnes-Hut-SNE uses 2 approximations: <br/>
	&emsp;1. Approximation of input data <b>similarity <i>pij</i></b> <br/>
	&emsp;2. Approximation of <b>t-SNE gradient</b> calculation <br/>
	
First approximation is done using <b>k-dimensional (k-d) tree</b> for finding the first <i>p=3*perplxity</i> neighbours of each <b>input data</b> point. Complexity of this approach is <i>O(NlogN)</i>. Example <b>k-d tree</b> constructed on synthetic data is presented in picture below:	
</p>

<img src="https://user-images.githubusercontent.com/24530942/218111702-c3240ee8-97b7-43a4-affa-6b9dce6c3bea.png" height="250" width="300"> <br/>

<p align="justify"> 
t-SNE KL gradient is defined with formula:
</p>

<img src="https://user-images.githubusercontent.com/24530942/218116261-6f8e7e46-59c5-4daf-ab32-b3c7ea03576b.png" height="70" width="500"> <br/>

<p align="justify"> 
KL Gradient can be represented as follows:
</p>

<img src="https://user-images.githubusercontent.com/24530942/218118325-b3a8e6b5-40ad-4b48-8a53-dabe24a90fdc.png" height="80" width="500"> <br/>

<p align="justify"> 
	<b><i>F_attr</i></b> can be calculated in <i>O(pN)</i> time complexity. <b><i>F_rep</i></b> requires <i>O(N^2)</i> time complexity which we can reduce to <i>O(NlogN)</i> using Barnes-Hut approximation. Barnes-Hut-SNE constructs <b>quad tree</b> on output (low-dimensional) data <i>Y</i> and in each iteration of calculation of  <i>F_rep</i> it decides if current node can be used as summary of contribution to <i>F_rep</i> for all the data inside that node. Example <b>quad tree</b> constructed on synthetic data is presented in picture below:
</p>

<img src="https://user-images.githubusercontent.com/24530942/218120826-ce35b7ca-14a8-417c-a610-696b5e8cb8fe.png" height="250" width="300"> <br/>

## t-SNE and Barnes-Hut-SNE project structure <a name="p5" /></a>


## Databases and results <a name="p5" /></a>


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
