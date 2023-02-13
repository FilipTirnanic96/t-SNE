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
Time and memory complexity of t-SNE algorithm is <i>O(N^2)</i>, where N is number of data points, which is not appropriate for datasets with more than few thousand points. Barnes-Hut-SNE is approximation of t-SNE algorithms which requires <i>O(NlogN)</i> time and memory complexity and can be used for large datasets. Barnes-Hut-SNE uses 2 approximations: <br/>
	&emsp;1. Approximation of input data <b>similarity <i>pij</i></b> <br/>
	&emsp;2. Approximation of <b>t-SNE gradient</b> calculation <br/>
	
First approximation is done using <b>k-dimensional (k-d) tree</b> for finding the first <i>p=3*perplexity</i> neighbors of each <b>input data</b> point. Complexity of this approach is <i>O(NlogN)</i>. Example <b>k-d tree</b> constructed on synthetic data is presented in picture below:	
</p>

<img src="https://user-images.githubusercontent.com/24530942/218128524-47397349-59a7-41f7-b1ee-f1c1e3f361f5.png" height="250" width="300"> <br/>

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

<img src="https://user-images.githubusercontent.com/24530942/218128698-dfab8405-8696-421d-a39b-733479585167.png" height="250" width="300"> <br/>

## t-SNE and Barnes-Hut-SNE project structure <a name="p3" /></a>

<p align="justify">
Project structure can be found in picture below. Module <i>tsne_algorithm</i> is the core module and has <b>t-SNE</b> and <b>Barnes-Hut-SNE</b> implemented using <b>only numpy</b>. Implementation of <b>tree structures</b> (k-d tree and quad tree) can be found in module <i>tsne_algorithm/trees</i>. Module <i>tsne_test_script</i> has scripts for testing t-SNE implementation and it is used for <b>results visualisation</b>. Folder <i>dataset</i> contains data used for testing t-SNE implementation. Test data can be obtaind just by <b>unzipping 7z files</b>. <br/>
</p>

<img src="https://user-images.githubusercontent.com/24530942/218454261-d58ce0b8-b71e-4421-81c8-2b85a7699e2a.png" height="400" width="200"> <br/>

## Databases and results <a name="p4" /></a>

<p align="justify"> 
Two databases are use as test of tsne and Barnes-Hut-SNE implementation:<br/>
&emsp;1. <b>MNIST</b> - 6000 images of digits 0-9 with resolution 28x28. Sample images can be found below <br/>
</p>

<img src="https://user-images.githubusercontent.com/24530942/218143608-7697d1b7-45cc-4996-8c54-ee033e547964.png" height="250" width="300"> <br/>

<p align="justify"> 
&emsp;2. <b>Olivetti faces</b> - 400 images of 40 different people with resolution 64x64. Sample images can be found below. <br/>
</p>

<img src="https://user-images.githubusercontent.com/24530942/218144362-822d55e5-878c-46dd-8c7f-63cb5e6e271a.png" height="250" width="300"> <br/>

<p align="justify"> 
	<ins><b> Results on MNIST dataset on 4000 samples of t-SNE "exact" method </b> </ins> <br/>
</p>

<img src="https://user-images.githubusercontent.com/24530942/218434223-27492b0c-a990-4f28-a800-7f17a818de9f.png" height="350" width="750"> <br/>

<p align="justify"> 
	<ins><b> Results on MNIST dataset on 1000 samples of Barnes-Hut-SNE method </b> </ins> <br/>
</p>

<img src="https://user-images.githubusercontent.com/24530942/218435197-a8f38874-365c-43d2-af95-0b0316a4bb94.png" height="350" width="750"> <br/>


<p align="justify"> 
	<ins><b> Results on Olivetti faces dataset of t-SNE "exact" method </b> </ins> <br/>
</p>

<img src="https://user-images.githubusercontent.com/24530942/218441760-f1f72791-454b-45ac-91d9-b8b72ff4e336.png" height="300" width="850"> <br/>

<p align="justify"> 
	<ins><b> Results on Olivetti faces dataset of Barnes-Hut-SNE method </b> </ins> <br/>
</p>

<img src="https://user-images.githubusercontent.com/24530942/218442401-2e7d55cc-20d6-4079-a5eb-65cb9cc9a6de.png" height="300" width="850"> <br/>

## References <a name="p5" /></a>
**Implementation of t-SNE algorithm references**<br/>
<ins>t-SNE algorithm:</ins><br/>
https://distill.pub/2016/misread-tsne/<br/>
https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a<br/>
https://jeremy9959.net/Blog/tsne_annotated-fixed/<br/>
http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf<br/>
<ins>Adaptive learning rate update:</ins><br/>
https://web.cs.umass.edu/publication/docs/1987/UM-CS-1987-117.pdf<br/>
<ins>K-d tree:</ins><br/>
https://upcommons.upc.edu/bitstream/handle/2117/76382/hpgm_15_1.pdf<br/>
https://arxiv.org/pdf/cs/9901013.pdf<br>
https://en.wikipedia.org/wiki/K-d_tree<br/>
<ins>Neighbors Heap:</ins><br>
https://www.hackerearth.com/practice/notes/heaps-and-priority-queues/<br>
<ins>Quad tree</ins><br/>
https://en.wikipedia.org/wiki/Quadtree<br/>
<ins>Barnes-Hut aproximation in t-SNE:</ins><br/>
https://arxiv.org/pdf/1301.3342.pdf <br/>
https://jheer.github.io/barnes-hut/
