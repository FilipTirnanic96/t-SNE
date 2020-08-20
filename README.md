# t-SNE
Implementation of t-SNE algorithm<br/>
t-SNE docs:<br/>
https://towardsdatascience.com/t-sne-python-example-1ded9953f26<br/>
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/www2016_BigDataVisualization-1.pdf<br/>
https://distill.pub/2016/misread-tsne/<br/>
https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a<br/>
https://jeremy9959.net/Blog/tsne_annotated-fixed/<br/>
https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3<br/>
http://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf<br/>
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/manifold/_utils.pyx<br/>
adaptive learning_rate update:<br/>
https://web.cs.umass.edu/publication/docs/1987/UM-CS-1987-117.pdf<br/>
if gain(t)gain(t-1) > 0 -> deviative has same direction -> error fcn slope low -> higher learning_rate<br/>
if gain(t)gain(t-1) < 0 -> deviative has altering direction -> error fcn slope high -> lower learning_rate<br/>

KD tree:<br/>
https://upcommons.upc.edu/bitstream/handle/2117/76382/hpgm_15_1.pdf<br/>
https://github.com/scikit-learn/scikit-learn/blob/67cbaefcffe8b094f1996336f3bfcc565ccde514/sklearn/neighbors/binary_tree.pxi<br/>
https://github.com/scipy/scipy/blob/v1.5.2/scipy/spatial/kdtree.py#L182-L945<br/>
https://arxiv.org/pdf/cs/9901013.pdf (sliding split point)<br>
https://en.wikipedia.org/wiki/K-d_tree (wiki)<br>
Neighbors Heap:<br>
https://www.hackerearth.com/practice/notes/heaps-and-priority-queues/<br>

Barnes-Hut aproximation in t-SNE:<br/>
https://arxiv.org/pdf/1301.3342.pdf
