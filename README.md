# atis
This is the code associated with the paper:

Learning to map context dependent sentences to executable formal queries. Alane Suhr, Srinivasan Iyer, and Yoav Artzi. In NAACL, 2018. [paper](http://alanesuhr.com/atis.pdf) [bib](https://aclanthology.coli.uni-saarland.de/papers/N18-1203/n18-1203.bib)

I will be slowly adding and documenting the code. Please open an issue if you have any questions about the code or getting it to run.

If you want access to the data, code for preprocessing it, and the database to execute the queries, please email me (`suhr@cs.cornell.edu`). ATIS comprises LDC93S5, LDC94S19, and LDC95S26.

The main file is `run.py`. You can train with my parameters by calling `train.sh`, and evaluate by calling `eval.sh` (but will need to provide the save file name, and you can edit this script so that it selects segments from gold queries instead, as well as evaluating on different splits of the data).  

## Prerequisites
You need Crayon installed and running. See [this page](https://github.com/clab/dynet/tree/master/examples/tensorboard) for details. If you don't want to use Crayon, you can comment it out. You can use the provided scripts `remove_experiments.py` and `delete_all_experiments.py` in order to remove old experiments.
