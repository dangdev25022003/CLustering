import numpy as np
import faiss
from faiss.contrib.ondisk import merge_ondisk
from rwbinarydata import fvecs_read, ivecs_read


tmpdir = '/tmp/'

# train the index
xt = fvecs_read("sift1M/sift_learn.fvecs")
index = faiss.index_factory(xt.shape[1], "IVF4096,Flat")
print("training index")
index.train(xt)
print("write " + tmpdir + "trained.index")
faiss.write_index(index, tmpdir + "trained.index")


# add 1/4 of the database to 4 independent indexes
independents = 4
for independent in range(independents):
    xb = fvecs_read("sift1M/sift_base.fvecs")
    i0, i1 = int(independent * xb.shape[0] /
                 4), int((independent + 1) * xb.shape[0] / 4)
    index = faiss.read_index(tmpdir + "trained.index")
    print("adding vectors %d:%d" % (i0, i1))
    index.add_with_ids(xb[i0:i1], np.arange(i0, i1))
    print(index.shape)
    print("write " + tmpdir + "block_%d.index" % independent)
    faiss.write_index(index, tmpdir + "block_%d.index" % independent)


print('loading trained index')
# construct the output index
index = faiss.read_index(tmpdir + "trained.index")

block_fnames = [
    tmpdir + "block_%d.index" % bno
    for bno in range(4)
]

merge_ondisk(index, block_fnames, tmpdir + "merged_index.ivfdata")

print("write " + tmpdir + "populated.index")
faiss.write_index(index, tmpdir + "populated.index")


# perform a search from disk
print("read " + tmpdir + "populated.index")
index = faiss.read_index(tmpdir + "populated.index")
index.nprobe = 16

# load query vectors and ground-truth
xq = fvecs_read("sift1M/sift_query.fvecs")
gt = ivecs_read("sift1M/sift_groundtruth.ivecs")

D, I = index.search(xq, 5)

recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(xq.shape[0])
print("recall@1: %.3f" % recall_at_1)
