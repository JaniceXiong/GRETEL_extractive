from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import pickle

file = open("embed",'rb')
embed= pickle.load(file)
file.close()
file = open("label",'rb')
label= pickle.load(file)
file.close()

#label_test = label[int(num_doc * (training_ratio + validation_ratio)):]
label = label[0]
doc=embed[0]
fea = TSNE(n_components=2).fit_transform(doc)
pdf = PdfPages('embed.pdf')
cls = np.unique(label)
label = labels
fea_num = [fea[label == i] for i in cls]
for i, f in enumerate(fea_num):
    if cls[i] in range(10):
        plt.scatter(f[:, 0], f[:, 1], label=cls[i],marker = '.')
    else:
        plt.scatter(f[:, 0], f[:, 1], label=cls[i],marker = '.')
plt.tight_layout()
pdf.savefig()
plt.show()
pdf.close()
