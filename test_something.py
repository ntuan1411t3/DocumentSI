import pickle as pkl

with open("DocSI/imgs_docsi.pkl", "rb") as f:
    obj = pkl.load(f)

print(obj.shape)
