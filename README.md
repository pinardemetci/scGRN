
# Identifying differentially active regulatory elements in single cells via sparse GRNs 

(Work in progress). Sparse architecture is benchmarked against common GNN modelsand interpretability methods.
Firstly, we must make the GRN. We use the ```pyscenic``` package to do this. Check out ```Infer GRN.ipynb``` on how we do this. This will make an ```adjacencies.tsv``` file, which will be used for the pytorch dataset.

Check out the dataset files for the details on parsing the tsv file and making the pytorch dataset. It is fairly simple, and will require a bit of tweaking for each dataset. But basically, each example in the dataset will have the same adjacency matrix, but the node features and the labels will be different. The node features are the number of RNA molecules that was found in the single cell sequencing. Then the label will be the cell type.

Check out the Train Model jupyter notebooks on how to train the model. The model is in ```gcnmodel.py```. We will load the model in, load the datasets, and train the model for 150-200 epochs depending on the performance of the model. 

Then, in AUC + Interpretations, we can see how to make the AUC curves for the datasets, as well as the beginnings of using GNNExplainer. 
