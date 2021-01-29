# Updates

  * Absolute paths were changed to relative paths througout througout the repo
  * External libraries scvis and integrated-gradients are provided (and updated, see below)
  * Code (both core and scvis external library) was upgraded from TF1.x to TF2.x
    * By [Using the provided upgrade script](https://www.tensorflow.org/guide/upgrade);
    * and updating Code/load_scvis.py:40 - changing parameter prob to 0.0 instead of 1.0;
    * and updating Code/load_scvis.py:5 - calling tf.compat.v1.disable_eager_execution();
    * and updating Code/train_ae.py:10 - calling tf.compat.v1.disable_eager_execution();
    * and updating Code/train_multiclass.py:13 - calling tf.compat.v1.disable_eager_execution();
    * and updating Bipolar-IG/load_aug.py:5 - calling tf.compat.v1.disable_eager_execution();
    * and updating Bipolar-IG/load_aug.py:44 - changing parameter prob to 0.0 instead of 1.0;
    * and updating Bipolar/load_vae.py:44 - changing parameter prob to 0.0 instead of 1.0;
    * and updating scvis/lib/scvis/likelihood.py - new log_likelihood_student function (depending on tensorflow-probability package);
    * and updating scvis/lib/scvis/model.py:12 - calling tf.compat.v1.disable_eager_execution();
    * and updating scvis/lib/scvis/model.py:210 - changing parameter prob to 0.0 instead of 1.0;
    * and updating scvis/lib/scvis/vae.py:138 - changing parameter prob to 0.1 instead of 0.9;
    * and updating scvis/lib/scvis/run.py:108 - specifying loader for YAML.load;
    * and by making sure that the upgrade script and parameter changings do not cancel out - in fact it did not, as we ran the update script twice (and _1 - (1 - (prob)) = prob_, see scvis/lib/scvis/vae.py:139).
  * Deprecated matplotlib code was upgraded
    * By directly importing Colorbar in Code/myplot.py:14-15
  * Bipolar-IG/run.ipynb was deleted as it was considered not relevant, not working and old
    * Consequently, Bipolar-IG/run-ig.ipynb was renamed to Bipolar/run-ig.ipynb
  * Code/train_reg.py and Code/train_class.py were irrelevant and thus deleted
  * All models were re-trained
    * By deleting all VAE-trained models (in {dataset-name}/Model);
    * and deleting all saved explanations (in {dataset-name}-K/delta([0-9]*).npy);
    * and re-training according to the procedure listed below
    
# Procedure for re-training models and re-learning explanations
  
  0. Make sure your environment aligns to the provided `requirements.txt`
  1. Install the updated variant of CSVIS by running the command `python setup.py install` in the scvis folder
  2. Choose a dataset (Bipolar/Heart/Iris/Housing)
  3. If still present: deleting all saved models and explanation by recursively deleting `{dataset-name}/Model`, and deleting `{dataset-name}-K/delta([0-9]*).npy`
  4. Go to the {dataset-name} folder and run (bash script) `train_scvis.sh`
  5. Inside {dataset-name} run vertices.py and manually select clusters
      - First check on line `16` how many classes need to be chosen
      - Run the command `python vertices.py`
      - After the plot opens, mentally decide which classes to choose
      - Place the points _around_ the class, and press any key to continue
      - After the last class has been marked, close the plot
  6. Go to the corresponding 'K-folder', open `{dataset-name}-K/run.ipynb`
  7. In cell 7, set `run` to `True`
  8. Run all cells in {dataset-name}-K/run.ipynb
  9. For the Heart/Iris/Housing dataset: recursively delete `{dataset-name}-modified/Model`, run (bash script) `{dataset-name}-modified/train_scvis.sh` and run all cells in `{dataset-name}-modified/run.ipynb`
  10. Now open `{dataset-name}/run.ipynb`, and respectively set `corrupt` and `run` to `True` in the fourth and third to last cells. Run all cells.
  11. To run the Synthetic dataset: recursively delete `Synthetic/Model`, and run `Synthetic/run.ipynb`; again delete `Synthetic/Model` and run `Synthetic/run-ig.ipynb`

# Original description

Explaining Low Dimensional Representations

A common workflow in data exploration is to learn a low-dimensional representation of the data, identify groups of points in that representation, and examine the differences between the groups to determine what they represent. 
We treat this as an interpretable machine learning problem by leveraging the model that learned the low-dimensional representation to help identify the key differences between the groups. 
To solve this problem, we introduce a new type of explanation, a Global Counterfactual Explanation (GCE), and our algorithm, Transitive Global Translations (TGT), for computing GCEs. 
TGT identifies the differences between each pair of groups using compressed sensing but constrains those pairwise differences to be consistent among all of the groups.
Empirically, we demonstrate that TGT is able to identify explanations that accurately explain the model while being relatively sparse, and that these explanations match real patterns in the data.


This repo contains an implementation of TGT as well as all of the code to reproduce the results in the [paper](https://proceedings.icml.cc/book/2020/hash/ccbd8ca962b80445df1f7f38c57759f0).  
