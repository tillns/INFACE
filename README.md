# INFACE: Large-Scale 3D Infant Face Model

Official Python implementation of the [INFACE](https://cgl.ethz.ch/publications/papers/paperSch24a.php) 
3D infant face model. 

<p align="center">
<img src="images/baby_face_variations.gif">
</p>

## Installation

Basic code functionality requires only a minimal python installation, using h5py for loading the model parameters
and numpy for the computations. Additional visualization and mesh handling capabilities can be enabled via open3d
(if you want to load and save meshes without open3d, the mesh class implements some json handling for mesh formatting), 
and for some advanced reconstruction methods of unknown regions, scipy, igl, fbpca, and scikit-sparse.

Example for complete installation:
```
conda create -n inface python=3.10
conda activate inface
pip install numpy h5py open3d=0.16 scipy igl fbpca scikit-sparse
```

## Download Models

We provide both, PCA and autoencoder model without disentanglement, and an additional version 
of the autoencoder with disentanglement of expression and age variation, as described in the paper.

Please contact [Till Schnabel](till.schnabel@inf.ethz.ch) to get access to the model parameters.
Access is only provided to academic researchers. Usage of the model is restricted solely to research purposes.

## Code Usage

### Code Structure

- mesh.py: We provide a basic mesh class that stores vertices and triangles as numpy arrays, 
can load meshes from and save them to disk, and visualize them using Open3D's visualizer.
- morphable_model.py: The MorphableModel class is an abstract class that implements a morphbable model visualizer 
(based on open3D), (partial) face reconstruction, model sampling, and parameter loading from HDF5 files. 
The PCA and AE class inherit from this class.
- AE_morphable_model.py: Specific class for autoencoder. Implements encode and decode function via numpy (not torch).
Accepts both, the normal and the disentangled version. The disentangled version further offers methods 
for adjusting the expression and age of a mesh, as well as transferring the expression from a source to a target mesh.
The file can be run as script for basic visualization --
note that the visualization of the normal autoencoder is not very interesting, since the latent space is 
neither orthogonal nor ordered; in the disentangled version, the slider components are automatically chosen to cover
parts of the identity, expression, and age space.
- PCA_morphable_model.py: Specific class for PCA model. Implements encode and decode function also via numpy.
Can be run as script for basic visualization.

Next, we provide some basic usages of the code. Refer to the docs in morphable_model.py for all details. 

### Basic Visualization

The PCA model can be visualized via
```
python PCA_morphable_model.py --path_to_hdf5_file /PATH/TO/pca_model.h5
```

The autoencoder models can be visualized via
```
python AE_morphable_model.py --path_to_hdf5_file /PATH/TO/ae_model.h5
python AE_morphable_model.py --path_to_hdf5_file /PATH/TO/ae_disent_model.h5
```

Other methods can be used with the help of our custom Mesh class:

### (Partial) Face Reconstruction

For example, a mesh file can be loaded along with the mask of unknown vertices, then reconstructed via the 
morphable model, then the reconstructed mesh can be visualized and saved. 
The argument back_match_known_vertices can be optionally set to True to have the known vertices match exactly the input.
```
mesh = Mesh.load("/PATH/TO/SOME/MESH/FILE.ply")
unknown_vertex_mask = np.load("/PATH/TO/UNKNOWN/VERTEX/MASK.npy")
model = AEMorphableModel("/PATH/TO/ae_model.h5")
reconstructed_mesh = model.reconstruct_mesh(mesh, unknown_vertex_mask=unknown_vertex_mask, back_match_known_vertices=True)
reconstructed_mesh.show()
reconstructed_mesh.export("/PATH/TO/SAVE/MESH.ply")
```
Like this, a partial face reconstruction such as the one presented in our paper can be achieved:
<p align="center">
<img src="images/partial_face_reconstruction.png">
</p>

### Expression Neutralization

Expressions can be neutralized for example like this:
```
mesh = Mesh.load("/PATH/TO/SOME/MESH/FILE.ply")
disentangled_model = AEMorphableModel("/PATH/TO/ae_disentangled_model.h5")
mesh_exp_neutralized = disentangled_model.set_expression(mesh, disentangled_model.latent_mean)
mesh_exp_neutralized.show()
mesh_exp_neutralized.export("/PATH/TO/SAVE/MESH.ply")
```
We showed an example for that in our paper:
<p align="center">
<img src="images/expression_neutralization.png">
</p>

### Mesh Sampling
Random infant face meshes can be generated with our models, e.g.:
```
model = AEMorphableModel("/PATH/TO/ae_model.h5")
rand_face_meshes = model.sample_meshes(5)
for num_mesh, mesh in enumerate(rand_face_meshes):
    mesh.export(f"/PATH/TO/SAVE/MESH{num_mesh}.ply")
```

### Expression Transfer
Combining expression neutralization and mesh sampling, expressions can be transferred to 
multiple randomly sampled meshes:
```    
source_mesh = Mesh.load("/PATH/TO/SOME/MESH/FILE.ply")
disentangled_model = AEMorphableModel("/PATH/TO/ae_disentangled_model.h5")

rand_face_meshes = disentangled_model.sample_meshes(5)
for num_mesh, mesh in enumerate(rand_face_meshes):
    mesh_expression_transferred = disentangled_model.transfer_expression(source_mesh, mesh)
    mesh_expression_transferred.export(f"/PATH/TO/SAVE/MESH{num_mesh}.ply")
```
In the folder "sampled_expressions", we provide four randomly sampled latent expression codes similar to the expressions
we used in the video (the actual codes we cannot share due to privacy concerns, because they are based on 
actual patient data, cf. section Dataset below). 
After clustering our dataset by expression codes, we found that these four expressions
already cover a large part of the expression variation over the whole dataset. The expressions can be transferred 
to randomly sampled faces and ages can be fixed to plus and minus two standard deviations like so:
```    
disentangled_model = AEMorphableModel("/PATH/TO/ae_disentangled_model.h5")
expressions = ["crying", "smiling", "surprised", "whining"]
sampled_meshes = disentangled_model.sample_meshes(5)
age_direction = -1
age_mean, age_std = disentangled_model.latent_mean[-1], disentangled_model.latent_std[-1]
for num_sample, sampled_mesh in enumerate(sampled_meshes):
    for expression in expressions:
        expression_latent = np.load(f"sample_expressions/{expression}_expression.npy")
        mesh_transferred_expression = disentangled_model.set_expression(sampled_mesh, expression_latent)
        mesh_transferred_expression_young = disentangled_model.adjust_age(
            mesh_transferred_expression, age_mean - 2*age_direction*age_std)
        mesh_transferred_expression_old = disentangled_model.adjust_age(
            mesh_transferred_expression, age_mean + 2*age_direction*age_std)
        mesh_transferred_expression_young.export(f"/PATH/TO/SAVE/rand_sample{num_sample}_young_{expression}.ply")
        mesh_transferred_expression_old.export(f"/PATH/TO/SAVE/rand_sample{num_sample}_old_{expression}.ply")
```
By smoothly interpolating between these expression codes, a video similar to the one at the top can be generated.



## Information about Meshes

Note that meshes to be encoded need to be in correct correspondence with the morphable models. 
This repository does not provide a registration algorithm. We refer to common repositories, such as
[NICP](https://github.com/menpo/menpo3d/blob/master/menpo3d/correspond/nicp.py) 
for a general registration from a template and
[FLAME](https://github.com/soubhiksanyal/FLAME_PyTorch) 
that shows how a morphable model can be fit to arbitrary input meshes.


## Dataset
Due to privacy concerns, we cannot share the dataset our models were trained on. 
Please contact [Till Schnabel](till.schnabel@inf.ethz.ch) if you're from an academic institution and
you're interested in setting up a data sharing agreement.

## Referencing INFACE
When using this code or model in a scientific publication, please cite
```
@InProceedings{10.1007/978-3-031-72384-1_21,
author="Schnabel, Till N. and Lill, Yoriko and Benitez, Benito K. and Nalabothu, Prasad and Metzler, Philipp and Mueller, 
Andreas A. and Gross, Markus and G{\"o}zc{\"u}, Baran and Solenthaler, Barbara",
title="Large-Scale 3D Infant Face Model",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="217--227",
isbn="978-3-031-72384-1"
}
```


