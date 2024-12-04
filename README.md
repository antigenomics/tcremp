# TCRemP: T-Cell Receptor sequence embedding via Prototypes
![Splash](assets/splash.png)
TCRemP is a package developed to perform T-cell receptor (TCR) sequence embedding. TCR sequences encode antigen specificity of T-cells and their repertoire obtained using [AIRR-Seq](https://www.antibodysociety.org/the-airr-community/) family of technologies serves as a blueprint the individual's adaptive immune system.
In general, it is very challenging to define and measure similarity between TCR sequences that will properly reflect closeness in antigen recongition profiles. Defining a proper language model for TCRs is also a hard task due to their immense diversity both in terms of primary sequence organization and in terms of their protein structure.
Our pipeline follows an agnostic approach and vectorizes each TCR based on its similarity to a set of ad hoc chosen TCR "probes". Thus we follow a prototype-based approach and utilize commonly encountered TCRs either sampled from a probabilistic V(D)J rearrangement model (see Murugan et al. 2012) or a pool of real-world TCR repertoires to construct a coordinate system for TCR embedding.

The workflow is the following:
* TCRemP pipeline starts with a selection of ``k`` prototype TCR alpha and beta sequences, then it computes the distances from every of ``n`` input TCR alpha-beta pairs to ``2 * k`` prototypes for V, J and CDR3 regions, resulting in ``6 * k`` parameters (or ``3 * k`` for cases when only one of the chains is present).
> Distances are computed using local alignment with BLOSUM matrix, as implemented in our [mirpy](https://github.com/antigenomics/mirpy) package; we plan to move all computationally-intensive code there.
* Resulting distances are treated as embedding co-ordinates and and are subject to principal component analysis (PCA). One can monitor the information conveyed by each PC, whether they are related to features such as Variable or Joining genes, CDR3 region length or a certain epitope.
> N.B. TCRemP is currently in active development, please see below for the list of features, current documentation, a proof-of-concept example. All encountered bugs can be submitted to the ``issues`` section of the @antigenomics repository.

Using TCRemP one can:
- perform an embedding for a set of T-cell clonotypes, defined by TCR’s Variable (V) and Joining (J) gene IDs and complementarity determining region 3 (CDR3, amino acid sequence placed at the V-J junction). The embedding is performed by mapping those features to real vectors using similarities to a set of **prototype** TCR sequences
- embed a set of clones, pairs of TCR alpha and beta chain clonotypes
- analyze the mapping by performing dimensionality reduction and evaluating principal components (PCs)
- cluster the embeddings using [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) method with parameter selection using knee/elbow method
- visualize T-cell clone and clonotype embeddings using tSNE, coloring the visualization by user-specified clonotype labels, such as antigen specificities
- infer cluster that are significantly enriched in certain labels, e.g. TCR motifs belonging to CD8+ T-cell subset or specific to an antigen of interest

Planned features:
- [in progress] co-embed samples with  [VDJdb database](https://github.com/antigenomics/vdjdb-db) to predict TCRs associated with certain antigens, i.e. “annotate” TCR repertoires
- [in progress] perform imputation to correctly handle mixed single-/paired-chain data
- [in progress] implement B-cell receptor (BCR/antibody) prototypes to apply the method to antibody sequencing data

# Getting started

## Installation procedure and first run

One can simply install the software out-of-the-box using [pip](https://pypi.org/project/pip/) with py3.11:

```{bash}
conda create -n tcremp ipython python=3.11
conda activate tcremp
pip install git+https://github.com/antigenomics/tcremp
```

Or, in case of package version problems or other issues, clone the repository manually via git, create corresponding [conda](https://docs.conda.io/en/latest/) environment and install directly from sources:

```{bash}
git clone https://github.com/antigenomics/tcremp.git
cd tcremp
conda create -n tcremp ipython python=3.11
conda activate tcremp
pip install .
```

Check the installation by running:

```{bash}
tcremp-run -h # note that first run may be slow
cd $tcremp_repo # where $tcremp_repo is the path to cloned repository
tcremp-run -i data/example/VDJdb_data_paired_example.csv -c TRA_TRB -o data/example/ -n 10 -l antigen.epitope
```

check that there were no errors and observe the results stored in ``data/example/results/`` folder. You can then go through the ``example.ipynb`` notebook to run the analysis and visualize the results. You can proceed with your own datasets by substituting example data with your own properly formatted clonotype tables.

## Preparing the input data

The input data typically consists of a table containing clonotypes as defined above, either TCR alpha, or beta, or both. One can additionally tag clonotypes/clones with user-defined ids, e.g. cell barcodes, and labels, e.g. antigen specificity or phenotype. One can also use a custom clonotype table instead of a pre-built set of prototypes (see ``data/example/VDJdb_data_paired_example.csv``).

  
### Input format

#### Common requirements

1. V and J gene names should be provided based on [IMGT](https://www.imgt.org/) naming, e.g. ``TRAV35*03`` or ``TRBV11-2``. TCRemP will always use the major allele, so the alleles above will be transformed into ``TRBV*01`` and ``TRBV11*01``  
2. The data should not contain any missing data for any of the columns: V, J and CDR3. Although for paired-chain format one of the chains may be missing.
3. There should be no symbols except for 20 amino acids in CDR3s

#### Input columns
| Column name     | Description                                                           | Required             |
|-----------------|-----------------------------------------------------------------------|----------------------|
| a_cdr3aa        | TCR alpha CDR3 amino acid sequence                                    | required or blank(*) |
| a_v             | TCR alpha V gene ID                                                   | required or blank    |
| a_j             | TCR alpha J gene ID                                                   | required or blank    |
| b_cdr3aa        | TCR beta CDR3 amino acid sequence                                     | required or blank    |
| b_v             | TCR beta V gene ID                                                    | required or blank    |
| b_j             | TCR beta J gene ID                                                    | required or blank    |
| clonotype_index | user provided clonotype id, will be transferred to results            | optional             |
| label           | user provided label, will be used in cluster enrichment post-analysis | optional             |

(*) At least one of TCR chains should be present in each row with V, J and CDR3aa records filled. For single-chain data, the other chain may have all three records blank.


#### Single chain table example

Either wide with missing values

|   clonotype_index   |  a_cdr3aa   |   a_v   |  a_j   |    b_cdr3aa     |   b_v    |   b_j   |   label   |
|:-------------------:|:-----------:|:-------:|:------:|:---------------:|:--------:|:-------:|:---------:|
| GACTGCGCATCGTCGG-28 | CAGHTGNQFYF | TRAV35	 | TRAJ49 |                 |          |         | IVTDFSVIK |
| GACTGCGCATCGTCGG-28 |             |         |        | CASSWGGGSHYGYTF | TRBV11-2 | TRBJ1-2 | IVTDFSVIK |


#### Paired chain example

A simple wide format

|       data_id       |  a_cdr3aa   |   a_v   |  a_j   |    b_cdr3aa     |   b_v    |   b_j   |   label   |
|:-------------------:|:-----------:|:-------:|:------:|:---------------:|:--------:|:-------:|:---------:|
| GACTGCGCATCGTCGG-28 | CAGHTGNQFYF | TRAV35	 | TRAJ49 | CASSWGGGSHYGYTF | TRBV11-2 | TRBJ1-2 | IVTDFSVIK |

 
## Running TCRemP

### Basic usage

Run the tool as 

```{bash}
tcremp-run --input my_input_data.txt --output my_folder --chain TRA_TRB
```

The command above will:
- checks input data format and proofreads the dataset
- extracts TCR alpha and beta clonotypes from ``my_input_data.txt``
- calculates distance scores from clonotypes for the built-in set of ``3000`` prototypes for each chain
- performs **PCA** and saves transformed data 
- runs **DBSCAN** clustering with parameters ``min_samples = 2``, ``eps`` value inferred by knee method, and saves resulting clusters
All input will be saved in ``my_folder/``

If one runs

```{bash}
tcremp-run --input my_input_data.txt --output my_folder --chain TRA_TRB --label epitope
```

the resulting clusters will be saved with both user-provided labels and cluster labels.

The following command will skip the clustering and only save embeddings:

```{bash}
tcremp-run --input my_input_data.txt --chain TRA_TRB --clstr_model none
```
 
### Command line parameters

The parameters for running ``tcremp-run`` main script are the following:

| parameter                  | short usage   | description                                                                                                                                                                | available values                     | required | default value           |
|----------------------------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|----------|-------------------------|
| --input                    | -i            | input clonotype table                                                                                                                                                      | path to file                         | yes      | -                       |
| --chain                    | -c            | single or paired clonotype chains                                                                                                                                          | TRA, TRB, TRA_TRB                    | yes      | -                       |
| --output                   | -o            | pipeline output folder                                                                                                                                                     | path to directory                    | no       | tcremp_{inputfilename}/ |
| --cluster-algo             | -a            | clustering model to be used: dbscan, kmeans or run without clustering step (value 'none')                                                                                  | none, dbscan, kmeans                 | no       | dbscan                  |
| --labels-col               | -l            | name of the input file column with data classes for clustering. If provided, this value will be added to clustering output table and label of each cluster will be defined | str                                  | no       | -                       |
| --species                  | -s            | species of built-in prototypes to be used                                                                                                                                  | HomoSapiens, (MusMusculus - planned) | no       | HomoSapiens             |
| --n-prototypes             | -n            | number of prototypes to be selected for embedding supplemented prototype table                                                                                             | integer                              | no       | 3000                    |
| --random                   | -r            | random seed for random prototype selection. If not provided or 0, first n prototypes are selected                                                                          | integer                              | no       | 0                       |
| --prototypes_path          | -p            | path to the custom input prototype table                                                                                                                                   | path to file                         | no       | -                       |
| --prototypes-cdr3aa-column | -p_cdr3aa_col | name of cdr3aa column for custom file with prototypes. "cdr3aa" will be used if not specified                                                                              | cdr3aa column name                   | no       | cdr3aa                  |
| --prototypes-cdr3nt-column | -p_cdr3nt_col | name of cdr3nt column for custom file with prototypes. None will be used if not specified                                                                                  | cdr3nt column name                   | no       | -                       |
| --prototypes-v-column      | -p_v_col      | name of v segment column for custom file with prototypes. "v" will be used if not specified                                                                                | v segment column name                | no       | v                       |
| --prototypes-j-column      | -p_j_col      | name of j segment column for custom file with prototypes. "j" will be used if not specified                                                                                | j segment column name                | no       | j                       |
| --clonotype_index          | -d            | column containing user-provided clonotype id in input data, will be transfered to output tables                                                                            | str                                  | no       | -                       |
| --lower-len-cdr3           | -llen         | filter out cdr3 with len <llen                                                                                                                                             |                                      |          |                         |
| --higher-len-cdr3          | -hlen         | filter out cdr3 with len >hlen                                                                                                                                             |                                      |          |                         |

 
### Output

All output files will contain the following **common columns**:
- tcremp_id - assigned identifier to each row of the input table
- cloneId - identifier of unique extracted clonotype. In case of single chain pipeline, stands for TRA or TRB clonotype, in case of paired chain pipeline stands for paired TRA and TRB clonotype.

The output folder will contain the following files:

| File name                     | description                                                                                                                                                                                                                       |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| tcremb_dists_{chain}.txt      | Embeddings (the set of distances to prototypes) for each input clonotype row in the table, the chain can be: TRA, TRB or TRA_TRB for paired-chain format                                                                          |
| tcremb_pca_{chain}.txt        | Principal components of the embedding, the chain can be: TRA, TRB or TRA_TRB for paired-chain format                                                                                                                              |
| tcremb_tsne_{chain}.txt       | Calculated tSNE co-ordinates for each input clonotype row in the table, the chain can be: TRA, TRB or TRA_TRB for paired-chain format                                                                                             |
| tcremb_clstr_res_{chain}.txt  | Cluster assignments for each input clonotype row in the table. If the cluster is -1, the clonotype does not belong to any cluster (can be treated as “noise” in some settings). Chain can be: TRA, TRB or TRA_TRB for paired data |
| filtered_out_data.txt         | rows that were excluded during data cleansing with exclusion reason                                                                                                                                                               |
| clonotypes_{chain}.txt        | Clonotypes used in the single-chain pipeline, chain can be: TRA or TRB                                                                                                                                                            |
| clonotypes_paired_{chain}.txt | Clonotypes used in the paired-chain pipeline                                                                                                                                                                                      |
| prototypes_{chain}_{n}.txt    | prototypes on which the distances was calculated, where ``n`` is the number of selected prototypes, chain is either TRA or TRB. The file is absent if default subset of prototypes was used for calculation                       |
| res_{chain}.txt               | Raw distance table for single chain mapping. Chain can be: TRA or TRB                                                                                                                                                             |
| res_paired_{chain}.txt        | Raw distance table for paired chain mapping. Chain can be: TRA or TRB                                                                                                                                                             |
