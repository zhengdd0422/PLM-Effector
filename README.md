**PLM-Effector: A Hybrid Deep Learning Framework for Accurate Prediction of Bacterial Secreted Proteins**

Protein secretion is a key process in bacteria, enabling proteins to reach extracellular spaces, other microbes, or host cells. Bacteria employ diverse secretion systems, each with distinct structures, substrates, and biological roles. Effector proteins delivered by these systems manipulate host processes, from immune evasion to cytoskeletal disruption, highlighting the importance of accurate prediction for understanding bacterial pathogenicity.
We developed **PLM-Effector**, a hybrid framework that combines pre-trained protein language models with deep learning architectures to achieve robust, type-specific prediction of secreted proteins, including **T1SEs, T2SEs, T3SEs, T4SEs, and T6SEs**. The framework systematically benchmarks multiple embeddings, evaluates both N-terminal and C-terminal regions, and identifies the most informative features for each secretion system. These features are integrated through a **two-layer ensemble stacking strategy**, capturing complex patterns that single-feature or single-system models often miss. By leveraging discriminative sequence representations and optimized neural models, PLM-Effector outperforms existing effector predictors across these secretion types, providing a generalizable, high-performing framework for bacterial secreted protein prediction.


**Dataset for Training Your Own Models**

To train your own models, you can download our datasets.zip, which contains:
T1SE_train.fasta, T1SE_test.fasta
T2SE_train.fasta, T2SE_test.fasta
T3SE_train.fasta, T3SE_test.fasta
T4SE_train.fasta, T4SE_test.fasta
T6SE_train.fasta, T6SE_test.fasta


**Web Server**

For online predictions, please visit our web server:
http://www.mgc.ac.cn/zhengdd/PLM-Effector/


**Local GPU Usage**

PLM-Effector can be run on local GPU servers. Install the provided Conda environment and execute predictions without restrictions on the number of GPUs, supporting even whole-genome scale analyses.
conda env create -f py39_cuda11.3.yml 

**Quick Demo** 

To facilitate quick testing without downloading the full training dataset, we provide **example FASTA files** for all effector types in the **tmp** folder. You can run predictions as follows:

 
**Run predictions on example sequences**
conda activate py39_cuda11.3
python run_pipeline.py --usefile_id T1SE_example --effector_type T1SE
python run_pipeline.py --usefile_id T2SE_example --effector_type T2SE
python run_pipeline.py --usefile_id T3SE_example --effector_type T3SE
python run_pipeline.py --usefile_id T4SE_example --effector_type T4SE
python run_pipeline.py --usefile_id T6SE_example --effector_type T6SE


**Note**: The trained models are large. Before running the demo, please download the pretrained models from:
http://www.mgc.ac.cn/PLM-Effector/downloads.html
and place them in the trained_models folder within the PLM-Effector repository.

