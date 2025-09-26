# PLM-Effector:  A Hybrid Deep Learning Framework for Accurate Prediction of Bacterial Secreted Proteins
Protein secretion is a key process in bacteria, enabling proteins to reach extracellular spaces, other microbes, or host cells. Bacteria use diverse secretion systems, each with distinct structures, substrates, and biological roles. Effector proteins delivered by these systems manipulate host processes, from immune evasion to cytoskeletal disruption, highlighting the importance of accurate prediction for understanding bacterial pathogenicity. We developed PLM-Effector, a hybrid framework that combines pre-trained protein language models with deep learning architectures to achieve robust, type-specific prediction of secreted proteins, including T1SE, T2SE, T3SE, T4SE, and T6SE. It systematically benchmarks multiple embeddings, evaluates both N-terminal and C-terminal regions, and identifies the most informative features for each secretion system. These features are integrated through a two-layer ensemble stacking strategy, capturing complex patterns that single-feature or single-system models often miss. By leveraging discriminative sequence representations and optimized neural models, PLM-Effector outperforms existing effector predictors across these secretion types, providing a generalizable, high-performing framework for prediction of bacterial secreted proteins.

# If you want to train your own models to predict secreted proteins, you can download the datasets:
T1SE_train.fasta  T2SE_train.fasta  T3SE_train.fasta  T4SE_train.fasta  T6SE_train.fasta  
T1SE_test.fasta   T2SE_test.fasta   T3SE_test.fasta   T4SE_test.fasta   T6SE_test.fasta  

# To predict online, please visit our web server: 
http://www.mgc.ac.cn/zhengdd/PLM-Effector/.

# To run PLM-Effector on local GPU servers, install the provided Conda environment and execute it without restrictions on the number of GPUs, supporting even whole-genome scale analyses.
conda env create -f py39_cuda11.3.yml  
python PLM-Effector_predict.py   --usefile_id T1SE_example --effector_type T1SE --cuda 0  
python PLM-Effector_predict.py   --usefile_id T2SE_example --effector_type T2SE --cuda 0  
python PLM-Effector_predict.py   --usefile_id T3SE_example --effector_type T3SE --cuda 0  
python PLM-Effector_predict.py   --usefile_id T4SE_example --effector_type T4SE --cuda 0  
python PLM-Effector_predict.py   --usefile_id T6SE_example --effector_type T6SE --cuda 0  

# If you use any related datasets or sourcecode in your work, please cite the following publication:
Dandan Zheng
