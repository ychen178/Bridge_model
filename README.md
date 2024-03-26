# Bridge Model for Integrating and Harmonizing Genomic Data Across Institutions

Cancer is a complex disease driven by genomic alterations, and tumor sequencing is becoming a mainstay of clinical care for cancer patients. The emergence of multi-institution sequencing data presents a powerful resource for learning real-world evidence to enhance precision oncology. However, leveraging such multi-institutional sequencing data presents significant challenges. Variations in gene panels result in loss of information when the analysis is conducted on common gene sets. Additionally, differences in sequencing techniques and patient heterogeneity across institutions add complexity. High data dimensionality, sparse gene mutation patterns, and weak signals at the individual gene level further complicate matters. Motivated by these real-world challenges, we introduce the Bridge model. It uses a quantile-matched latent variable approach to derive integrated features to preserve information beyond common genes and maximize the utilization of all available data while leveraging information sharing to enhance both learning efficiency and the model's capacity to generalize. By extracting harmonized and noise-reduced lower-dimensional latent variables, the true mutation pattern unique to each individual is captured.


### Reference
**Yuan Chen\***, Ronglai Shen, Xiwen Feng, Katherine Panageas. Unlocking the Power of Multi-institutional Data: Integrating and Harmonizing Genomic Data Across Institutions. 2024+. arXiv preprint https://arxiv.org/abs/2402.00077

### Code 
In ```bridge_functions.py```, we provide the code for the Bridge model. In ```simulation_functions.py```, we provide the code related to the simulation studies in the paper. In  ```run_simulations_setting1.py``` and ```run_simulations_setting2.py```, we provide code to run the simulation studies under different simulation settings in the paper. 
