# Package with the codes to analyse the Gray LFP dataset

For details on how to use it see the notebooks.

### DEPENDENCIES

- Numpy
- Scipy
- Frites
- xFrites
- Numba
- Xarray

#### Pipeline

1. save_power SIDX TT BR ALIGN MONDEY DECIM 

sbatch run.sh "power" MONKEY_NAME

2. rate_modulations SIDX MONDEY ALIGNED THR DECIM 

sbatch run.sh "ratemod" MONKEY_NAME

3. mi_power_analysis SIDX MONDEY ALIGNED THR DECIM 

sbatch run.sh "powerenc" MONKEY_NAME

4. mi_crackle_analysis SIDX MONDEY ALIGNED THR DECIM 

sbatch run.sh "crkenc" MONKEY_NAME

5. temporal_components SIDX THR MONKEY SURR TTYPE BEHAVIOR THR_TYPE DECIM 

sbatch run.sh "avalanche" MONKEY_NAME
