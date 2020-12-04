# DCASE2021-Task1
Codes related to DCASE2021 Task 1 - Acoustic Scene Classification

Acoustic Scene Classification is a machine listening task whose goal is to map 
an audio clip into a pre-defined scene target.

The mismatch devices consideration appears when the audio clips have been recorded 
with different devices.

### DCASE2020 submission

The submission was carried out using the squeeze-excitation network presented in 
https://ieeexplore.ieee.org/abstract/document/9118879 

For more details please of the submission please we refer to the technical report:
http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Naranjo-Alcazar_34_t1.pdf

The best system obtained **65.1%** accuracy being **54.1%** the baseline in development
stage

In evaluation stage, the system performed **61.9%** accuracy being **51.4%** the baseline.

If use this papers, please cite them:

```
@article{naranjo2020acoustic,
  title={Acoustic scene classification with squeeze-excitation residual networks},
  author={Naranjo-Alcazar, Javier and Perez-Castanos, Sergi and Zuccarello, Pedro and Cobos, Maximo},
  journal={IEEE Access},
  volume={8},
  pages={112287--112296},
  year={2020},
  publisher={IEEE}
}
```

```
@techreport{naranjo2020task,
  title={TASK 1 DCASE 2020: ASC WITH MISMATCH DEVICES AND REDUCED SIZE MODEL USING RESIDUAL SQUEEZE-EXCITATION CNNS},
  author={Naranjo-Alcazar, Javier and Perez-Castanos, Sergi and Zuccarello, Pedro and Cobos, Maximo},
  year={2020},
  institution={DCASE2020 Challenge, Tech. Rep}
}
```

### First modifications

Due to submissions presented in the DCASE2020 edition some modifications to our model.
Training proceudre (callbacks, epochs and mixup remain the same)

| Modification        | Frequency bins| Accuracy    (%)       |
| :-------------: |:-------------:| :-------------:| 
| Same model, **focal loss**    | 64| 65.56 | 
| Removing Dense 100 units layer, flatten reshape, **focal loss**   | 64 | 63.57 | 
| Removing Dense 100 units layer, global average reshape, **focal loss**      | 64 |66.36      |  
| Removing Dense 100 units layer, global average reshape, no freq pooling, **focal loss**   | 64 | **67.81**      |
| Removing Dense 100 units layer, global average reshape, no freq pooling, **focal loss**   | 128 |  67.00      | 

#### Discussion

- **focal loss** improves system's performance
- no reducing the frequency bins also improves system's performance
- adding more frequency bins does not improve system's performance if doing the same procedure
- Flatten and Dense layers are more prone to overfitting in this scenario

## Run Code

- This repo assumes that it has been saved as a subdirectory of HOME

- You only have to configure the `config.py` file as desired to launch trainings