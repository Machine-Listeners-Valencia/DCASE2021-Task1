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

In evaluation stage, the system performed **61.9%** accuracy being **51.4** the baseline.

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