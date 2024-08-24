Characterizing T cell receptors-antigen binding patterns with Large Language Model
=============================================

Introduction
------------
TPepRet is a pan model for predicting TCR-peptide binding.

Prerequisites:
-------------
All environmental information can be found in "environment.yaml".

data preparation:
-------------
1. The CDR3 and peptide sequence pairs, as the first column is CDR3 sequence, the second column is peptide sequence, are written to the file 'test.txt';
2. Place the test.txt to '{CODES_PATH}/TPepRet/for_prediction/';
3. Then run the command 'python indep_test.py' and the output will be written in the '{CODES_PATH}/TPepRet/for_prediction/outputs/' folder.

Note: You can customize your file source and output location to your needs in the '# configue' block at the end of the file 'indep_test.py'. 
For a more detailed description of the model, please refer to the paper: Characterizing T cell receptors-antigen binding patterns with Large Language Model
