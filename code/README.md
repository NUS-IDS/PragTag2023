
Most code assumes the input files are in the JSON format used in the competition.

For converting JSON files to T5/FlanT5 format, use the code in 

1. convertData_FlanT5.py (args1: data.json, args2: out.csv)
2. convertData_T5.py     (args1: data.json, args2: outfile-pfx (for creating .source and .target files)

For converting the auxiliary data files to the required JSON format, use code in 
1. parseAuxData_ARR.py (args1: AuxARR-data-dir, args2: out.json)
2. parseAuxData_F1000.py (args1: AuxF1000-data-dir, args2: out.json)


For training T5/FlanT5 models use
1. finetune_FlanT5.py (args1: data-dir (containing train.csv/dev.csv), args2: out-dir (where model and tokenizer will be written))
2. finetune_T5.py     (args1: data-dir (must have train&dev.source train&dev.target), args2: out-dir (where model and tokenizer will be written))

For obtaining predictions use
1. predict_FlanT5.py (args1: model/tokenizer-dir, args2: inp-data.json, args3: out-preds.json)
2. predict_T5.py (args1: model/tokenizer path, args2: inp-data.json, args3: out-preds.json)

For RoBERTa model fine-tuning and prediction code, please use the starting_kit provided in the competition
(https://codalab.lisn.upsaclay.fr/competitions/13334#participate-get_starting_kit)

For using unlabeled data, first parse the files to JSON format, predict using the various models, then combine using
getAugDataForT5.py . This code takes in two arguments with expected file names as follows:
args1: data-dir containing data.json, data.json, flanT5_preds.json, T5_preds.json, RoBERTa_preds.json 
and args2: outfile.json

