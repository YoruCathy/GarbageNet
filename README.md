`report.pdf` is the report.  
`GarbageMet.apk` can be installed on Android mobile phones.  
`./unseen/` is our `unseen` dataset.  
`./vis_unseen/` is the inference results of the images in the `unseen` dataset.  
Codes and trained models, together with train logs, are in `./codes_and_model`.  

To train and use the model, the file structure should be arranged as:  
 -----./data/  
 |----./result/  
 |----./vis/  
 |----GarbageClassification.py  
 |----train.py
 |----infer.py
 -----get_flops.py
