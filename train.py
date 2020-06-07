from GarbageClassification import GarbageClassification

gc = GarbageClassification(backbone="MobileNet",gpu="1",logname="realcosinelr")
gc.set_environment()
pipeline = gc.prepare_pipeline()
gc.train(pipeline)