from GarbageClassification import GarbageClassification

gc = GarbageClassification()
gc.set_environment()
pipeline = gc.prepare_pipeline()
gc.train(pipeline)
