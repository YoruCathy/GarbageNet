from GarbageClassification import GarbageClassification

gc = GarbageClassification(gpu="1")
gc.set_environment()
gc.test_speed("result/03_0.74_MobileNet.h5")