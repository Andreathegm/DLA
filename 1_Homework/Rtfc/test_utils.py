from src.utils import *

def test_estimate_model_vram():
    print("Testing estimate_model_vram()...")
    model = get_model("vit_b_16", weights='DEFAULT')
    estimate_model_vram(model,get_device())

def test_list_model_feats():
    print("Testing list_model_feats()...")
    print(list_models_feats())
    print(list_models_feats("Giovanni"))

def run_test_utils():
    test_estimate_model_vram()
    test_list_model_feats()

run_test_utils()

