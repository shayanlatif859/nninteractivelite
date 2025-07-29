import numpy as np

print("checkpoint_cleansing is being used here!!")

def cleanse_checkpoint_for_release(checkpoint: str):
    a = np.load(checkpoint, weights_only=False)
    del a['optimizer_state']
    del a['init_args']['dataset_json']
    a['trainer_name']='nnInteractiveTrainer_stub'
    np.save(a, checkpoint)

