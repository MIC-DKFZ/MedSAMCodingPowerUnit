import argparse
from nnunetv2.inference.CVPR_challenge.predict_cvpr_data import main as predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str)
    parser.add_argument('-o', '--output_dir', type=str)
    args = parser.parse_args()
    args.model_path = '/opt/app/_model_for_container'
    args.checkpointname = 'checkpoint_final.pth'
    args.modality = None
    args.device = 'cpu'
    args.fold = 0
    args.num_gpus = 1
    predict(args)