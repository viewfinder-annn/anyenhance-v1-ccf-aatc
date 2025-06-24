import os
import json
from tqdm import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_v1_src', type=str, required=True, help='Path to the xxx/train_v1 directory')
    args = parser.parse_args()
    print(f"args {args}")
    
    train_v1_src = args.train_v1_src
    
    clean_dir = os.path.join(train_v1_src, "clean")
    noisy_dir = os.path.join(train_v1_src, "noisy")
    encoded_dir = os.path.join(train_v1_src, "encoded")

    voicefixer_dir = os.path.join(train_v1_src, "generated/voicefixer")
    demucs_dir = os.path.join(train_v1_src, "generated/demucs")
    frcrn_dir = os.path.join(train_v1_src, "generated/frcrn")
    nsnet2_dir = os.path.join(train_v1_src, "generated/nsnet2")
    tfgridnet_dir = os.path.join(train_v1_src, "generated/tfgridnet")

    storm_dir = os.path.join(train_v1_src, "generated/storm")
    sgmse_dir = os.path.join(train_v1_src, "generated/sgmse+")
    anyenhance_dir = os.path.join(train_v1_src, "generated/anyenhance")
    masksr_dir = os.path.join(train_v1_src, "generated/masksr")
    llase_g1_dir = os.path.join(train_v1_src, "generated/llase-g1")

    output_filename = os.path.join(train_v1_src, "train_v1.jsonl")

    other_distortion_candidate_dirs = [
        encoded_dir,
        
        voicefixer_dir,
        demucs_dir,
        frcrn_dir,
        nsnet2_dir,
        tfgridnet_dir,
        
        storm_dir,
        sgmse_dir,
        anyenhance_dir,
        masksr_dir,
        llase_g1_dir,
    ]

    clean_filenames = [f for f in os.listdir(clean_dir) if os.path.isfile(os.path.join(clean_dir, f))]

    output_records = []

    for filename in tqdm(clean_filenames):
        clean_path = os.path.join(clean_dir, filename)
        noisy_path = os.path.join(noisy_dir, filename)

        other_distortion_paths = []
        for dist_dir in other_distortion_candidate_dirs:
            full_dist_path = os.path.join(dist_dir, filename)
            if os.path.exists(full_dist_path):
                other_distortion_paths.append(full_dist_path)

        record = {
            "clean": clean_path,
            "noisy": noisy_path,
            "other_distortion": other_distortion_paths
        }
        output_records.append(record)

    with open(output_filename, 'w', encoding='utf-8') as f:
        for record in output_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')