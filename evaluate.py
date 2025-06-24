import os
import argparse

def evaluate(enhanced_folder, gt_folder=None, output_folder=None, dnsmos=False, intrusive=False, wer=False, device='cuda'):
    if output_folder is None:
        output_folder = os.path.dirname(enhanced_folder)
    os.makedirs(output_folder, exist_ok=True)

    final_output_folder = enhanced_folder

    if dnsmos:
        from evaluation.dnsmos import calculate_dnsmos_score
        calculate_dnsmos_score(final_output_folder, './evaluation/DNSMOS', csv_path=os.path.join(output_folder, 'dnsmos.csv'))
    
    if intrusive and gt_folder:
        from evaluation.intrusive_se_metrics import calculate_intrusive_score
        calculate_intrusive_score(gt_folder, final_output_folder, csv_path=os.path.join(output_folder, 'intrusive.csv'))

    if wer and gt_folder:
        from evaluation.wer import calculate_wer_score
        calculate_wer_score(gt_folder, final_output_folder, csv_path=os.path.join(output_folder, 'wer.csv'), device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate enhanced audio outputs against ground truth.')
    parser.add_argument('--enhanced_folder', type=str, required=True, help='Folder containing enhanced audio files.')
    parser.add_argument('--output_folder', type=str, default=None, help='Folder for storing results.')
    parser.add_argument('--gt_folder', type=str, default=None, help='Folder containing ground truth audio files.')
    
    parser.add_argument('--dnsmos', action='store_true', help='Compute DNSMOS scores for the output audio.')
    parser.add_argument('--intrusive', action='store_true', help='Compute intrusive SE metrics (STOI, PESQ, Si-SDR).')
    parser.add_argument('--wer', action='store_true', help='Compute Word Error Rate (WER) scores.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation (default: cuda).')

    args = parser.parse_args()
    
    print(args)
    
    evaluate(args.enhanced_folder, args.gt_folder, args.output_folder, args.dnsmos, args.intrusive, args.wer, args.device)