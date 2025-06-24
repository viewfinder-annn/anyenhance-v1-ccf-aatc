import os
import json
import time
import random
import json5
import torch
import torchaudio
import dac
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, DistributedType
from accelerate import DistributedDataParallelKwargs
from transformers import get_scheduler as get_transformers_scheduler
from anyenhance import AnyEnhance_v1, MaskGitTransformer, AudioEncoder, EncoderLoss
from dataset.jsonl_dataset import JsonlAudioDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import shutil

def pad_or_truncate(x, length=512*256):
    if x.size(-1) < length:
        # x = torch.nn.functional.pad(x, (0, length - x.size(-1)))
        repeat_times = length // x.size(-1) + 1
        x = x.repeat(1, repeat_times)
        x = x[..., :length]
    elif x.size(-1) > length:
        x = x[..., :length]
    return x

def get_dataloader(config, device):
    dataset = JsonlAudioDataset(jsonl_file_path=config['jsonl_file_path'], seq_len=config.get('seq_len', 512*256), sr=config.get('sample_rate', 44100))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    return dataloader

def get_model(config, device):
    # Load DAC models
    dac_model = dac.DAC.load(config['dac_path']).to(device)
    dac_model.to(device)
    dac_model.eval()
    dac_model.requires_grad_(False)

    # Initialize transformer and audio encoder
    transformer_config = config['MaskGitTransformer']
    audio_encoder_config = config['AudioEncoder']
    transformer = MaskGitTransformer(**transformer_config)
    audio_encoder = AudioEncoder(**audio_encoder_config)

    # Initialize AnyEnhance_v1 model
    maskgit_config = config['AnyEnhance_v1']
    model_class = AnyEnhance_v1
    model = model_class(
        vq_model=dac_model,
        transformer=transformer,
        audio_encoder=audio_encoder,
        **maskgit_config
    ).to(device)
    
    print(f"model Params: {round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2)}M")

    return model

def get_optimizer(model, config):
    if config['optimizer'] == 'adam':     
        # Retrieve specific parameters for Adam optimizer
        optimizer_config = config['adam']
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            betas=tuple(optimizer_config['betas']),
            eps=optimizer_config['eps']
        )
        return optimizer
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented")

def get_scheduler(optimizer, config):
    if 'scheduler' not in config:
        return None
    elif config['scheduler'] == 'linear':
        scheduler_config = config['linear']
        print(f"current device count: {torch.cuda.device_count()}")
        scheduler = get_transformers_scheduler(
            name='linear',
            optimizer=optimizer,
            num_warmup_steps=scheduler_config['num_warmup_steps'] * torch.cuda.device_count(),
            num_training_steps=scheduler_config['num_training_steps'] * torch.cuda.device_count()
        )
        return scheduler
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

def get_loss(config):
    if 'encoder_loss' in config:
        return EncoderLoss(config['encoder_loss'])
    else:
        return None

def train_loop(config, exp_name, model, dataloader, optimizer, scheduler, device, epochs=10, test_noisy_path=None, save_every_step=1000, eval_every_step=1000, resume_path=None, exp_path='./exp/'):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    encoder_loss_calculator = get_loss(config['train'])
    model, optimizer, dataloader, scheduler, encoder_loss_calculator = accelerator.prepare(model, optimizer, dataloader, scheduler, encoder_loss_calculator)
    
    print(f"train_loop {resume_path}")
    
    model_dst = os.path.join(exp_path, f'{exp_name}/model')
    os.makedirs(model_dst, exist_ok=True)
    audio_dst = os.path.join(exp_path, f'{exp_name}/output')
    os.makedirs(audio_dst, exist_ok=True)
    
    # input_noisy_paths = random.sample([os.path.join(test_noisy_path, f) for f in os.listdir(test_noisy_path) if f.endswith('.wav')], 10)
    input_noisy_paths = [os.path.join(test_noisy_path, f) for f in os.listdir(test_noisy_path) if f.endswith('.wav') or f.endswith('.flac') or f.endswith('.mp3')][:10]
    if accelerator.is_main_process:
        log_dst = os.path.join(exp_path, f'{exp_name}/logs')
        writer = SummaryWriter(log_dst)
    else:
        writer = None
    
    if epochs < 0:
        epochs = int(1e9)

    if resume_path is not None:
        print(f"Resuming training from {resume_path}")
        checkpoint = torch.load(os.path.join(resume_path, 'checkpoint.pth'), map_location=device)
        model_state_dict = torch.load(os.path.join(resume_path, checkpoint['model']), map_location=device)
        print(f"model_state_dict keys: {list(model_state_dict.keys())[:10]}")
        print(f"model keys: {list(model.state_dict().keys())[:10]}")
        if list(model.state_dict().keys())[0].startswith('module.') and not list(model_state_dict.keys())[0].startswith('module.'):
            model_state_dict = {f'module.{k}': v for k, v in model_state_dict.items()}
        # model.load_state_dict(torch.load(os.path.join(resume_path, checkpoint['model']), map_location=device), strict=False)
        model.load_state_dict(model_state_dict, strict=False)
        if os.path.exists(os.path.join(resume_path, checkpoint['optimizer'])):
            optimizer.load_state_dict(torch.load(os.path.join(resume_path, checkpoint['optimizer']), map_location=device))
        if checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(torch.load(os.path.join(resume_path, checkpoint['scheduler']), map_location=device))
        global_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
    else:
        global_epoch = 0
        global_step = 0
    
    saved_ckpts = []
    keep_ckpts = config["train"]["keep_ckpts"] if "keep_ckpts" in config["train"] else 2
    keep_every_step = config["train"]["keep_every_step"] if "keep_every_step" in config["train"] else 100000
    
    save_part_loss = 0
    eval_part_loss = 0

    for epoch in range(global_epoch, epochs):
        sum_loss = 0
        model.train()
        if accelerator.is_main_process:
            dataloader_iter = tqdm(dataloader)
        else:
            dataloader_iter = dataloader
        for batch in dataloader_iter:
            clean_signals, noisy_signals = batch
            clean_signals = clean_signals.to(device) # [batch_size, 1, seq_len]
            noisy_signals = noisy_signals.to(device)
            output = model(clean_audios=clean_signals, noisy_audios=noisy_signals)
            ce_loss, audio_embeds = output
            if encoder_loss_calculator is not None:
                encoder_loss_dict = encoder_loss_calculator(clean_audios=clean_signals, audio_embeds=audio_embeds)
                encoder_loss = sum([enc_loss * weight for enc_loss, weight in encoder_loss_dict.values()])
                if writer is not None:
                    for key, (enc_loss, weight) in encoder_loss_dict.items():
                        writer.add_scalar(f'Step/{key}', enc_loss.item(), global_step)
                        writer.add_scalar(f'Step/{key}_weight', weight, global_step)
                loss = ce_loss + encoder_loss
            else:
                loss = ce_loss
            if writer is not None:
                writer.add_scalar('Step/ce_loss', ce_loss.item(), global_step)
                writer.add_scalar('Step/loss', loss.item(), global_step)
                writer.add_scalar('Step/lr', optimizer.param_groups[0]['lr'], global_step)
            
            # loss.backward()
            accelerator.backward(loss)
                
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            
            sum_loss += loss.item()
            save_part_loss += loss.item()
            eval_part_loss += loss.item()
            global_step += 1

            if global_step % eval_every_step == 0 and accelerator.is_main_process:
                if input_noisy_paths is not None:
                    model.eval()
                    with torch.no_grad():
                        noisy_audios = []
                        for input_path in input_noisy_paths:
                            signal, sr = torchaudio.load(input_path)
                            signal = signal.to(device)
                            # resample to 44.1k
                            signal = torchaudio.transforms.Resample(sr, config["dataset"]["sample_rate"]).to(device)(signal)
                            signal = pad_or_truncate(signal, length=config["dataset"]["seq_len"])
                            noisy_audios.append(signal)
                        noisy_audios = torch.stack(noisy_audios)
                        noisy_audios.squeeze_(1)
                        noisy_audios = noisy_audios.to(device)
                        output_dir = f'{audio_dst}/epoch-{epoch}-step-{global_step}-loss-{round(eval_part_loss/eval_every_step, 4)}'
                        os.makedirs(output_dir, exist_ok=True)
                        # 分批处理
                        for i in range(noisy_audios.shape[0]):
                            noisy_audio = noisy_audios[i].unsqueeze(0)
                            if len(batch) == 4:
                                demo_task_type = torch.tensor([0], dtype=torch.long).to(device)
                                ids, clean_audios = accelerator.unwrap_model(model).generate(noisy_audio, task_type=demo_task_type)
                            else:
                                ids, clean_audios = accelerator.unwrap_model(model).generate(noisy_audio)
                            # use original audio name
                            torchaudio.save(f'{output_dir}/{os.path.basename(input_noisy_paths[i]).split(".")[0]}_enhanced.wav', clean_audios[0].detach().cpu(), config["dataset"]["sample_rate"])
                            torchaudio.save(f'{output_dir}/{os.path.basename(input_noisy_paths[i]).split(".")[0]}_noisy.wav', noisy_audio.detach().cpu(), config["dataset"]["sample_rate"])
                    model.train()
                eval_part_loss = 0

            accelerator.wait_for_everyone()
            if global_step % save_every_step == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} Step {global_step} Loss: {save_part_loss/save_every_step}")
                model_name = f"epoch-{epoch}-step-{global_step}-loss-{round(save_part_loss/save_every_step, 4)}"
                save_path = os.path.join(model_dst, model_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_path, 'model.pt'))
                # print(model.state_dict().keys())
                torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))
                if scheduler is not None:
                    torch.save(scheduler.state_dict(), os.path.join(save_path, 'scheduler.pt'))
                checkpoint = {
                    'model': 'model.pt',
                    'optimizer': 'optimizer.pt',
                    'scheduler': 'scheduler.pt' if scheduler is not None else None,
                    'epoch': epoch,
                    'global_step': global_step
                }
                torch.save(checkpoint, os.path.join(save_path, 'checkpoint.pth'))
                print(f"Saved checkpoint: {save_path}")
                if global_step % keep_every_step != 0:
                    saved_ckpts.append(save_path)
                else:
                    print(f"Keep checkpoint: {save_path}")
                if len(saved_ckpts) > keep_ckpts:
                    old_ckpt = saved_ckpts.pop(0)
                    try:
                        shutil.rmtree(old_ckpt)
                        print(f"Deleted old checkpoint: {old_ckpt}")
                    except Exception as e:
                        print(f"Error deleting checkpoint {old_ckpt}: {e}")
                save_part_loss = 0

        if writer is not None:
            # add epoch loss to tensorboard
            writer.add_scalar('Epoch/loss', sum_loss/len(dataloader), epoch)
        print(f"Epoch {epoch} Finished, Avg Loss: {sum_loss/len(dataloader)}")

    if writer is not None:
        writer.close()
    return model

def main(config_path, resume_path=None, exp_path='./exp/'):
    print(f"main {resume_path}")
    # Load configuration
    with open(config_path, 'r') as f:
        config = json5.load(f)
    exp_name = f"{time.strftime('%Y%m%d-%H:%M')}-{os.path.basename(config_path).replace('.json', '')}"
    exp_dst = os.path.join(exp_path, exp_name)
    if resume_path is not None:
        exp_name_temp = resume_path[:resume_path.find('/model')].split('/')[-1]
        # remove YYYYMMDD-HH:MM
        exp_name_config = exp_name_temp[15:]
        print(f"exp_name_config {exp_name_config}")
        if os.path.exists(os.path.join(exp_path, exp_name_temp)) and os.path.basename(config_path).replace('.json', '') == exp_name_config:
            exp_name = exp_name_temp
            exp_dst = os.path.join(exp_path, exp_name_temp)
    print(f"exp_dst {exp_dst}")
    os.makedirs(exp_dst, exist_ok=True)
    with open(f'{exp_dst}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    device = config['train']['device']
    dataloader = get_dataloader(config['dataset'], device)
    model = get_model(config['model'], device)
    optimizer = get_optimizer(model, config['train'])
    scheduler = get_scheduler(optimizer, config['train'])
    model = train_loop(config, exp_name, model, dataloader, optimizer, scheduler, device, config['train']['epochs'], config['dataset']['test_noisy_path'], config['train']['save_every_step'], config['train']['eval_every_step'], resume_path, exp_path)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model based on the given configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--exp_path', type=str, default='./exp/', help='Path to the exp.')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to the model to resume training from.')
    args = parser.parse_args()
    print(f"args {args}")
    main(args.config, args.resume_path, args.exp_path)