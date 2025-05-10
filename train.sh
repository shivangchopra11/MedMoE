source /nethome/schopra47/nvme/miniconda3/bin/activate vlm

export HF_HOME=/nethome/schopra47/flash/.cache/

conda activate vlm
cd /nethome/schopra47/nvme/bio/MedMoE
wandb login

python src/train.py --multirun experiment=pretraining_medmoe trainer=ddp trainer.devices=8 logger=wandb