# MedMoE

**MedMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding** <br>
Shivang Chopra*, Lingchao Mao*, Gabriela Sanchez-Rodriguez*, Andrew J. Feola, Jing Li, Zsolt Kira

## Environment Setup
`conda env create --file=environment.yaml`

## Pretraining

### Single Node
`python src/train.py experiment=pretraining_medmoe logger=wandb`

### Multi-Node
`python src/train.py --multirun experiment=pretraining_medmoe trainer=ddp trainer.devices=8 logger=wandb`

## Reference
```bibtex
@article{medmoe,
  title={MedMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding},
  author={Shivang Chopra and Lingchao Mao and Gabriela Sanchez-Rodriguez and Andrew J. Feola and Jing Li and Zsolt Kira},
  booktitle={Workshop on Multimodal Foundation Models for Biomedicine in CVPR},
  year={2025}
}
```
