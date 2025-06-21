# MedMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Image">
</p>

> [Shivang Chopra](https://shivangchopra11.github.io/)<sup>1</sup>, [Gabriela Sanchez-Rodriguez](https://feola.bme.gatech.edu/people-2/gabriela-sanchez-rodriguez-2/)<sup>1,2</sup>, [Lingchao Mao](https://lingchm.github.io/)<sup>1</sup>, [Andrew J. Feola](https://bme.gatech.edu/bme/faculty/Andrew-J.-Feola)<sup>1,2,3</sup>, [Jing Li](https://www.isye.gatech.edu/users/jing-li)<sup>1</sup>, [Zsolt Kira](https://faculty.cc.gatech.edu/~zk15/)<sup>1</sup> <br>
**<sup>1</sup>Georgia Institute of Technology, <sup>2</sup>Emory University, <sup>3</sup>Joseph M Cleland Atlanta VAMC**

## Environment Setup
`conda env create --file=environment.yaml`

## Pretraining

### Single GPU
`python src/train.py experiment=pretraining_medmoe logger=wandb`

### Multi GPU
`python src/train.py --multirun experiment=pretraining_medmoe trainer=ddp trainer.devices=8 logger=wandb`

## Reference
```bibtex
@article{medmoe,
  title={MedMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding},
  author={Shivang Chopra and Gabriela Sanchez-Rodriguez and Lingchao Mao and Andrew J. Feola and Jing Li and Zsolt Kira},
  booktitle={Workshop on Multimodal Foundation Models for Biomedicine in CVPR},
  year={2025}
}
```

## Acknowledgement
Our code repository is mainly built on [UniMed-CLIP](https://github.com/mbzuai-oryx/UniMed-CLIP) and [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). We thank the authors for releasing their code.
