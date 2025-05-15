# MedMoE: Modality-Specialized Mixture of Experts for Medical Vision-Language Understanding

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Image">
</p>

> [Shivang Chopra*](https://shivangchopra11.github.io/), [Lingchao Mao*](https://lingchm.github.io/), [Gabriela Sanchez-Rodriguez*](https://feola.bme.gatech.edu/people-2/gabriela-sanchez-rodriguez-2/), [Andrew J. Feola](https://bme.gatech.edu/bme/faculty/Andrew-J.-Feola), [Jing Li](https://www.isye.gatech.edu/users/jing-li), [Zsolt Kira](https://faculty.cc.gatech.edu/~zk15/) <br>
**Georgia Institute of Technology**

*Equally contributing first authors

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
