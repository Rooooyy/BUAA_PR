# Getting Started

This page provides basic tutorials about the usage of MMDetection.
For installation instructions, please see [INSTALL.md](INSTALL.md).

## Inference with pretrained models

We provide testing scripts to evaluate fair1m_Airplane 

### Test a dataset

- [x] single GPU testing
- [x] visualize detection results

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}]]]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.

Examples:

```shell
python tools/test.py configs/fair1m_configs.py \
                     model/latest.pth \ 
                     --out results/fair1m_res.pkl
```

## Inference
To inference multiple images in a folder, you can run:

```
python tools/inference.py ${CONFIG_FILE} ${CHECKPOINT} ${IMG_DIR} ${OUTPUT_DIR}
```

Examples:

```shell
python tools/inference.py configs/fair1m_configs.py  \
                          model/latest.pth           \
                          data/fair1m/test_split/images  \
                          results/images             
```

## Train a model

### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--work_dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.

Difference between `resume_from` and `load_from`:
`resume_from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
`load_from` only loads the model weights and the training epoch starts from 0. It is usually used for finetuning.

Examples:

```shell
python tools/train.py configs/fair1m_configs.py  \
                      --work_dir model          \
                      --load_from model/fair1m-a9d290f9.pth
                 [or  --resume_from model/fair1m-a9d290f9.pth]
```

## Useful tools

### Analyze logs

You can plot loss/mAP curves given a training log file. Run `pip install seaborn` first to install the dependency.


```shell
python tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

Examples:

```shell
python tools/analyze_logs.py plot_curve plot_curve     \
                            model/20210531_140701.log.json \
                            --keys s1.loss_cls s1.loss_bbox loss \
                            --legend loss_cls loss_bbox loss_total
```
