# ProblematicSelfSupervisedOOD

This is the repository for the paper [Can We Ignore Labels in Out of Distribution Detection](https://openreview.net/forum?id=falBlwUsIH)


To run the Resnet Based experiments, use the command 

```python main_experiment.py --dataset $1 --training_mode $2 --random_state $3```

Use the --help command for additional information on the arguments. 

To set up the data, see the data folder readme. 

The gradcam visualization relies on SimCLR loss, which is a loss between multiple images. 
To solve this issue, we consider only the gradients used to generate the representation
for the image and ignore the gradients used for other images.  

For CLIPN experiments, go to the CLIPN folder and use the command ```python3 zero_shot_infer_adj.py --adj_dataset Food```. 
Options for the adj_dataset are Food, Face, and Cars. 

For LMD experiments, go to the LMD folder. First copy the data folder from the repository root
, which was set up following instructions above. 

Then run ```python main.py --workdir results/cifar10_adj_0/ --config configs/adj/cifar10_configs_0_id.py --mode train```
to train the LMD model. Valid configs are located in ```configs/adj/```. The number at the end indicates the seed. 

After training the model, you can then generate reconstructions.   ```python recon.py --config configs/adj/cifar10_adj_0_id.py\
  --ckpt_path results/cifar10_adj_0/checkpoints/checkpoint_100.pth --in_domain ADJ \
  --out_of_domain ADJ --batch_size 50 --mask_type checkerboard_alt --mask_num_blocks 8 \
  --reps_per_image 10 --workdir results/cifar10_adj_0/ID_vs_OOD/```

After reconstruction is complete, you can then evaluate results with ```python detect.py --result_path \
results/cifar10_adj_0/ID_vs_OOD/checkerboard_alt_blocks8_reps10/ \
--reps 10 --metric LPIPS```