# ProblematicSelfSupervisedOOD

To run the main experiment, use the command 

```python main_experiment.py --dataset $1 --training_mode $2 --random_state $3```

Use the --help command for additional information on the arguments. 

To set up the data, see the data folder readme. 

The gradcam visualization relies on SimCLR loss, which is a loss between multiple images. 
To solve this issue, we consider only the gradients used to generate the representation
for the image and ignore the gradients used for other images.  
