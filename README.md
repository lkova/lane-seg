# lane-seg
Experimenting with lane segmentation. The first implementation is with vanilla **UNET** on the following dataset https://www.kaggle.com/thomasfermi/lane-detection-for-carla-driving-simulator.

The dataset is simple dataset from carla simulator.

# Training
Parameters that can be sent during training:\
Learning Rate, **--lr**\
Number of epochs, **--epochs**\
Batch size, **--batch_size**\
Resume training, **--resume**\
Number of workers, **--num_workers**\
Dataset path, **--dataset_path**

Example:\
```python train.py --dataset_path data/ --lr 0.001 --batch_size 16```

# Evaluation on a single image
```python eval.py --image_path /path/to/image.png```


# To-do
1. Add accuracy into training
2. Improve tensorboard output: include more information, add images
3. Add more information for saving checkpoints
4. Add evaluation for the whole folder instead of image per image
5. Extrapolate lines
6. Mark driving lane
7. Play around with augmentations, and generate bigger dataset

# Future work
1. Explore better solutions for this problem, like LaneNet
2. Explore other datasets
