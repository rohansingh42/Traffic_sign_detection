# Traffic Sign Detection and Classification

Traffic signs are detected using MSER algorithm in OpenCV and then classified using a CNN. The sample dataset can be found [here](https://drive.google.com/drive/u/0/folders/0B8DbLKogb5ktTW5UeWd1ZUxibDA)

# Dependencies
- skimage
- tensorflow
- numpy
- matplotlib


# Running the Code:
- The directory structure is as follows:
```
- data
- code
- output  
```
- Paste the `Training`, `Testing` and  `input` data in the `data` folder.
- The `code` folder contains the source code for the implemented pipeline. Pre-built model of CNN is present so no need for training. You can run it using:

```
python proj6.py
```
- Training can be done by running _train.py_. It has the following arguments (use help for Parameter description):
  - BasePath
  - CheckPointPath
  - NumEpochs
  - DivTrain
  - MiniBatchSize
  - LoadCheckPoint
  - LogsPath
- The output frames will be generated in the `output` folder. The output demo video is [here]()
