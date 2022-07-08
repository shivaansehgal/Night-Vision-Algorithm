# Night-Vision-Algorithm
| Ground Truth             |  Original Image           | Output from our Model|
:-------------------------:|:-------------------------:|:-------------------------:
![gt1](/images/10003_00_100_gt.png)  | ![ori1](/images/10003_00_100_ori.png) | ![out1](/images/10003_00_100_out.png)


### Dataset:

[Sony Dataset](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)

Seeing in the Dark (SID) dataset consists of 5094 pairs of short-exposure and long-exposure images of shape 4240* 2832. The directory `long/` corresponds to long-exposure images which serve as ground truth for the model, and the directory `short/` corresponds to the short-exposure images captured at different exposure levels (0.03 to 0.1s).

### Run locally:

* Clone the repository:

  `git clone https://github.com/Computer-Vision-IIITH-2021/project-wandavision.git`

* Navigate to the code folder:

  `cd see_in_dark`

* Download and extract the dataset folders `/long` and `/short` in the `Sony_test/` folder.
* Create two new folders: 
  `mkdir saved_model/`
  
  `mkdir test_result_new/`
  
**Training from scratch:**

* Change the directory locations on the files as per your directory structure.
* Train the model from scratch: `python train_Sony.py`
* Test the model: `python test_Sony.py`

**Using the pre-trained model:**

* Pretrained model is saved as `checkpoint_sony_e4000.pth` under `Sony_test/saved_model`
* To test the pretrained model: `python test_Sony.py`
* Test images are stored in `test_results_Sony/` directory.

### Get quantitative results:

* Run all cells of the notebook `Quantitative_results.ipynb` changing the test directory path to where the test results are stored.
* The notebook generates PSNR and SSIM values between pairs of Ground Truth and output images.

## Example Output:

| Ground Truth             |  Original Image           | Output from our Model|
:-------------------------:|:-------------------------:|:-------------------------:
![gt2](/images/10016_00_250_gt.png)  | ![ori2](/images/10016_00_250_ori.png) | ![out2](/images/10016_00_250_out.png)
![gt3](/images/10074_00_300_gt.png)  | ![ori3](/images/10074_00_300_ori.png) | ![out3](/images/10074_00_300_out.png)
![gt4](/images/10167_00_300_gt.png)  | ![ori4](/images/10167_00_300_ori.png) | ![out4](/images/10167_00_300_out.png)

### Original paper:

https://cchen156.github.io/paper/18CVPR_SID.pdf
