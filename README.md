# Kaggle RSNA2024 Competition
In this repo, I would like to share the work I did for the RSNA 2024 Lumbar Spine Degenerative Classification Kaggle competition (https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification). 
The challenge focused on the classification of five lumbar spine degenerative conditions: Left Neural Foraminal Narrowing, Right Neural Foraminal Narrowing, Left Subarticular Stenosis, Right Subarticular Stenosis, and Spinal Canal Stenosis. For each imaging study in the dataset, a severity scores (Normal/Mild, Moderate, or Severe) was provided for each of the five conditions across the lumbar intervertebral disc levels L1/L2, L2/L3, L3/L4, L4/L5, and L5/S1.

The dataset consited of patient studies, which included from two to five scans. Each scan could have been of three types: Sagittal T2, Sagittal T1, Axial T2. Moreover, each image in the scan posessed metadata in accodance to the dicom file structure convention, which included image orientation and slice thickness. Lastly, about 70% of the scans had images that were labelled with coordinates by the doctors.

I dealt with the problem with two approaches: 
1) Multitask model: the general idea was to make use of the labelled images and coordinates, to avoid providing unnecessary noise to the model. Therefore, in Preprocessing_for_multitask_model.ipynb, I proceded to:
   - create a model to detect the image in the scan where the condition is most visible (using the 70% of training data that had that information) using an ensemble model of RandomForest, SVM, MLP and XGBoost on features of the scan extracted using the skimage library.
   - select three images around the detected one
   - crop them around the area of interest to reduce the size of the image without losing important information (the regions were determined statistically for Axial T2 and using characteristic of the scans for Sagittal).
   - Update the known coordinates after cropping
   - Resizing and saving the images.
   In the Multitask_model notebook, I proceeded to:
   - Define a pytorch dataset that would: load the images
   - Store them in a nested tensor (as axial images had a different shape compared to the Sagittal ones)
   - Provide a mask for the missing coordinates as well as the missing images for computing the loss
   - Create a model made of a backbone model from timm used as feature extractor, a keypoint regressor to predict the coordinates, and a classifier.
   In the training loop, the images would be passed through the model once to infer the coordinates, then a helper function would crop the image around the coordinates, the images would get super resolved by a superresolution pretrained model, then fed back to my model which would then use different classifiers for each condition and location based on which image was fed and where it was cropped.

2) Sequence model: here instead, I passed the whole scan (selected 10 evenly spaced images per scan type, ordered them and cropped them in the Preprocess_for_sequence_model) to a model that had a pytorch pretrained timm feature extractor that would extract features from each image in the scan in order and pass it to a transformer encoder to process the relationship between the images, that then would pass its information to a classifier for all the conditions and levels.
