# obstruction-removal

The Repo consists of 4 folders - 
  1 for data creation from images in .jpg format to .npy format for the model to be able to use them as input for training.
  3 folders, each contain the code for each of the 3 layers (i) Initial layer Construction, (ii) Predictor-Flow Improvisation and (iii) Feature Corrector

The dataset for the model can be downloaded from these links - https://drive.google.com/file/d/1Pf2WxCBqBLdfAi2gdV9I-_Y479aXTVZU/view?usp=sharing   
Copy the contents of the .zip file in a folder called data.

You can find the link for the raw dataset here - ####add the dataset

If you want to create and train the model on your own dataset, the structure of your dataset should look like this -<br> 
------------- Fencing<br>
      ------------- 0/1/2/3...(folder name, training example)<br>
          ------------- mixed(contains 7 frames)<br>
              ------------- 0.jpg/1.jpg...(frame containing the image to be cleaned, 896x512)<br>
          ------------- vid1(contains 7 frames)<br>
              ------------- 0.jpg/1.jpg...(frame containing the background image(whether reflection or fencing), 896x512)<br>         
          ------------- vid2(contains 7 frames)<br>
              ------------- 0.jpg/1.jpg...(frame containing the foreground image(whether reflection or fences, depending on the problem), 896x512)<br>

The model has to be trained end-to-end and the pre-trained model cannot be provided at this time due their large size.U
