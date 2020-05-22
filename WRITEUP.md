# Project Write-Up



## Explaining Custom Layers

The process behind converting custom layers involves...
Custom  Layers are those layers which are not supported by the Model Optimizer. The process behind converting custom layers involves registering the custom layers as extensions to the Model Optimizer or replacing the unsupported subgraph with a different subgraph. The process goes as follow:

-> The Model Extension Generator tool is used to automatically create templates for all the extensions needed by the Model Optimizer to convert and the Inference Engine to execute the custom layer.
-> Then we need to edit the extractor extension template file and operation extension template file.
-> Finally generate the Model IR Files.


Some of the potential reasons for handling custom layers are, as mentioned above Custom Layers are those which are not supported by the Model Optimizer, there are many activation functions like the sigmoid, Relu, tanh, etc. which are supported , but what if the user wants to develop their own activation function or the developer developes their own network which uses some set of functions or layers which are not supported by the Model Optimizer, hence Custom Layers are required so that the developer can use their own function (not supported by the Model Optimizer) and convert it to an Intermediate Representation. 

## Comparing Model Performance

For, this project, i have used Faster_RCNN_Inception_V2_Coco model.  

I have used mean Average Precision(mAP) to calculated the accuracy of the model before and after conversion to Intermediate Representation(IR).

The accuracy before conversion was 75.19

The difference between model accuracy pre- and post-conversion was 3.47

The size of the model pre-conversion was 57.2 MB and post-conversion was 53.2 MB

The inference time of the model pre-conversion was 1264.784ms and post-conversion was 947.038 ms


## Assess Model Use Cases

Some of the potential use cases of the people counter app are...
ATM
Social Distancing


Each of these use cases would be useful because...
In ATM, the app can be used for security purpose, like if there are more than 2 people in one machine, we can alert. Also, since we are counting the duration, we can also check how much time a person spends on ATM, on an average it takes 4-5 mins to withdraw cash, but if a person stays there for long time like 10-15 mins, then we can alert the security.

With the recent corona outbreak, not more than 4  people should be there together, we can use this app to create an alert, if the number of people increases by 4 then  we can create an alert.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

Lighting has some impact on the model, because under extreme dark situations, like in night, people wont be visible hence the camera cannot detect the person(s).

Model accuracy is an important factor because we need to prevent False Positive and False Negative cases. High model accuracy makes a model robust.

Focal Length/Image size,they have a minimal impact because the model is trained for all types of images, it's a pre-trained model. Although a very bad image quality can result into bad results.


## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

