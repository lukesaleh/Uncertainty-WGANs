# Uncertainty-Evolved-Multi-Gen-GANs
Project for Applied Machine Learning 2, titled Uncertainty-Aware Generator Framework for WGANs. 

Utilize anaconda to install the provided environment.yml file to run the notebooks.

Run the notebooks using the anaconda environment, simply press "run" while in the overall project directory.

The dataset currently explored is the chest xray dataset, which is imported via hugging face.

The model can be trained for xray images using the uncertainty_gan_xray.py file. Then, it can be finetuned using the finetune_gan_xray file. The models may be tested by changing the config setup in uncertainty_gan_xray to test mode and changing the model name to be the finetuned model. 

Some known issues with the code are a lack of parameterization and a need to separate the test scripts from the training scripts. Additionally, there are soem model architectures that have yet to be tested that will be tested in the future. 

Author: Luke Saleh (lukesaleh@ufl.edu)
