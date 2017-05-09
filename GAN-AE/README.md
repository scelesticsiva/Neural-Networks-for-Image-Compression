## How the GAN-AE code works:

Arrange parameters by selecting options from FLAGS and changing the parameters in GAN_AE class variables . Using a good set of parameters is important since the network is a little parameter-sensitive. Suggested parameters are given in the report. Test on high resolution images (for example: Lena) is given since it required more work. Cifar test images are given in the GAN_AE class variables, can be simulated easily. 

Different loss options:
- Wasserstein loss
- DCGAN loss
- Improved Wasserstein loss
- Wasserstein + L1 loss
- Wasserstein + L2 loss
- DCGAN + L1 loss
- DCGAN + L2 loss
- Improved Wasserstein + L2 loss

Adding L1 distance did not work as good as adding L2 distance to the loss function. Improved wasserstein loss did not work better than Wasserstein loss. These facts will be explored more. 

Suggestions : 

Use RmsProp for Wasserstein, Adam for DCGAN.
Choose lower learning rate for Wasserstein, a little higher for DCGAN
More than 500.000 iterations is suggested to get good reconstructed images. At each iteration, we are sampling from data to construct a batch.   
Choose alpha parameter around/more than 10 (weight of the similarity loss)



