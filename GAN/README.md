## Lossy image compression using generative adversarial networks

It contains two structures with 2 different generator model: 
- 3 layer fully convolutional network
- Much smaller version of Resnet (given in the report)

2 different generators are written in two different functions. In order to use each of them, generator function's name in the training stage should be changed. (generator / generator_res)

Loss functions: 
- Wasserstein gan 
- Wasserstein gan + L2 loss
- Dcgan 
- Dcgan + L2 loss

Parameters and the loss function options can be selected by utilizing FLAGS or class variables. 




