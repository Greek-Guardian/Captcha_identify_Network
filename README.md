# Captcha_identify_Network
 This is a captcha identifying network. Firstly, I generate the captcha samples using PIL. Then I build a cnn-network to identify them. This net work is surprisingly efficient and only need a small dataset and very few epochs.

Problems encountered:

1. CrossEntropyLoss: Using one-hat format targets, I found that the loss calculated was not zero. In fact, criterion(a, a) was not even zero. The truth is, that torch.nn.CrossEntropyLoss() only takes input which has not yet been logsoftmaxed, and the target format should be of index, not of on-hat.
2. Dead neurons: I encountered a case where all of the network outputs were zero tensors. In fact, due to the format of labels and the sigmoid function at the end of the network, zero tensor output was a local minimum, causing that all of the neurons became zero value. What's worse, I used ReLU as the activation function, whose gradient is zero at the value of zero. This led to zero gradient of the whole network. The whole network was dead. Huhuh. I then solved this case by deleting the sigmoid function at the end of the network and changing ReLU to PReLU which always has no-zero gradient.
