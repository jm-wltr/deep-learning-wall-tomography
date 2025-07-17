tensorboard --logdir artifacts/pixel_nn

PixelMLP_32bs_1e-05lr_2025-06-25_13-47-09 is a really bad one that only predicted all black (because it was missing final batch normalization in autoencoder).
![Epoch 500 of first model](image-1.png)

PixelMLP_32bs_1e-04lr_2025-06-26_11-42-31 was much better and reached 92% accuracy for train data, and about 86% accuracy for test data. The train loss descended approximately linearly (and continued to descend) while the training loss started to increase from about the 400th epoch.
![alt text](image.png)