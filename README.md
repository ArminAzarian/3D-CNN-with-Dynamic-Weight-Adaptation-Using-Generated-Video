# Generative Video Creation with 3D CNN Weight Prediction (Generated Video Input)

## Architecture Overview

(Placeholder Image: A diagram showing the VAE architecture.  Include a separate 3D CNN that takes the *generated video* as input and predicts weight updates for the VAE.)

The architecture consists of a Variational Autoencoder (VAE) acting as the generator and a Discriminator network.  A separate 3D CNN, the `WeightPredictor3DCNN`, is trained to predict weight updates for the VAE based on the *generated video*.

## Network Details

### VAE (Generator)

*   **Architecture:** Encoder (Conv2D, ReLu, MaxPool2D) -> Latent Space (Linear layers for mu/logvar) -> Decoder (ConvTranspose2D, ReLu, Sigmoid).  See `model_definitions.py` for specific layer details.
*   **Input:** Noise vector.
*   **Output:** Generated image (B, C, H, W).

### Discriminator

*   **Architecture:** (The layers are symmetric with the encoder.) See `model_definitions.py` for layer details.
*   **Input:** Real or generated image (B, C, H, W).
*   **Output:** Probability of the input being a real image.

### WeightPredictor3DCNN

*   **Architecture:** 3D CNN (Conv3D, ReLu, MaxPool3D) with a Linear layer to predict weight updates.
*   **Parameters:**
    *   Input Channels: 3 (RGB channels).
    *   Number of Filters: 32.
    *   Output Size: **Must match the number of parameters you want to adjust in the VAE**.
*   **Weight Update Prediction:** The 3D CNN processes the *generated video* and predicts a vector of weight *updates* that are applied to the VAE's layers.  The generated video is created by stacking slightly modified versions of the generated image.

## Training Data

*   Images of shape (B, C, H, W) = (16, 3, 128, 128).
*   The `ImageFolder` dataset from `torchvision` is used for loading images.

## Training Procedure

1.  The VAE generates an image.
2.  A synthetic video is created from the generated image to feed into the 3D CNN.
3.  The `WeightPredictor3DCNN` is trained to predict weight updates based on the synthetic video.
4.  The weight updates are applied to the VAE's weights *before* calculating the generator loss.
5.  The Discriminator is trained to distinguish between real and generated images using the combined Wasserstein loss with gradient penalty.
6.  The VAE's original parameters are also trained to generate realistic images.

## Usage

1.  Ensure you have PyTorch, torchvision, and tqdm installed.
2.  Organize your image data into a directory structure compatible with `ImageFolder`.
3.  Calculate the correct `WEIGHT_UPDATE_SIZE` based on the VAE parameters you want to adjust.
4.  Run the `training.py` script to train the model.
