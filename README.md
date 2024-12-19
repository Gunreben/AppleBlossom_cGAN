# AppleBlossom_cGAN

    Initial Training Process:
        Successfully trained a Conditional GAN (cGAN) for BBCH stage classification.
        Dataset structure: /mnt/appleBlossomDataset/Apple Blossom Images/train/BBCH [stage number].
        Parameters used in training:
            Latent vector size (nz): 500 (original training value).
            Number of classes (num_classes): 10.

    Issues During Training:
        .DS_Store Issue: Non-directory files like .DS_Store caused errors in dataset processing. Resolved by filtering for valid directories and image files.
        Initial loss values indicated normal GAN behavior, with both generator and discriminator improving over epochs.

    Image Generation After Training:
        A separate script was written to generate single images from the trained model.
        Encountered a size mismatch error due to a discrepancy in num_classes between the training and generation scripts. Fixed by aligning the num_classes value with the training setting.
        Discovered that the latent vector size (nz) was set incorrectly in the generation script (default 100 vs. actual 500). Corrected to match the training.

    Continuing Training:
        Resuming training required loading the saved generator (cgan_generator.pth), discriminator (cgan_discriminator.pth), and optimizers.
        The training script initially had a mismatch in nz, causing a size mismatch in the generator’s fc layer. Resolved by setting nz to 500.

Key Mistakes:

    Latent Vector Size (nz):
        Used nz=100 in the generation and continuation scripts while training was done with nz=500.

    Number of Classes (num_classes):
        Changed num_classes to 5 in the generation script, which conflicted with the training setting of 10.

    Dataset Processing Errors:
        Non-directory files (e.g., .DS_Store) caused issues during dataset loading.

    Resuming Training:
        Initially overlooked saving and loading optimizer states, which are necessary for seamless continuation of training.

Next Steps:

    Ensure All Scripts Use Correct Parameters:
        Latent vector size (nz=500).
        Number of classes (num_classes=10).

    Save Model Configurations:
    Save hyperparameters (e.g., nz, num_classes) along with the model checkpoints to avoid mismatches in the future.

    Continue Training:
    Use the fixed training script to add more epochs, ensuring generator and discriminator states match the saved model.
