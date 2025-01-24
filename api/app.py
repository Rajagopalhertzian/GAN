import torch
import streamlit as st
import matplotlib.pyplot as plt
from torch import nn
import numpy as np

# Define Conditional Generator (same as the one in training)
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, image_size):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x)

# Load the pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
num_classes = 10
image_size = 28 * 28
generator = ConditionalGenerator(latent_dim, num_classes, image_size).to(device)
generator.load_state_dict(torch.load("conditional_generator.pth"))
generator.eval()

# Streamlit UI setup
st.title("Conditional GAN Image Generator")
st.write("Generate FashionMNIST images using a conditional GAN model.")

# Take input from user
latent_input = st.text_input("Enter a latent vector (comma separated, 100 values)")
label_input = st.selectbox("Select a class label", range(num_classes))

# Parse the input and generate the image
if latent_input:
    try:
        latent_vector = np.array([float(x) for x in latent_input.split(",")])
        if len(latent_vector) != latent_dim:
            st.error(f"Latent vector must have {latent_dim} values.")
        else:
            latent_vector = torch.tensor(latent_vector, dtype=torch.float32).unsqueeze(0).to(device)
            label = torch.tensor([label_input]).to(device)

            with torch.no_grad():
                generated_image = generator(latent_vector, label).cpu().squeeze(0)
            
            # Rescale and display the image
            generated_image = (generated_image + 1) / 2  # Rescale to [0, 1]
            generated_image = np.clip(generated_image.numpy(), 0, 1)

            plt.imshow(generated_image.reshape(28, 28), cmap="gray")
            plt.axis("off")
            st.pyplot()

    except ValueError:
        st.error("Invalid latent vector. Please enter numbers separated by commas.")
