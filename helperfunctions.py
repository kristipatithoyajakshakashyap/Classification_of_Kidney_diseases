import matplotlib.pyplot as plt 
import random
from PIL import Image 
import torch 
from typing import Tuple, Dict, List
import os 
import torchvision
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
device

def plot_transformed_image(image_paths, transform, n=3, seed=None):
    """
    Selects random images from path of images and loads/transforms 
    them plot the original vs the transformed version.
    """
    if seed:
        random.seed(seed)
    random_image_path = random.sample(image_paths, k=n)
    for image_path in random_image_path:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nSize: {f.size}")
            ax[0].axis(False)

            # Transform and plot target image 
            transformed_image = transform(f).permute(1,2,0) # note we will need to change shape of matplotlib (c, h, w) -> (h, w, c)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
            ax[1].axis('Off')

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            

def display_random_image(dataset: torch.utils.data.Dataset,
                         classes: List[str]=None,
                         n:int = 10,
                         display_shape: bool=True,
                         seed:int=None):
    # 2. Adjust display if n is too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display, purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    # 3. Set the seed
    if seed:
        random.seed(seed)
    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    # 5. Setup plot 
    plt.figure(figsize=(16, 8))
    # 6. Loop through random indexes and plot it
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        # 7. Adjust tensor dimensions for plotting 
        targ_image_adjust = targ_image.permute(1,2,0)  # [C,H,W] -> [H,W,C]
        # Plot adjusted samples
        plt.subplot(1,n,i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"Class: {classes[targ_label]}"
            if display_shape: 
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
        
        
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    """
    for dirpath, dirnames,filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")
        

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of rsults dictionary"""
    # Get the loss values of results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    # Figure out number of epochs 
    epochs = range(len(results["train_loss"]))
    # Setup a plot 
    plt.figure(figsize=(15,7))
    # Plot the loss 
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy 
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label="train_acc")
    plt.plot(epochs, test_accuracy, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();

def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform = None,
                        device=device):
    """Makes a prediction on a target image with trained model and plots the image and prediction."""
    # Load in the image 
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    # Divide the image pixel values by 255 to get them between [0,1]
    target_image = target_image / 255.
    # Transform if necessary
    if transform: 
        target_image = transform(target_image)
    # Make sure model is on the target device 
    model.to(device)
    # Turn on eval/inference mode and make prediction 
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(0)
        # Make the prediction on image with an extra dimension 
        target_image_pred = model(target_image.to(device))
    # Convert logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    # Conver prediction probabilities -> prediction labels 
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    # Plot the image alongside the prediction and prediction probability 
    plt.imshow(target_image.squeeze().permute(1,2,0)) # Remove batch and rearrange [h,w,c]
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu()}"
    plt.title(title)
    plt.axis("off")
    
    
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)