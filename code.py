import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torchvision.transforms.functional as TF
import warnings
from PIL import ImageDraw


warnings.filterwarnings("ignore", category=UserWarning)

# Define a custom cropping function
def custom_crop(image):
    return TF.crop(image, 0, 0, min(image.size[0], 100), min(image.size[1], 100)) 
  # Path to the dataset
dataset_path = "C:/Users/User/Desktop/Fruits Classification"

  # List to store image paths
image_paths = []

  # Loop through all files in the dataset directory
for root, dirs, files in os.walk(dataset_path):
    for file in files:
          # Check if the file is an image
        if file.endswith((".jpg", ".png", ".jpeg")):
              # Construct the full path to the image file
            img_path = os.path.join(root, file)
              # Append the image path to the list
            image_paths.append(img_path)
        else:
            print(f"Ignoring non-image file: {os.path.join(root, file)}")

  # Print the number of images loaded
print(f"Number of images loaded: {len(image_paths)}")


# Define transformations for training data
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.Lambda(lambda x: custom_crop(x)),
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define transformations for validation/test data
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets
train_dataset = ImageFolder(root='C:/Users/User/Desktop/Fruits Classification/train', transform=train_transforms)
val_dataset = ImageFolder(root='C:/Users/User/Desktop/Fruits Classification/valid', transform=val_transforms)
test_dataset = ImageFolder(root='C:/Users/User/Desktop/Fruits Classification/test', transform=val_transforms)

# Batch Size
batch_size = 16

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Use the model ResNet-50
model = models.resnet50(pretrained=True)

# Modify the last layer for your specific classification task
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Reduce the Number of Epochs
num_epochs = 5

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
  # Testing
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
          outputs = model(inputs)
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on test set: {100 * accuracy:.2f}%")


  # Validation
model.eval()  # Set the model to evaluation mode
correct_val = 0
total_val = 0
with torch.no_grad():
    for inputs, labels in val_loader:
          outputs = model(inputs)
          _, predicted = torch.max(outputs, 1)
          total_val += labels.size(0)
          correct_val += (predicted == labels).sum().item()

accuracy_val = correct_val / total_val
print(f"Accuracy on validation set: {100 * accuracy_val:.2f}%")

# Load the Faster R-CNN model pre-trained on our dataset
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def classify_fruit(image_path):
      # Preprocess the input image
    preprocess = transforms.Compose([
          transforms.ToTensor(),
      ])
    input_image = Image.open(image_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

      # Pass the preprocessed image through the model to obtain predictions
    with torch.no_grad():
        output = model(input_batch)

      # Get predicted boxes and scores
    boxes = output[0]['boxes']
    scores = output[0]['scores']

      # Select only predictions with high confidence scores
    threshold = 0.5
    selected_indices = scores > threshold
    selected_boxes = boxes[selected_indices]

      # Get the predicted label from the image path
    predicted_label = image_path.split('/')[-2]  # Assuming the folder name is the label

      # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(input_image)
    for box in selected_boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline='black', width=2)
        draw.text((box[0], box[1]), f"{predicted_label}", fill='black')

      # Display the image with bounding boxes and labels
    input_image.show()

  # Example usage:
#image_path = "C:/Users/User/Desktop/Fruits Classification/test/Apple/Apple (233).jpeg"
#image_path = "C:/Users/User/Desktop/Fruits Classification/test/Mango/Mango (1669).jpeg"
image_path = "C:/Users/User/Desktop/Fruits Classification/test/Strawberry/Strawberry (792).jpeg"
classify_fruit(image_path)