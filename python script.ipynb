{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "05cf40c0-149c-47b2-91f3-fde5925370a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "cw1_dl_hnd242f_<your_index_number>/\n",
    "│\n",
    "├── RoadNet/                     <- Dataset folder (with images, labels)\n",
    "├── paper_review_notebook.ipynb <- Your Jupyter Notebook\n",
    "├── README.md\n",
    "└── cnn_workflow.png"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4418555-a8de-47f9-b746-488be2fc1c1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "\n",
    "# Set dataset path\n",
    "data_dir = \"./Ottawa-Dataset\"  # Adjust based on your folder name\n",
    "categories = os.listdir(data_dir)\n",
    "\n",
    "print(\"Categories found:\", categories)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a5f97cd-65ee-42cf-8208-8f0d1aee41b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "img_size = 64  # Resize to 64x64\n",
    "X = []\n",
    "y = []\n",
    "\n",
    "for idx, category in enumerate(categories):\n",
    "    folder_path = os.path.join(data_dir, category)\n",
    "    for img_name in os.listdir(folder_path):\n",
    "        img_path = os.path.join(folder_path, img_name)\n",
    "        try:\n",
    "            img = cv2.imread(img_path)\n",
    "            img = cv2.resize(img, (img_size, img_size))\n",
    "            X.append(img)\n",
    "            y.append(idx)\n",
    "        except:\n",
    "            continue\n",
    "\n",
    "X = np.array(X) / 255.0  # Normalize pixel values\n",
    "y = to_categorical(y, num_classes=len(categories))\n",
    "print(\"Dataset shape:\", X.shape, y.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5786b7f7-2880-4803-8f41-2dc7433f56d2",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "print(\"Training set:\", X_train.shape, \"Testing set:\", X_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df6c2ade-517b-446a-abde-d1e88de2111c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "datagen = ImageDataGenerator(\n",
    "    rotation_range=15,\n",
    "    horizontal_flip=True,\n",
    "    zoom_range=0.2\n",
    ")\n",
    "datagen.fit(X_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3219c1b-a04f-4651-8b84-d4b24bc02152",
   "metadata": {},
   "source": [
    "CNN Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6fc23e03-8319-4d73-8114-6f08bee295a1",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout\n",
    "\n",
    "model = Sequential([\n",
    "    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),\n",
    "    MaxPooling2D(pool_size=(2, 2)),\n",
    "\n",
    "    Conv2D(64, (3, 3), activation='relu'),\n",
    "    MaxPooling2D(pool_size=(2, 2)),\n",
    "\n",
    "    Conv2D(128, (3, 3), activation='relu'),\n",
    "    MaxPooling2D(pool_size=(2, 2)),\n",
    "\n",
    "    Flatten(),\n",
    "    Dropout(0.5),\n",
    "    Dense(128, activation='relu'),\n",
    "    Dense(len(categories), activation='softmax')  # Output layer\n",
    "])\n",
    "\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1002ec42-5f78-45bf-a624-39204684e8b4",
   "metadata": {},
   "source": [
    "Compile the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af0295ca-a90d-4c5d-a397-07d70b8bccc5",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(\n",
    "    loss='categorical_crossentropy',\n",
    "    optimizer='adam',\n",
    "    metrics=['accuracy']\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc853308-7171-44da-8d77-dc486e76ebfb",
   "metadata": {},
   "source": [
    "Train the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e8f7af78-7222-4dcf-a65c-e324d3a4f0e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Without augmentation\n",
    "history = model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.2)\n",
    "\n",
    "# Or with augmentation\n",
    "# history = model.fit(datagen.flow(X_train, y_train, batch_size=32), \n",
    "#                     epochs=25, validation_data=(X_test, y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ce2672a-7545-44db-b26c-aa7acaaa526d",
   "metadata": {},
   "source": [
    "Evaluate the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8672475c-70c7-49f1-9082-1e9ad2c53842",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_loss, test_accuracy = model.evaluate(X_test, y_test)\n",
    "print(\"Test Accuracy:\", test_accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "529689ce-9235-432a-b0b7-3ca5eee7eaf6",
   "metadata": {},
   "source": [
    "Plot Training Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1bca010d-41ec-4074-8929-b10be5712bc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(history.history['accuracy'], label='Train')\n",
    "plt.plot(history.history['val_accuracy'], label='Validation')\n",
    "plt.title('Model Accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.legend()\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(history.history['loss'], label='Train')\n",
    "plt.plot(history.history['val_loss'], label='Validation')\n",
    "plt.title('Model Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.ylabel('Loss')\n",
    "plt.legend()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "988396ec-73fc-4d63-8bf4-2dccccd6e5fa",
   "metadata": {},
   "source": [
    "Generate Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "557a222a-6b3d-48d6-bcba-574b7fe5b0bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Predict and convert one-hot labels to class indices\n",
    "y_pred = model.predict(X_test)\n",
    "y_pred_classes = np.argmax(y_pred, axis=1)\n",
    "y_true = np.argmax(y_test, axis=1)\n",
    "\n",
    "# Get the unique classes actually present in the test set\n",
    "unique_classes = np.unique(np.concatenate((y_true, y_pred_classes)))\n",
    "\n",
    "# Build the confusion matrix only for those classes\n",
    "cm = confusion_matrix(y_true, y_pred_classes, labels=unique_classes)\n",
    "\n",
    "# Use class names only for the unique classes\n",
    "display_labels = [categories[i] for i in unique_classes]\n",
    "\n",
    "# Plot the confusion matrix\n",
    "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)\n",
    "disp.plot(cmap='Blues', xticks_rotation='vertical')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
