# Credits. This code has been adapted from :
# https://github.com/adap/flower/tree/main/examples/advanced-tensorflow

from typing import Dict, Optional, Tuple, List
import flwr as fl

import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing import image

import pandas as pd
import edgeimpulse as ei


save_evaluation = True #new feature
save_updated_model = True #new feature

# server address = {IP_ADDRESS}:{PORT}
server_address = "10.46.134.7:5050"

# list with the classes for the image classification
classes = ["head", "hardhat"]
class_labels = {classes: i for i, classes in enumerate(classes)}
number_of_classes = len(classes)

# defining image size, 
# a larger one means more data goes to the model(good thing) but processing time and model size will increase
IMAGE_SIZE = (160, 160)

federatedLearningcounts = 3
local_client_epochs = 20
local_client_batch_size = 8

file_name = "loss_metrics"

class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
        def aggregate_evaluate(
            self,
            server_round,
            results,
            failures,
        ):
            """Aggregate evaluation accuracy using weighted average and write results to .csv file"""

            if not results:
                return None, {}

            # Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            aggregated_accuracy = sum(accuracies) / sum(examples)
            #print(f"AGGREGATION OF CLIENT MODELS EVALUATED")

            if save_evaluation: 
                new_data = {'aggregated_loss': [aggregated_loss], 'aggregated_accuracy': [aggregated_accuracy]}
                df = pd.DataFrame(new_data)

            # Check if the file exists
                if os.path.isfile(file_name):
                # Append new data to the existing CSV file
                    df.to_csv(file_name, mode='a', header=False, index=False)
                else:
                # If the file does not exist, write the header as well
                    df.to_csv(file_name, mode='w', header=True, index=False)
                
                # Return aggregated loss and metrics (i.e., aggregated accuracy)
            return aggregated_loss, {"accuracy": aggregated_accuracy}

        def aggregate_fit(
        self,
        server_round,
        results,
        failures,
        ):
            """Save updated global model after each federeated learning round"""

            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if save_updated_model:

                if aggregated_parameters is not None:
                # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Save aggregated_ndarrays
                    print(f"Saving round {server_round} aggregated_ndarrays...")
                    np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

            return aggregated_parameters, aggregated_metrics


def main() -> None:
    # load and compile model for : server-side parameter initialization, server-side parameter evaluation
    
    # loading and compiling Keras model, choose either MobileNetV2 (faster) or EfficientNetB0. 
    # feel free to add more Keras applications
    # https://keras.io/api/applications/
    """
    Model               MobileNetV2     EfficientNetB0
    Size (MB)           14              29
    Top-1 Accuracy      71.3%           77.1%	
    Top-5 Accuracy      90.1%           93.3%
    Parameters          3.5M            5.3M
    Depth               105             132	
    CPU inference ms    25.9            46.0
    GPU inference ms    3.8             4.9
    """

    """
    # uncomment to load an EfficientNetB0 model
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(160, 160, 3), weights=None, classes=2
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None, #4D tensor output of last convolutional layer
        classes=2,
        classifier_activation="softmax"
    )
    # freeze the layers in the base model so they don't get updated
    base_model.trainable = False

    # define classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output) #average pooling to output of base model
    x = tf.keras.layers.Dense(128, activation='relu')(x) #addes fully connected layer with 128 units (learn higher level frauters from pooled features)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x) #adds final output layer with 2 units with softmax activation fct

    # create the final model
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # print a summary of the model architecture
    #model.summary()
    

        # create strategy
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(
            model.get_weights()),
    )

    # start Flower server (SSL-enabled) for X rounds of federated learning
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=federatedLearningcounts),
        strategy=strategy
    )

def load_dataset():
    # defining the directory with the server's test images. We only use the test images!
    directory = "datasets/dataset_server"
    sub_directories = ["test", "train"]

    loaded_dataset = []
    for sub_directory in sub_directories:
        path = os.path.join(directory, sub_directory)
        images = []
        labels = []

        print("Server dataset loading {}".format(sub_directory))

        for folder in os.listdir(path):
            label = class_labels[folder]

            # iterate through each image in the folder
            for file in os.listdir(os.path.join(path,folder)):
                # get path name of the image
                img_path = os.path.join(os.path.join(path, folder), file)

                # open and resize the image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # append the image and its corresponding label to loaded_dataset
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype= 'float32')
        labels = np.array(labels, dtype= 'int32')

        loaded_dataset.append((images, labels))
    
    return loaded_dataset

#server-side model evaluation after paramter aggregation (using data on server)
def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # load data and model here to avoid the overhead of doing it in `evaluate` itself
    (training_images, training_labels), (test_images, test_labels) = load_dataset()
    print("[Server] test_images shape:", test_images.shape)
    print("[Server] test_labels shape:", test_labels.shape)

    # the `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        print("======= server round %s/%s evaluate() ===== " %(server_round, federatedLearningcounts))
        # update model with the latest parameters
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print("======= server round %s/%s accuracy : %s =======" %(server_round, federatedLearningcounts,accuracy))

        if (server_round == federatedLearningcounts):
            # save the decentralized ML model locally on the server computer
            print("Saving updated model locally..")
            #model.save('saved_models/mobilenetv2.h5')  # save model in .h5 format
            model.save('saved_models/mobilenetv2')      # save model in SavedModel format

            # test the updated model
            test_updated_model(model)
             
        return loss, {"accuracy": accuracy}
    return evaluate

def fit_config(server_round: int):
    # return training configuration dict for each round
    config = {
        "batch_size": local_client_batch_size,
        "local_epochs": local_client_epochs,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    #val_steps = 5 if server_round < 4 else 10
    val_steps = 4
    return {"val_steps": val_steps}

#testing on one image from dataset_test folder (specify path)
def test_updated_model(model):
    # test the model by giving it an image and get its prediction
    test_image_head_path = "datasets/dataset_test/head_5.jpg"
    test_image_head = cv2.imread(test_image_head_path)
    test_image_head = cv2.cvtColor(test_image_head, cv2.COLOR_BGR2RGB)
    test_image_head = cv2.resize(test_image_head, IMAGE_SIZE)

    test_image_hardhat_path = "datasets/dataset_test/hardhat_4.jpg"
    test_image_hardhat = cv2.imread(test_image_hardhat_path)
    test_image_hardhat = cv2.cvtColor(test_image_hardhat, cv2.COLOR_BGR2RGB)
    test_image_hardhat = cv2.resize(test_image_hardhat, IMAGE_SIZE)

    print("Testing the final model on an image.....")
    # chose either test_image_head or test_image_hardhat for the prediction
    image_test_result = model.predict(np.expand_dims(test_image_hardhat, axis=0))
    # print the prediction scores/confidence for each class
    # index 0 = head, index 1 = hardhat
    print(image_test_result[0])

    # an easy trick to see the model's prediction scores ("confidence") for each class
    # we can get the highest score/confidence among all classes
    # map the highest score's index to its class
    highest_prediction_score = max(image_test_result[0])
    highest_prediction_score_index = 0
    for i in range(len(image_test_result[0])):
        if image_test_result[0][i] == highest_prediction_score:
            highest_prediction_score_index = i

    most_confident_class = classes[highest_prediction_score_index]
    print("The model mostly predicted %s with a score/confidence of %s" %(most_confident_class, highest_prediction_score))

    """ Some results after testing the model with a head's image:
    Testing the model on an image.....
    1/1 [==============================] - 3s 3s/step
    [9.992312e-01 7.688053e-04]
    The model mostly predicted head with a score/confidence of 0.9992312
    """

    """ Some results after testing the model with a hardhat's image:
    Testing the model on an image.....
    1/1 [==============================] - 2s 2s/step
    [0.00373875 0.9962612 ]
    The model mostly predicted hardhat with a score/confidence of 0.9962612
    """

if __name__ == "__main__":
    main()
