
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, Model
import numpy as np

# Load and preprocess Omniglot dataset
def load_and_preprocess_omniglot():
    dataset = tfds.load('omniglot', as_supervised=True)
    train_ds = dataset['train']
    test_ds = dataset['test']

    def preprocess(image, label):
        image = tf.image.resize(image, (28, 28))  # Resize images to 28x28
        image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
        image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
        return image, label

    train_ds = train_ds.map(preprocess).shuffle(1000).batch(32)
    test_ds = test_ds.map(preprocess).batch(32)
    return train_ds, test_ds

# Define the embedding model (CNN backbone)
def create_embedding_model(input_shape=(28, 28, 1)):
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(128, name="embedding")(x)  # Explicit name for the embedding layer
    return Model(inputs, outputs)

# Define the triplet loss
def triplet_loss(anchor, positive, negative, margin=0.2):
    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    return tf.maximum(positive_dist - negative_dist + margin, 0.0)

# Create the triplet network
def create_triplet_network(embedding_model):
    anchor_input = layers.Input(shape=(28, 28, 1), name='anchor')
    positive_input = layers.Input(shape=(28, 28, 1), name='positive')
    negative_input = layers.Input(shape=(28, 28, 1), name='negative')

    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    model = Model(inputs=[anchor_input, positive_input, negative_input],
                  outputs=[anchor_embedding, positive_embedding, negative_embedding])
    return model

# Training step for the triplet network
@tf.function
def train_step(model, optimizer, anchor, positive, negative):
    with tf.GradientTape() as tape:
        anchor_out, positive_out, negative_out = model([anchor, positive, negative])
        loss = tf.reduce_mean(triplet_loss(anchor_out, positive_out, negative_out))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Prepare data for triplet training
def prepare_triplet_data(dataset):
    images, labels = [], []
    for image, label in dataset.unbatch():
        images.append(image)
        labels.append(label.numpy())
    images = np.array(images)
    labels = np.array(labels)

    def create_triplet(i):
        anchor = images[i]
        positive = images[np.random.choice(np.where(labels == labels[i])[0])]
        negative = images[np.random.choice(np.where(labels != labels[i])[0])]
        return anchor, positive, negative

    triplets = [create_triplet(i) for i in range(len(images))]
    anchors, positives, negatives = zip(*triplets)
    return np.array(anchors), np.array(positives), np.array(negatives)

# Evaluate 5-way 1-shot classification
def evaluate_few_shot(model, dataset, num_classes=5, num_samples=1):
    images, labels = [], []
    for image, label in dataset.unbatch():
        images.append(image)
        labels.append(label.numpy())
    images = np.array(images)
    labels = np.array(labels)

    # Randomly sample N classes and 1 example per class
    unique_labels = np.unique(labels)
    sampled_classes = np.random.choice(unique_labels, num_classes, replace=False)
    support_set = []
    for cls in sampled_classes:
        support_set.append(images[np.where(labels == cls)[0][:num_samples]])

    # For each query image, find the nearest neighbor in embedding space
    correct = 0
    for i, query_image in enumerate(images):
        if labels[i] not in sampled_classes:
            continue
        query_embedding = model.predict(query_image[np.newaxis, ...])  # Corrected here
        distances = [
            np.linalg.norm(query_embedding - model.predict(sup))
            for sup in support_set
        ]
        predicted_class = sampled_classes[np.argmin(distances)]
        if predicted_class == labels[i]:
            correct += 1
    return correct / len(images)

# Main function
def main():
    train_ds, test_ds = load_and_preprocess_omniglot()
    embedding_model = create_embedding_model()
    triplet_model = create_triplet_network(embedding_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Prepare triplet data
    anchors, positives, negatives = prepare_triplet_data(train_ds)

    # Train the model
    for epoch in range(10):
        for step in range(len(anchors) // 32):
            batch_anchors = anchors[step * 32:(step + 1) * 32]
            batch_positives = positives[step * 32:(step + 1) * 32]
            batch_negatives = negatives[step * 32:(step + 1) * 32]
            loss = train_step(triplet_model, optimizer, batch_anchors, batch_positives, batch_negatives)
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

    # Evaluate on 5-way 1-shot task
    accuracy = evaluate_few_shot(embedding_model, test_ds)
    print(f"5-way 1-shot classification accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()