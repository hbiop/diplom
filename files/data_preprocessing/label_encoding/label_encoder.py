import numpy as np

def label_encoder(labels):
    unique_labels = np.unique(labels)
    label_mapping = {label: index for index, label in enumerate(unique_labels)}
    encoded_labels = np.vectorize(label_mapping.get)(labels)
    return encoded_labels, label_mapping


if __name__ == "__main__":
    labels = np.array(['apple', 'banana', 'yellow', 'apple', 'red'])
    encoded_labels, mapping = label_encoder(labels)
    print("Encoded Labels:", encoded_labels)
    print("Mapping:", mapping)