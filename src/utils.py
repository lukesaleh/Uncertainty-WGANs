import matplotlib.pyplot as plt

# display examples from xray dataset from huggingface
def display_xray_examples(dataset, num_examples=6):
    plt.figure(figsize=(15, 5))
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(dataset[i]['image'], cmap='gray')
        plt.axis('off')
    plt.show()