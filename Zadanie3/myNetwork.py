import numpy as np
import mnist

def linear(z):
    return z

def ReLU(z):
    return T.maximum(0.0, z)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def cost_derivative(a_L, y):
    return a_L - y

class Conv3x3:
    # A Convolution layer using 3x3 filters.
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.biases = np.random.randn(num_filters)
        self.weights = np.array([np.random.randn(3, 3)/9
                        for x in range(num_filters)])
        
    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, inpt):
        h, w = inpt.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        self.last_input = inpt

        for it in range(self.num_filters):
            for im_region, i, j in self.iterate_regions(inpt):
                output[i, j, it] = np.sum(im_region * self.weights[it])
        return output

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_weights = np.zeros(self.weights.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_weights[f] += d_L_d_out[i, j, f] * im_region

        self.weights -= learn_rate * d_L_d_weights

class MaxPool2:

    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, inpt):
        h, w, num_filters = inpt.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        self.last_input = inpt

        for im_region, i, j in self.iterate_regions(inpt):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, inpt):
        self.last_input_shape = inpt.shape

        inpt = inpt.flatten()
        self.last_input = inpt

        input_len, nodes = self.weights.shape

        totals = np.dot(inpt, self.weights) + self.biases
        self.last_totals = totals

        return np.exp(totals)/np.sum(np.exp(totals), axis = 0)

    def backprop(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            return d_L_d_inputs.reshape(self.last_input_shape)


def forward(image, label):
    out = conv.forward(image / 255)
    out = maxPool.forward(out)
    out = softMax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, label, lr=.005):
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softMax.backprop(gradient, lr)
    gradient = maxPool.backprop(gradient)
    conv.backprop(gradient, lr)

    return loss, acc


train_images = mnist.train_images()[:5000]
train_labels = mnist.train_labels()[:5000]

test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]

conv = Conv3x3(8)
maxPool = MaxPool2()
softMax = Softmax(13 * 13 * 8, 10)

for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train!
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0

        l, acc = train(im, label, 0.02)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

'''

out = conv.forward((image / 255) - 0.5)
out = pool.forward(out)
out = softmax.forward(out)

# Calculate initial gradient
gradient = np.zeros(10)
# ...

# Backprop
gradient = softmax.backprop(gradient)
gradient = pool.backprop(gradient)
gradient = conv.backprop(gradient)
'''