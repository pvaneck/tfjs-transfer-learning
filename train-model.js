const tf = require('@tensorflow/tfjs-node');
const util = require("./util.js");


// Fashion-MNIST training & test data
const trainDataUrl = 'file://./fashion-mnist/fashion-mnist_train.csv';
const testDataUrl = 'file://./fashion-mnist/fashion-mnist_test.csv';

// mapping of Fashion-MNIST labels (i.e., T-shirt=0, Trouser=1, etc.)
const labels = [
  'T-shirt/top',
  'Trouser',
  'Pullover',
  'Dress',
  'Coat',
  'Sandal',
  'Shirt',
  'Sneaker',
  'Bag',
  'Ankle boot'
];

// Train a model with a subset of the data
// Use the first n classes
const numOfClasses = 5;

const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 1;

const batchSizeDefault = 100;
const epochsValue = 10;


// Load and transform data.
const loadData = function (dataUrl, batchSize=batchSizeDefault, numElements=-1) {
   // Normalize data values between 0 and 1.
  const normalize = ({xs, ys}) => {
    return {
        xs: Object.values(xs).map(x => x / 255),
        ys: ys.label
    };
  };

  // Transform input array (xs) to 3D tensor.
  // Convert label to one-hot vector.
  const transform = ({xs, ys}) => {
    const zeros = (new Array(numOfClasses)).fill(0);

    return {
        xs: tf.tensor(xs, [imageWidth, imageHeight, imageChannels]),
        ys: tf.tensor1d(zeros.map((z, i) => {
            return i === ys ? 1 : 0;
        }))
    };
  };

  return tf.data
    .csv(dataUrl, {columnConfigs: {label: {isLabel: true}}})
    .map(normalize)
    .filter(f => f.ys < numOfClasses)
    .take(numElements)
    .map(transform)
    .batch(batchSize);
};


// Define the model architecture.
const buildModel = function () {
  const model = tf.sequential();

  // Add the model layers.
  model.add(tf.layers.conv2d({
    inputShape: [imageWidth, imageHeight, imageChannels],
    filters: 8,
    kernelSize: 5,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2
  }));
  model.add(tf.layers.conv2d({
    filters: 16,
    kernelSize: 5,
    padding: 'same',
    activation: 'relu'
  }));
  model.add(tf.layers.maxPooling2d({
    poolSize: 3,
    strides: 3
  }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({
    units: numOfClasses,
    activation: 'softmax'
  }));

  // Compile the model.
  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
};


const run = async function () {

  const trainData = loadData(trainDataUrl);
  const testData = loadData(testDataUrl);
  dataset = await testData.toArray();

  const saveModelPath = 'file://./fashion-mnist-tfjs';

  const model = buildModel();
  model.summary();

  const info = await util.trainModel(model, trainData, epochsValue);
  console.log(info);

  console.log('Evaluating model...');
  await util.evaluateModel(model, testData);

  console.log('Saving model...');
  await model.save(saveModelPath);
}

run();