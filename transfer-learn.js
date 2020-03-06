const tf = require('@tensorflow/tfjs-node');
const util = require("./util.js");


// Fashion-MNIST training & test data
const trainDataUrl = 'file://./fashion-mnist/fashion-mnist_train.csv';
const testDataUrl = 'file://./fashion-mnist/fashion-mnist_test.csv';

// Mapping of Fashion-MNIST labels (i.e., T-shirt=0, Trouser=1, etc.)
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

// Train a model with a subset of the data.
// Use the first n classes.
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
            return i === (ys - numOfClasses) ? 1 : 0;
        }))
    };
  };

  return tf.data
    .csv(dataUrl, {columnConfigs: {label: {isLabel: true}}})
    .map(normalize)
    .filter(f => f.ys >= (labels.length - numOfClasses))
    .take(numElements)
    .map(transform)
    .batch(batchSize);
};


// Define the model architecture.
const buildModel = function (baseModel) {

  // Remove the last layer of the base model. This is the softmax
  // classification layer used for classifying the first 5 classes
  // of Fashion-MNIST. This leaves us with the 'Flatten' layer as the
  // new final layer.
  baseModel.layers.pop();

  // Freeze the weights in the base model layers (feature layers) so they
  // don't change when we train the new model.
  for (layer of baseModel.layers) {
    layer.trainable = false;
  }

  // Create a new sequential model starting from the layers of the
  // previous model.
  const model = tf.sequential({
    layers: baseModel.layers
  });

  // Add a new softmax dense layer. This layer will have the trainable
  // parameters for classifying our new classes.
  model.add(tf.layers.dense({
    units: numOfClasses,
    activation: 'softmax',
    name: 'topSoftmax'
  }));

  model.compile({
    optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
  });

  return model;
}


const run = async function () {

  const trainData = loadData(trainDataUrl, batchSizeDefault, 3000);
  const testData = loadData(testDataUrl);

  const baseModelUrl = 'file://./fashion-mnist-tfjs/model.json';
  const saveModelPath = 'file://./fashion-mnist-tfjs-transfer';

  const baseModel =  await tf.loadLayersModel(baseModelUrl);
  const newModel = buildModel(baseModel);
  newModel.summary();

  const info = await util.trainModel(newModel, trainData, epochsValue);
  console.log(info);

  console.log('Evaluating model...');
  await util.evaluateModel(newModel, testData);

  console.log('Saving model...');
  await newModel.save(saveModelPath);
}

run();
