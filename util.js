const tf = require('@tensorflow/tfjs-node');


// Train the model against the training data.
const trainModel = async function (model, trainingData, epochs) {
  const options = {
    epochs: epochs,
    verbose: 0,
    callbacks: {
      onEpochBegin: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1} of ${epochs} ...`)
      },
      onEpochEnd: async (epoch, logs) => {
        console.log(`  Training set loss: ${logs.loss.toFixed(4)}`)
        console.log(`  Training set accuracy: ${logs.acc.toFixed(4)}`)
      }
    }
  };
  return await model.fitDataset(trainingData, options);
};


// Verify the model against the test data.
const evaluateModel = async function (model, testingData) {
  const result = await model.evaluateDataset(testingData);
  const testLoss = result[0].dataSync()[0];
  const testAcc = result[1].dataSync()[0];

  console.log(`  Test set loss: ${testLoss.toFixed(4)}`);
  console.log(`  Test set accuracy: ${testAcc.toFixed(4)}`);
};


module.exports = {
    trainModel,
    evaluateModel,
};
