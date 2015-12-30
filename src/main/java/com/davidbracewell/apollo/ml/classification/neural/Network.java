package com.davidbracewell.apollo.ml.classification.neural;

import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.io.Serializable;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class Network implements Serializable {
  private static final long serialVersionUID = 1L;
  private final Layer[] layers;
  private final EncoderPair encoderPair;

  public Network(@NonNull List<Tuple2<Integer, Activation>> config, @NonNull EncoderPair encoderPair) {
    this.encoderPair = encoderPair;
    this.layers = new Layer[config.size() + 1];
    for (int i = 0; i < layers.length; i++) {
      int inputSize = (i == 0 ? encoderPair.numberOfFeatures() : layers[i - 1].getMatrix().numberOfColumns());
      int outputSize = (i == layers.length - 1 ? encoderPair.numberOfLabels() : config.get(i).v1);
      Activation af = i + 1 == layers.length ? new Sigmoid() : config.get(i).v2;
      layers[i] = new Layer(inputSize, outputSize, af);
    }
  }

  public int numberOfLayers() {
    return layers.length;
  }

  public Classification evaluate(Vector input) {
    return new Classification(forward(input)[layers.length-1].row(0).toArray(), encoderPair.getLabelEncoder());
  }

  public Matrix[] forward(Vector v) {
    Matrix[] outputs = new Matrix[numberOfLayers()];
    Matrix input = v.toMatrix();
    for (int i = 0; i < numberOfLayers(); i++) {
      outputs[i] = layers[i].evaluate(input);
      input = outputs[i];
    }
    return outputs;
  }

  public double backprop(Matrix[] outputs, Vector instance, Vector yVec, double learningRate) {
    Matrix[] deltas = new Matrix[numberOfLayers()];
    double error = 0;
    Matrix input = outputs[outputs.length - 1];
    for (int i = 0; i < encoderPair.numberOfLabels(); i++) {
      error += Math.pow(yVec.get(i) - input.get(0, i), 2);
    }
    deltas[numberOfLayers() - 1] = yVec.toMatrix().subtract(input).scaleSelf(layers[numberOfLayers() - 1].gradient(input));
    for (int i = numberOfLayers() - 2; i >= 0; i--) {
      deltas[i] = deltas[i + 1].multiply(layers[i + 1].getMatrix().transpose()).scaleSelf(layers[i].gradient(outputs[i])).scale(learningRate);
    }
    layers[0].getMatrix().addSelf(instance.transpose().multiply(deltas[0]));
    for (int i = 1; i < numberOfLayers(); i++) {
      layers[i].getMatrix().addSelf(outputs[i - 1].transpose().multiply(deltas[i]));
    }
    return error;
  }


}// END OF Network
