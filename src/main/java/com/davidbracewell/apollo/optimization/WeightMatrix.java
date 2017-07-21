package com.davidbracewell.apollo.optimization;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.Getter;
import lombok.NonNull;

import java.io.Serializable;
import java.util.stream.IntStream;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * The type Weight matrix.
 *
 * @author David B. Bracewell
 */
public class WeightMatrix implements Serializable {
   private static final long serialVersionUID = 1L;
   private final Vector[] weights;
   private final Vector biases;
   @Getter
   private final int numberOfLabels;
   @Getter
   private final int numberOfFeatures;
   @Getter
   private final boolean isBinary;

   /**
    * Instantiates a new Weight matrix.
    *
    * @param numLabels   the num labels
    * @param numFeatures the num features
    */
   public WeightMatrix(int numLabels, int numFeatures) {
      Preconditions.checkArgument(numLabels > 1, "Must have at least two labels");
      this.isBinary = (numLabels <= 2);
      this.numberOfLabels = numLabels;
      this.numberOfFeatures = numFeatures;


      this.weights = new Vector[numberOfWeightVectors()];
      this.biases = Vector.sZeros(numberOfWeightVectors());
      for (int i = 0; i < numberOfWeightVectors(); i++) {
         this.weights[i] = Vector.sZeros(numFeatures);
      }

   }

   /**
    * Add.
    *
    * @param gradientMatrix the gradient matrix
    */
   public void add(@NonNull GradientMatrix gradientMatrix) {
      for (int i = 0; i < numberOfWeightVectors(); i++) {
         this.weights[i].addSelf(gradientMatrix.get(i).getWeightGradient());
         this.biases.increment(i, gradientMatrix.get(i).getBiasGradient());
      }
   }

   public Vector backward(@NonNull Vector input) {
      Vector output = Vector.dZeros(numberOfFeatures);
      IntStream.range(0, numberOfFeatures)
               .parallel()
               .mapToObj(f -> {
                  double v = 0;
                  for (int l = 0; l < numberOfWeightVectors(); l++) {
                     v += input.get(l) * weights[l].get(f);
                  }
                  return $(f, v);
               })
               .forEach(p -> output.set(p.v1, p.v2));
      return output;
   }

   /**
    * Binary dot vector.
    *
    * @param input      the input
    * @param activation the activation
    * @return the vector
    */
   public Vector binaryDot(@NonNull Vector input, @NonNull Activation activation) {
      return new DenseVector(1)
                .set(0, activation.apply(weights[0].dot(input) + biases.get(0)));
   }

   /**
    * Dot vector.
    *
    * @param input      the input
    * @param activation the activation
    * @return the vector
    */
   public Vector dot(@NonNull Vector input, @NonNull Activation activation) {
      Vector output = Vector.sZeros(numberOfLabels);
      if (isBinary) {
         output.set(1, weights[0].dot(input) + biases.get(0));
      } else {
         for (int i = 0; i < numberOfWeightVectors(); i++) {
            output.set(i, weights[i].dot(input) + biases.get(i));
         }
      }
      output = activation.apply(output);

      if (isBinary && activation.isProbabilistic()) {
         output.set(0, 1d - output.get(1));
      } else if (isBinary) {
         output.set(0, -output.get(0));
      }

      return output;
   }

   /**
    * Gets bias.
    *
    * @param i the
    * @return the bias
    */
   public double getBias(int i) {
      return biases.get(i);
   }

   /**
    * Gets weight vector.
    *
    * @param i the
    * @return the weight vector
    */
   public Vector getWeightVector(int i) {
      return weights[i];
   }

   /**
    * Number of weight vectors int.
    *
    * @return the int
    */
   public int numberOfWeightVectors() {
      return isBinary ? 1 : numberOfLabels;
   }

   /**
    * Subtract.
    *
    * @param gradientMatrix the gradient matrix
    */
   public void subtract(@NonNull GradientMatrix gradientMatrix) {
      for (int i = 0; i < numberOfWeightVectors(); i++) {
         this.weights[i].subtractSelf(gradientMatrix.get(i).getWeightGradient());
         this.biases.decrement(i, gradientMatrix.get(i).getBiasGradient());
      }
   }

}// END OF WeightMatrix2
