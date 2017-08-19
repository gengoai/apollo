package com.davidbracewell.apollo.ml.classification.nn.slt;

import com.davidbracewell.apollo.linalg.DenseFloatMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.optimization.WeightInitializer;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.conversion.Cast;
import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public abstract class WeightLayer extends Layer {
   private static final long serialVersionUID = 1L;
   protected final Activation activation;
   protected final Matrix weights;
   protected final Matrix bias;
   protected final double l1;
   protected final double l2;

   public WeightLayer(int inputSize, int outputSize, Activation activation, WeightInitializer weightInitializer, double l1, double l2) {
      super(inputSize, outputSize);
      this.activation = activation;
      this.weights = weightInitializer.initialize(DenseFloatMatrix.zeros(outputSize, inputSize));
      this.bias = DenseFloatMatrix.zeros(outputSize);
      this.l1 = l1;
      this.l2 = l2;
   }

   @Override
   public Matrix backward(Matrix input, Matrix output, Matrix delta, double learningRate, int layerIndex) {
      delta.muli(activation.valueGradient(output));
      Matrix dzOut = layerIndex > 0
                     ? weights.transpose().mmul(delta)
                     : null;
      val dw = delta.mmul(input.transpose())
                    .divi(input.numCols());
      val db = delta.rowSums()
                    .divi(input.numCols());
      weights.subi(dw.muli(learningRate));
      bias.subi(db.muli(learningRate));
      if (l1 > 0) {
         //L1 Regularization
         double shrinkage = l1 * learningRate;
         weights.mapi(x -> {
            val xp = FastMath.signum(x) * FastMath.max(0, FastMath.abs(x) - shrinkage);
            if (FastMath.abs(xp) < 1e-9) {
               return 0d;
            }
            return xp;
         });
      }

      return dzOut;
   }

   @Override
   public Matrix forward(Matrix input) {
      return activation.apply(weights.mmul(input).addiColumnVector(bias));
   }

   protected static abstract class WeightLayerBuilder<T extends WeightLayerBuilder> extends LayerBuilder<T> {
      @Getter
      private Activation activation = Activation.SIGMOID;
      @Getter
      private WeightInitializer weightInitializer = WeightInitializer.DEFAULT;
      @Getter
      private double l1 = 0;
      @Getter
      private double l2 = 0;

      public T activation(@NonNull Activation activation) {
         this.activation = activation;
         return Cast.as(this);
      }

      public T l1(double l1) {
         this.l1 = l1;
         return Cast.as(this);
      }

      public T l2(double l2) {
         this.l2 = l2;
         return Cast.as(this);
      }

      public T weightInitializer(@NonNull WeightInitializer weightInitializer) {
         this.weightInitializer = weightInitializer;
         return Cast.as(this);
      }

   }

}// END OF WeightLayer
