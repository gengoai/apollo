package com.davidbracewell.apollo.ml.classification.nn.slt;

import com.davidbracewell.apollo.linalg.DenseFloatMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.optimization.WeightInitializer;
import com.davidbracewell.apollo.optimization.activation.Activation;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.tuple.Tuple2;
import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import org.apache.commons.math3.util.FastMath;

/**
 * @author David B. Bracewell
 */
public abstract class WeightLayer extends Layer {
   private static final long serialVersionUID = 1L;
   @Getter
   protected final Activation activation;
   protected final double l1;
   protected final double l2;
   protected Matrix weights;
   protected Matrix bias;
   protected transient Matrix v;

   public WeightLayer(int inputSize, int outputSize, Activation activation, WeightInitializer weightInitializer, double l1, double l2) {
      super(inputSize, outputSize);
      this.activation = activation;
      this.weights = weightInitializer.initialize(DenseFloatMatrix.zeros(outputSize, inputSize));
      this.bias = DenseFloatMatrix.zeros(outputSize);
      this.v = DenseFloatMatrix.zeros(outputSize, inputSize);
      this.l1 = l1;
      this.l2 = l2;
   }

   public WeightLayer(WeightLayer layer) {
      super(layer.getInputSize(), layer.getOutputSize());
      this.activation = layer.getActivation();
      this.bias = layer.bias.copy();
      this.weights = layer.weights.copy();
      this.l1 = layer.l1;
      this.l2 = layer.l2;
      this.v = DenseFloatMatrix.zeros(layer.getOutputSize(), layer.getInputSize());
   }


   @Override
   public BackpropResult backward(Matrix input, Matrix output, Matrix delta, boolean calculateDelta) {
      delta.muli(activation.valueGradient(output));
      Matrix dzOut = calculateDelta
                     ? weights.transpose().mmul(delta)
                     : null;
      val dw = delta.mmul(input.transpose());
      val db = delta.rowSums();
      return BackpropResult.from(dzOut, dw, db);
   }

   @Override
   public Tuple2<Matrix, Double> backward(WeightUpdate updater, Matrix input, Matrix output, Matrix delta, int iteration, boolean calcuateDelta) {
      return updater.update(this.weights, this.bias, input, output, delta.muli(activation.valueGradient(output)),
                            iteration, calcuateDelta);
   }

   @Override
   public Matrix backward(Matrix input, Matrix output, Matrix delta, double learningRate, int layerIndex, int iteration) {
      delta.muli(activation.valueGradient(output));
      Matrix dzOut = layerIndex > 0
                     ? weights.transpose().mmul(delta)
                     : null;
      val dw = delta.mmul(input.transpose())
                    .divi(input.numCols());
      val db = delta.rowSums()
                    .divi(input.numCols());
      v.muli(0.9).subi(dw.muli(learningRate));
      weights.addi(v);
      bias.subi(db.muli(learningRate));
      l1Update(learningRate, iteration);
      return dzOut;
   }

   @Override
   public Matrix forward(Matrix input) {
      return activation.apply(weights.mmul(input).addiColumnVector(bias));
   }

   @Override
   public Matrix getBias() {
      return bias;
   }

   @Override
   public Matrix getWeights() {
      return weights;
   }

   protected void l1Update(double learningRate, int iteration) {
      if (l1 > 0) {
         //L1 Regularization
         double shrinkage = l1 * (learningRate / iteration);
         weights.mapi(x -> {
            val xp = FastMath.signum(x) * FastMath.max(0, FastMath.abs(x) - shrinkage);
            if (FastMath.abs(xp) < 1e-9) {
               return 0d;
            }
            return xp;
         });
      }
   }

   @Override
   public double update(WeightUpdate weightUpdate, Matrix wGrad, Matrix bBrad, int iteration) {
      return weightUpdate.update(this.weights, this.bias, wGrad, bBrad, iteration);
   }

   @Override
   public void update(Matrix[] weights, Matrix[] bias) {
      Matrix wP = this.weights.getFactory().zeros(getOutputSize(),getInputSize());
      Matrix bP = this.weights.getFactory().zeros(getOutputSize());
      for (int i = 0; i < weights.length; i++) {
         wP.addi(weights[i]);
         bP.addi(bias[i]);
      }
      if( weights.length > 0 ) {
         wP.divi(weights.length);
         bP.divi(weights.length);
         this.weights = wP;
         this.bias = bP;
      }
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
