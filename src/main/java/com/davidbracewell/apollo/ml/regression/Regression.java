package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.encoder.EncoderPair;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.Counter;
import lombok.Getter;
import lombok.NonNull;

/**
 * <p>Base regression model that produces a real-value for an input instance.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Regression implements Model {
   private static final long serialVersionUID = 1L;
   @Getter
   private final PreprocessorList<Instance> preprocessors;
   private EncoderPair encoderPair;

   public Regression(RegressionLearner learner) {
      this.preprocessors = learner.getPreprocessors().getModelProcessors();
      this.encoderPair = learner.getEncoderPair();
   }


   /**
    * Estimates a real-value based on the input instance.
    *
    * @param instance the instance
    * @return the estimated value
    */
   public final double estimate(@NonNull Instance instance) {
      return estimate(getPreprocessors().apply(instance).toVector(encoderPair));
   }


   /**
    * Estimates a real-value based on the input vector.
    *
    * @param vector the vector
    * @return the estimated value
    */
   public abstract double estimate(NDArray vector);

   @Override
   public EncoderPair getEncoderPair() {
      return encoderPair;
   }

   /**
    * Gets the feature weights for the model.
    *
    * @return the feature weights
    */
   public abstract Counter<String> getFeatureWeights();

}// END OF Regression
