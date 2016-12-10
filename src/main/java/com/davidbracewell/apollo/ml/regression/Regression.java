package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.RegressionLabelEncoder;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.guava.common.base.Preconditions;
import lombok.NonNull;

/**
 * <p>Base regression model that produces a real-value for an input instance.</p>
 *
 * @author David B. Bracewell
 */
public abstract class Regression implements Model {
   private static final long serialVersionUID = 1L;
   private final PreprocessorList<Instance> preprocessors;
   private EncoderPair encoderPair;

   /**
    * Instantiates a new Regression model.
    *
    * @param encoderPair   the encoder pair
    * @param preprocessors the preprocessors
    */
   public Regression(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
      this.encoderPair = encoderPair;
      Preconditions.checkArgument(encoderPair.getLabelEncoder() instanceof RegressionLabelEncoder,
                                  "Regression only allows RealEncoder for labels.");
      this.preprocessors = preprocessors.getModelProcessors();
   }


   /**
    * Estimates a real-value based on the input instance.
    *
    * @param instance the instance
    * @return the estimated value
    */
   public final double estimate(@NonNull Instance instance) {
      return estimate(preprocessors.apply(instance).toVector(getEncoderPair()));
   }


   /**
    * Estimates a real-value based on the input vector.
    *
    * @param vector the vector
    * @return the estimated value
    */
   public abstract double estimate(Vector vector);

   /**
    * Gets the feature weights for the model.
    *
    * @return the feature weights
    */
   public abstract Counter<String> getFeatureWeights();


   @Override
   public EncoderPair getEncoderPair() {
      return encoderPair;
   }

}// END OF Regression
