package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.RealEncoder;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.counter.Counter;
import com.google.common.base.Preconditions;
import lombok.NonNull;

/**
 * The type Regression.
 *
 * @author David B. Bracewell
 */
public abstract class Regression implements Model {
  private static final long serialVersionUID = 1L;
  private final PreprocessorList<Instance> preprocessors;
  private EncoderPair encoderPair;

  /**
   * Instantiates a new Regression.
   *
   * @param encoderPair   the encoder pair
   * @param preprocessors the preprocessors
   */
  public Regression(@NonNull EncoderPair encoderPair, @NonNull PreprocessorList<Instance> preprocessors) {
    this.encoderPair = encoderPair;
    Preconditions.checkArgument(encoderPair.getLabelEncoder() instanceof RealEncoder, "Regression only allows RealEncoder for labels.");
    this.preprocessors = preprocessors.getModelProcessors();
  }


  /**
   * Estimate double.
   *
   * @param instance the instance
   * @return the double
   */
  public final double estimate(@NonNull Instance instance) {
    return estimate(preprocessors.apply(instance).toVector(getEncoderPair()));
  }


  /**
   * Estimate double.
   *
   * @param vector the vector
   * @return the double
   */
  public abstract double estimate(Vector vector);

  /**
   * Gets feature weights.
   *
   * @return the feature weights
   */
  public abstract Counter<String> getFeatureWeights();


  @Override
  public EncoderPair getEncoderPair() {
    return encoderPair;
  }

}// END OF Regression
