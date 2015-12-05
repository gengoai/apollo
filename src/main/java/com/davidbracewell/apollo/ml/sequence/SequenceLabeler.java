package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Featurizer;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import lombok.NonNull;

/**
 * The type Sequence labeler.
 *
 * @author David B. Bracewell
 */
public class SequenceLabeler extends Model {
  private final PreprocessorList<Instance> preprocessors;
  private Featurizer featurizer;

  /**
   * Instantiates a new Model.
   *
   * @param labelEncoder   the label encoder
   * @param featureEncoder the feature encoder
   * @param preprocessors  the preprocessors
   */
  public SequenceLabeler(@NonNull Encoder labelEncoder, @NonNull Encoder featureEncoder, PreprocessorList<Instance> preprocessors) {
    super(labelEncoder, featureEncoder);
    this.preprocessors = preprocessors;
  }

  @Override
  protected void finishTraining() {
    super.finishTraining(); //Call super to freeze encoders
    preprocessors.trimToSize(getFeatureEncoder());
  }

  /**
   * Sets featurizer.
   *
   * @param featurizer the featurizer
   */
  public void setFeaturizer(Featurizer featurizer) {
    this.featurizer = featurizer;
  }


}// END OF SequenceLabeler
